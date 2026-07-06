# -*- coding: utf-8 -*-
"""Replay completed sample-size checkpoints on eval_n20000.

This is a small orchestration helper for the train/eval sample-count matrix:

- train cap: read from completed samplecap/sampletail rows
- eval cap : 200, 2000, 20000 on the same E:/data/images/eval_n20000 root

It waits until eval_n20000 has all required classes, then calls
checkpoint_replay with separate train/eval diagnostic caps so pcls reports
match the requested per-class counts.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE_LEADERBOARD = REPO / "outputs" / "frozen_iter116J_orig814_v15direct_n2000" / "_leaderboard.csv"
DEST_LEADERBOARD = REPO / "outputs" / "frozen_iter116J_orig814_eval_n20000" / "_leaderboard.csv"
EVAL_ROOT = Path("E:/data/images/eval_n20000")
KEEP_FAMILY_SUBSTRINGS = (
    # Current sample-count question is answered on the active best family.
    # The baseline samplecap family repeatedly showed negative gap / high FAR,
    # so replaying it at 20k delays the improvement loop without changing the
    # next decision.
    "cmp10000_p05000_ab090_100",
    # Nearby high-POS families requested for tail/gap follow-up once their
    # train=200 rows finish in the source sweep.
    "cmp09500_p04500_ab090_100",
    "cmp10000_p04500_ab090_100",
    "cmp10000_p05000_ab080_100",
    "cmp09500_p05000_ab090_100",
)
REQUIRED_CLASSES = (
    "bank_boundary",
    "fork",
    "scratch",
    "scratch_rot",
    "bank_boundary+fork",
    "bank_boundary+scratch",
    "bank_boundary+scratch_rot",
    "fork+scratch",
    "fork+scratch_rot",
    "scratch+scratch_rot",
    "Normal",
    "Invalid",
    "CenterDonut",
    "CrossScratch",
    "DiagonalSmear",
    "Starburst",
)


def class_counts(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for name in REQUIRED_CLASSES:
        d = root / name
        out[name] = len([p for p in d.iterdir() if p.is_file()]) if d.exists() else 0
    return out


def is_complete(root: Path, n_per_class: int) -> bool:
    counts = class_counts(root)
    return all(counts.get(name, 0) >= n_per_class for name in REQUIRED_CLASSES)


def source_rows() -> list[dict[str, str]]:
    if not SOURCE_LEADERBOARD.exists():
        return []
    with SOURCE_LEADERBOARD.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        tag = row.get("tag", "")
        train_cap = row.get("train_cap_per_class", "")
        model = row.get("model", "")
        if row.get("status") != "done":
            continue
        if not tag.startswith(("samplecap_", "sampletail_")) or not tag.endswith("_ev00200"):
            continue
        if KEEP_FAMILY_SUBSTRINGS and not any(s in tag for s in KEEP_FAMILY_SUBSTRINGS):
            continue
        if train_cap not in {"50", "100", "200", "300", "400"}:
            continue
        if not model or not Path(model).exists():
            continue
        key = f"{tag}|{model}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    def score(row: dict[str, str]) -> tuple[float, float, float, int, str]:
        try:
            gap = float(row.get("eval_global_gap", "-999") or "-999")
        except ValueError:
            gap = -999.0
        try:
            f1 = float(row.get("eval_bit_F1", "0") or "0")
        except ValueError:
            f1 = 0.0
        try:
            far = float(row.get("eval_Total_FAR", "999") or "999")
        except ValueError:
            far = 999.0
        train_cap = int(row.get("train_cap_per_class", "0") or "0")
        return (-gap, -f1, far, -train_cap, row.get("tag", ""))

    return sorted(out, key=score)


def tag_done(tag: str) -> bool:
    if not tag:
        return True
    for row in source_rows():
        if row.get("tag") == tag and row.get("status") == "done":
            return True
    return False


def replay_done(base_tag: str, train_cap: int, eval_cap: int) -> bool:
    if not DEST_LEADERBOARD.exists():
        return False
    prefix = f"replay_{base_tag}_evaln{eval_cap}_"
    with DEST_LEADERBOARD.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "done":
                continue
            if not row.get("tag", "").startswith(prefix):
                continue
            if row.get("eval_n_per_class") != str(eval_cap):
                continue
            if row.get("train_eval_n_per_class") != str(train_cap):
                continue
            return True
    return False


def _newest_preds_under(root: Path) -> Path | None:
    if not root.exists():
        return None
    for subdir in ("eval_best", "eval_best_n2000", "eval_external", "eval"):
        candidates = [p for p in (root / subdir).glob("**/preds_chip.parquet") if p.exists()]
        if candidates:
            return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    candidates: list[Path] = []
    candidates.extend(
        p for p in root.glob("eval*/**/preds_chip.parquet")
        if "train" not in str(p).lower()
    )
    candidates = [p for p in candidates if p.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def source_eval_preds(row: dict[str, str]) -> Path | None:
    roots: list[Path] = []
    perf = row.get("performance_report", "")
    if perf:
        roots.append(Path(perf).parent)
    out_dir = row.get("out_dir", "")
    if out_dir:
        roots.append(Path(out_dir))
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        preds = _newest_preds_under(root)
        if preds:
            return preds
    return None


def stop_processes_matching(pattern: str) -> None:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -match '{pattern}' }} | "
        "ForEach-Object { try { Stop-Process -Id $_.ProcessId -Force } catch {} }"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        cwd=str(REPO),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def maybe_pause_sweep(args: argparse.Namespace) -> None:
    if not args.pause_sweep_after_tag:
        return
    if tag_done(args.pause_sweep_after_tag):
        print(f"[sample-replay] pause sweep after done tag={args.pause_sweep_after_tag}", flush=True)
        stop_processes_matching("chip_multilabel.recipe_sweep")
        stop_processes_matching("_train_chip_variant|chip_multilabel.run_stage1|_posneg_prob_diag")
        args.pause_sweep_after_tag = ""


def known_gpu_child_active() -> bool:
    ps = (
        "Get-CimInstance Win32_Process -Filter \"name = 'python.exe'\" | "
        "Where-Object { $_.CommandLine -match '_train_chip_variant|chip_multilabel.run_stage1|_posneg_prob_diag' -and "
        "$_.CommandLine -notmatch 'checkpoint_replay|eval_n20000_sample_replay' } | "
        "Select-Object -First 1 -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    return bool(result.stdout.strip())


def replay(row: dict[str, str], eval_cap: int, args: argparse.Namespace) -> int:
    train_cap = int(row["train_cap_per_class"])
    base_tag = row["tag"]
    name = f"{base_tag}_evaln{eval_cap}"
    model = row["model"]
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "chip_multilabel.checkpoint_replay",
        "--dataset",
        "frozen_iter116J_orig814_eval_n20000",
        "--model",
        f"{name}={model}",
        "--max-models",
        "1",
        "--eval-n-per-class",
        str(eval_cap),
        "--train-eval-n-per-class",
        str(train_cap),
        "--train-diag-cap",
        str(train_cap),
        "--eval-diag-cap",
        str(eval_cap),
        "--diag-device",
        args.diag_device,
        "--skip-invalid-heuristic",
    ]
    calib_preds = source_eval_preds(row)
    if calib_preds:
        cmd.extend(["--nb-calib-preds", str(calib_preds)])
    print(f"[sample-replay] START train={train_cap} eval={eval_cap} tag={base_tag}", flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def refresh_sample_report() -> None:
    cmd = [
        sys.executable,
        "-m",
        "chip_multilabel.sample_count_matrix_report",
        "--no-include-best",
    ]
    subprocess.run(cmd, cwd=str(REPO))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=20000)
    ap.add_argument("--eval-caps", default="200,2000,20000")
    ap.add_argument("--poll-seconds", type=int, default=120)
    ap.add_argument("--diag-device", default="cuda")
    ap.add_argument("--no-wait-idle", action="store_true")
    ap.add_argument("--pause-sweep-after-tag", default="")
    args = ap.parse_args()
    eval_caps = [int(x.strip()) for x in args.eval_caps.split(",") if x.strip()]

    while not is_complete(EVAL_ROOT, args.n_per_class):
        counts = class_counts(EVAL_ROOT)
        short = ", ".join(f"{k}:{v}" for k, v in counts.items() if v < args.n_per_class)
        print(f"[sample-replay] wait eval_n20000 incomplete: {short}", flush=True)
        time.sleep(args.poll_seconds)

    print("[sample-replay] eval_n20000 complete", flush=True)
    processed: set[tuple[str, int]] = set()
    while True:
        rows = source_rows()
        if not rows:
            print("[sample-replay] no completed samplecap rows yet", flush=True)
            return
        pending: list[tuple[dict[str, str], int]] = []
        for row in rows:
            train_cap = int(row.get("train_cap_per_class", "0") or "0")
            for eval_cap in eval_caps:
                key = (row.get("tag", ""), eval_cap)
                if replay_done(row.get("tag", ""), train_cap, eval_cap):
                    print(
                        f"[sample-replay] SKIP done train={train_cap} eval={eval_cap} tag={row.get('tag', '')}",
                        flush=True,
                    )
                    processed.add(key)
                elif key not in processed:
                    pending.append((row, eval_cap))
        if not pending:
            return
        row, eval_cap = pending[0]
        maybe_pause_sweep(args)
        while not args.no_wait_idle and known_gpu_child_active():
            print("[sample-replay] wait GPU child idle before replay", flush=True)
            time.sleep(args.poll_seconds)
            maybe_pause_sweep(args)
        rc = replay(row, eval_cap, args)
        key = (row.get("tag", ""), eval_cap)
        train_cap = int(row.get("train_cap_per_class", "0") or "0")
        if rc == 0 or replay_done(row.get("tag", ""), train_cap, eval_cap):
            processed.add(key)
            refresh_sample_report()
        else:
            print(f"[sample-replay] replay failed rc={rc} tag={row.get('tag')} eval={eval_cap}", flush=True)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
