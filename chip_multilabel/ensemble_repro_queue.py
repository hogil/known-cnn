# -*- coding: utf-8 -*-
"""Recover frozen-original zero-FAR vote ensembles with full reports.

This sidecar is intentionally separate from the training sweep.  It waits for
the current sweep recipe to finish, pauses the sweep parent, regenerates train
predictions for recoverable historical ensemble members, writes train/eval
per-class probability reports, then restarts the forever sweep.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from .recipe_sweep import DATASETS


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = "E:/data/images/classification_chips"
EVAL_ROOT = "E:/data/images/chip_multilabel_v15direct_n2000"
SNAPSHOT_DATASET = "frozen_original_200_snapshot"
SNAPSHOT_TRAIN_ROOT = "E:/data/images/classification_chips_200_snapshot_260529"
OUT_ROOT = ROOT / "outputs" / "frozen_original"
PYTHON = sys.executable


@dataclass(frozen=True)
class Member:
    name: str
    ckpt: str
    eval_parquet: str


MEMBERS = {
    "iter116J_s1": Member(
        "iter116J_s1",
        "outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/best_model.pth",
        "outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/eval_n2000_pred/stage1_260514_161529/preds_chip.parquet",
    ),
    "iter116J_clone_s77": Member(
        "iter116J_clone_s77",
        "outputs/iter116J_clone_s77/20260517_091330_T7_iter116J_clone_s77/best_model.pth",
        "outputs/iter116J_clone_s77/20260517_091330_T7_iter116J_clone_s77/eval_n2000_pred/stage1_260517_092932/preds_chip.parquet",
    ),
    "iter116J_g3_ls20_s77_v17": Member(
        "iter116J_g3_ls20_s77_v17",
        "outputs/iter116J_g3_ls20_s77_v17/20260518_115424_T7_iter116J_g3_ls20_s77/best_model.pth",
        "outputs/iter116J_g3_ls20_s77_v17/20260518_115424_T7_iter116J_g3_ls20_s77/eval_n2000_pred/stage1_260518_120545/preds_chip.parquet",
    ),
    "KD_v7_iter116J_a03_T2_skipcutmix": Member(
        "KD_v7_iter116J_a03_T2_skipcutmix",
        "outputs/KD_v7_iter116J_a03_T2_skipcutmix/20260517_095713_T7_KD_v7_iter116J_a03_T2_skipcutmix/best_model.pth",
        "outputs/KD_v7_iter116J_a03_T2_skipcutmix/20260517_095713_T7_KD_v7_iter116J_a03_T2_skipcutmix/eval_n2000_pred/stage1_260517_101336/preds_chip.parquet",
    ),
    "KD_v12_a030_T3_skipcm_v15": Member(
        "KD_v12_a030_T3_skipcm_v15",
        "outputs/KD_v12_a030_T3_skipcm_v15/20260518_044903_T7_KD_v12_a030_T3_skipcm/best_model.pth",
        "outputs/KD_v12_a030_T3_skipcm_v15/20260518_044903_T7_KD_v12_a030_T3_skipcm/eval_n2000_pred/stage1_260518_045541/preds_chip.parquet",
    ),
}


ENSEMBLES = [
    (
        "ens_5way_E22_KD_Tdiversity_NEW_CHAMPION",
        2,
        [
            "iter116J_s1",
            "iter116J_clone_s77",
            "iter116J_g3_ls20_s77_v17",
            "KD_v7_iter116J_a03_T2_skipcutmix",
            "KD_v12_a030_T3_skipcm_v15",
        ],
    ),
    (
        "ens_4way_3strong_KDv7_LS20s77_FINAL_CHAMPION",
        2,
        [
            "iter116J_s1",
            "iter116J_clone_s77",
            "iter116J_g3_ls20_s77_v17",
            "KD_v7_iter116J_a03_T2_skipcutmix",
        ],
    ),
    (
        "ensemble_3stud_v8_vote_majority_bits",
        2,
        [
            "iter116J_s1",
            "iter116J_clone_s77",
            "KD_v7_iter116J_a03_T2_skipcutmix",
        ],
    ),
]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TORCH_NUM_THREADS": "2",
            "PYTHONIOENCODING": "utf-8",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    return env


def _cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return ""


def _python_procs() -> list[psutil.Process]:
    out: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name"]):
        if "python" not in (proc.info.get("name") or "").lower():
            continue
        out.append(proc)
    return out


def _matching(*markers: str) -> list[psutil.Process]:
    return [p for p in _python_procs() if all(m in _cmdline(p) for m in markers)]


def _done(tag: str, leaderboard: Path) -> bool:
    if not leaderboard.exists():
        return False
    with leaderboard.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("tag") == tag and row.get("status") == "done":
                return True
    return False


def _terminate(procs: list[psutil.Process], log) -> None:
    current = os.getpid()
    targets = [p for p in procs if p.pid != current]
    for proc in targets:
        try:
            log.write(f"[ensemble-queue] terminate pid={proc.pid} cmd={_cmdline(proc)}\n")
            log.flush()
            proc.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    gone, alive = psutil.wait_procs(targets, timeout=10)
    for proc in alive:
        try:
            log.write(f"[ensemble-queue] kill pid={proc.pid} cmd={_cmdline(proc)}\n")
            log.flush()
            proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass


def _run(cmd: list[str], log_path: Path, log) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.write(f"[ensemble-queue] RUN {' '.join(cmd)}\n")
    log.flush()
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=_env(), stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")


def _newest_parquet(out_dir: Path) -> Path | None:
    hits = sorted(out_dir.rglob("preds_chip.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _pred_parquet(
    member: Member,
    split: str,
    root: str,
    out_root: Path,
    n_per_class: int,
    log,
    *,
    reuse_historical_eval: bool = True,
) -> Path:
    if split == "eval" and reuse_historical_eval and root == EVAL_ROOT:
        parquet = ROOT / member.eval_parquet
        if parquet.exists():
            return parquet

    out_dir = out_root / "_ensemble_member_preds" / split / member.name
    existing = _newest_parquet(out_dir)
    if existing:
        return existing
    ckpt = ROOT / member.ckpt
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    batch_size = os.environ.get("CHIP_ENSEMBLE_EVAL_BATCH_SIZE", os.environ.get("CHIP_EVAL_BATCH_SIZE", "96"))
    cmd = [
        PYTHON,
        "-u",
        "-m",
        "chip_multilabel.run_stage1",
        "--model",
        str(ckpt),
        "--eval-set",
        root,
        "--out-root",
        str(out_dir),
        "--variants",
        "I10",
        "--n-per-class",
        str(n_per_class),
        "--batch-size",
        batch_size,
        "--num-workers",
        "0",
        "--strength-min",
        "0.0",
        "--strength-max",
        "1.0",
        "--seed",
        "42",
        "--skip-invalid-heuristic",
    ]
    _run(cmd, out_dir / f"run_stage1_{split}.log", log)
    parquet = _newest_parquet(out_dir)
    if parquet is None:
        raise FileNotFoundError(f"no {split} preds_chip.parquet under {out_dir}")
    return parquet


def _ensemble_report(
    tag: str,
    k: int,
    names: list[str],
    train_paths: dict[str, Path],
    eval_paths: dict[str, Path],
    dataset: str,
    train_root: str,
    eval_root: str,
    out_root: Path,
    leaderboard: Path,
    log,
) -> None:
    out_dir = out_root / tag
    if (out_dir / "performance_report.md").exists() and _done(tag, leaderboard):
        log.write(f"[ensemble-queue] SKIP done tag={tag}\n")
        log.flush()
        return
    train_parquets = [str(train_paths[n]) for n in names]
    eval_parquets = [str(eval_paths[n]) for n in names]
    common = [
        "--tag",
        tag,
        "--cell",
        "T0__I10",
        "--k",
        str(k),
        "--out-dir",
        str(out_dir),
        "--dataset",
        dataset,
        "--train-root",
        train_root,
        "--eval-root",
        eval_root,
        "--leaderboard",
        str(leaderboard),
    ]
    _run(
        [
            PYTHON,
            "-u",
            "-m",
            "chip_multilabel.ensemble_vote_report",
            "--split",
            "train",
            "--root",
            train_root,
            "--parquets",
            *train_parquets,
            *common,
        ],
        out_dir / "ensemble_train.log",
        log,
    )
    _run(
        [
            PYTHON,
            "-u",
            "-m",
            "chip_multilabel.ensemble_vote_report",
            "--split",
            "eval",
            "--root",
            eval_root,
            "--parquets",
            *eval_parquets,
            *common,
        ],
        out_dir / "ensemble_eval.log",
        log,
    )


def _launch(cmd: list[str], stdout: Path, stderr: Path) -> int:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    env = _env()
    with stdout.open("ab") as out, stderr.open("ab") as err:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=err,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    return int(proc.pid)


def _restart_loop(out_root: Path, restart_datasets: str, restart_watchdog: bool, log) -> None:
    sweep_pid = _launch(
        [
            PYTHON,
            "-u",
            "-m",
            "chip_multilabel.recipe_sweep",
            "--datasets",
            restart_datasets,
            "--forever",
            "--diag-device",
            "cuda",
            "--eval-n-per-class",
            "2000",
            "--train-eval-n-per-class",
            "2000",
            "--diag-cap",
            "2000",
        ],
        out_root / "_sota_loop_stdout.log",
        out_root / "_sota_loop_stderr.log",
    )
    watch_pid = ""
    if restart_watchdog:
        watch_pid = str(
            _launch(
                [PYTHON, "-u", "-m", "chip_multilabel.sota_loop_watchdog", "--interval", "300"],
                out_root / "_watchdog_stdout.log",
                out_root / "_watchdog_stderr.log",
            )
        )
    log.write(f"[ensemble-queue] restarted sweep={sweep_pid} watchdog={watch_pid}\n")
    log.flush()


def _run_snapshot_repro(log) -> None:
    if not Path(SNAPSHOT_TRAIN_ROOT).exists():
        log.write(f"[ensemble-queue] skip snapshot repro, missing {SNAPSHOT_TRAIN_ROOT}\n")
        log.flush()
        return
    _run(
        [
            PYTHON,
            "-u",
            "-m",
            "chip_multilabel.recipe_sweep",
            "--datasets",
            SNAPSHOT_DATASET,
            "--max-recipes",
            "1",
            "--diag-device",
            "cuda",
            "--eval-n-per-class",
            "2000",
            "--train-eval-n-per-class",
            "2000",
            "--diag-cap",
            "2000",
        ],
        ROOT / "outputs" / SNAPSHOT_DATASET / "_snapshot_repro.log",
        log,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-tag", default="iter116J_exact_newest200_T7_LS030_g3_cmp05_p025_s1_ep10")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--dataset", default="frozen_original")
    ap.add_argument("--train-root", default="")
    ap.add_argument("--eval-root", default="")
    ap.add_argument("--out-root", default="")
    ap.add_argument("--leaderboard", default="")
    ap.add_argument("--n-per-class", type=int, default=2000)
    ap.add_argument("--restart-datasets", default="")
    ap.add_argument("--no-restart", action="store_true")
    ap.add_argument("--no-watchdog", action="store_true")
    ap.add_argument("--skip-snapshot", action="store_true")
    ap.add_argument("--no-reuse-historical-eval", action="store_true")
    args = ap.parse_args()

    ds = DATASETS.get(args.dataset)
    train_root = args.train_root or (ds.train if ds is not None else TRAIN_ROOT)
    eval_root = args.eval_root or (ds.eval if ds is not None else EVAL_ROOT)
    out_root = Path(args.out_root) if args.out_root else ROOT / "outputs" / args.dataset
    leaderboard = Path(args.leaderboard) if args.leaderboard else out_root / "_leaderboard.csv"
    restart_datasets = args.restart_datasets or args.dataset
    reuse_historical_eval = not args.no_reuse_historical_eval

    out_root.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "_ensemble_repro_queue.log"
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(
            f"[ensemble-queue] start {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"dataset={args.dataset} train={train_root} eval={eval_root} wait_tag={args.wait_tag}\n"
        )
        log.flush()
        while args.wait_tag and not _done(args.wait_tag, leaderboard):
            log.write(f"[ensemble-queue] waiting tag={args.wait_tag}\n")
            log.flush()
            time.sleep(args.poll_seconds)

        _terminate(_matching("chip_multilabel.sota_loop_watchdog"), log)
        _terminate(_matching("chip_multilabel.recipe_sweep", args.dataset), log)
        _terminate(
            _matching(args.dataset, "chip_multilabel._train_chip_variant")
            + _matching(args.dataset, "chip_multilabel.run_stage1")
            + _matching(args.dataset, "chip_multilabel._posneg_prob_diag"),
            log,
        )

        train_paths = {
            name: _pred_parquet(member, "train", train_root, out_root, args.n_per_class, log)
            for name, member in MEMBERS.items()
        }
        eval_paths = {
            name: _pred_parquet(
                member,
                "eval",
                eval_root,
                out_root,
                args.n_per_class,
                log,
                reuse_historical_eval=reuse_historical_eval,
            )
            for name, member in MEMBERS.items()
        }
        for tag, k, names in ENSEMBLES:
            _ensemble_report(
                tag,
                k,
                names,
                train_paths,
                eval_paths,
                args.dataset,
                train_root,
                eval_root,
                out_root,
                leaderboard,
                log,
            )

        if not args.skip_snapshot:
            _run_snapshot_repro(log)
        if not args.no_restart:
            _restart_loop(out_root, restart_datasets, not args.no_watchdog, log)
        log.write(f"[ensemble-queue] done {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
