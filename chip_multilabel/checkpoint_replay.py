# -*- coding: utf-8 -*-
"""Replay historical chip-multilabel checkpoints on the fixed frozen protocol.

This does not train.  It evaluates existing checkpoints on:

- train: E:/data/images/classification_chips
- eval : E:/data/images/chip_multilabel_v15direct_n2000

and writes the same mandatory train/eval/per-class probability reports as the
recipe sweep.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from pathlib import Path

from chip_multilabel.recipe_sweep import (
    DATASETS,
    REPO,
    append_leaderboard,
    diag_cmd,
    eval_cmd,
    parse_diag,
    parse_eval,
    parse_pcls_rows,
    pcls_gap_fields,
    read_leaderboard,
    run_logged,
    write_pcls_csv,
    write_pcls_report,
    write_performance_report,
)


PRIORITY_MODELS = [
    ("iter116J_frozen_best", "models/iter116J_frozen/best_model.pth"),
    (
        "fcm_margin_g3_cls07_ls30_nopair_best",
        "outputs/fcm_margin_g3_cls07_ls30_nopair/20260521_214303_T7_T7_margin_g3_cls07_ls30_nopair/best_model.pth",
    ),
    (
        "fcm_margin_g3_cls07_ls30_nopair_final",
        "outputs/fcm_margin_g3_cls07_ls30_nopair/20260521_214303_T7_T7_margin_g3_cls07_ls30_nopair/final_epoch_model.pth",
    ),
    (
        "fcm_margin_g3_cls07_ls30_pair_final",
        "outputs/fcm_margin_g3_cls07_ls30_pair/20260521_210846_T7_T7_margin_g3_cls07_ls30_pair/final_epoch_model.pth",
    ),
    (
        "fcm_margin_g4_cls05_ls30_nopair_best",
        "outputs/fcm_margin_g4_cls05_ls30_nopair/20260521_225230_T7_T7_margin_g4_cls05_ls30_nopair/best_model.pth",
    ),
    (
        "iter116J_exact_repro_20260524_best",
        "outputs/iter116J_exact_repro/20260524_123834_T7_iter116J_exact/best_model.pth",
    ),
    (
        "iter116J_exact_repro_20260522_best",
        "outputs/iter116J_exact_repro/20260522_142643_T7_iter116J_exact/best_model.pth",
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


def _newest_stage1_preds(out_root: Path) -> Path | None:
    preds = sorted(
        out_root.glob("**/preds_chip.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return preds[0] if preds else None


def _eval_diag_cmd_from_existing_preds(
    model: Path,
    root: str,
    tag: str,
    cap: int,
    device: str,
    out_dir: Path,
    cell_id: str,
) -> list[str]:
    preds = _newest_stage1_preds(out_dir / "eval_external")
    if not preds:
        return diag_cmd(model, root, tag, cap, device)
    batch_size = os.environ.get("CHIP_PCLS_FAST_BATCH_SIZE", os.environ.get("CHIP_PCLS_BATCH_SIZE", "96"))
    return [
        sys.executable,
        "-u",
        "-m",
        "chip_multilabel.pcls_from_stage1_split",
        "--model",
        str(model),
        "--root",
        root,
        "--stage1-preds",
        str(preds),
        "--cell-id",
        cell_id or "T0__I10",
        "--cap-per-class",
        str(cap),
        "--device",
        device,
        "--batch-size",
        batch_size,
        "--sample-seed",
        "42",
        "--split-seed",
        "42",
        "--tag",
        tag,
    ]


def _eval_cmd_replay(model: Path, root: str, out_root: Path, n_per_class: int, args: argparse.Namespace) -> list[str]:
    cmd = eval_cmd(model, root, out_root, n_per_class)
    if args.skip_invalid_heuristic:
        cmd.append("--skip-invalid-heuristic")
    return cmd


def _nb_reject_sweep_cmd(calib_preds: Path, eval_preds: Path, out_dir: Path, cell_id: str) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "chip_multilabel.nb_reject_sweep_report",
        "--calib-preds",
        str(calib_preds),
        "--eval-preds",
        str(eval_preds),
        "--out-dir",
        str(out_dir),
        "--cell",
        cell_id or "T0__I10",
    ]


def _clean_tag(name: str, path: Path) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.+-]+", "_", name).strip("_")
    if not clean:
        clean = re.sub(r"[^A-Za-z0-9_.+-]+", "_", path.parent.name).strip("_")
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"replay_{clean}_{digest}"


def _candidate_csv(path: Path) -> list[tuple[str, Path]]:
    if not path.exists():
        return []
    out: list[tuple[str, Path]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            p = Path(row.get("path", ""))
            if p.exists():
                out.append((p.parent.name + "_" + p.name.replace(".pth", ""), p))
    return out


def _models(args: argparse.Namespace) -> list[tuple[str, Path]]:
    seen: set[str] = set()
    out: list[tuple[str, Path]] = []

    def add(name: str, raw: str | Path) -> None:
        p = Path(raw)
        if not p.is_absolute():
            p = REPO / p
        if not p.exists():
            return
        key = str(p.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        out.append((name, p))

    for item in args.model:
        if "=" in item:
            name, raw = item.split("=", 1)
            p = Path(raw)
            add(name.strip() or p.parent.name + "_" + p.name.replace(".pth", ""), p)
        else:
            p = Path(item)
            add(p.parent.name + "_" + p.name.replace(".pth", ""), p)
    for name, raw in PRIORITY_MODELS:
        add(name, raw)
    if args.candidate_csv:
        for name, path in _candidate_csv(Path(args.candidate_csv)):
            add(name, path)
    if args.max_models > 0:
        out = out[: args.max_models]
    return out


def replay_one(ds_name: str, name: str, model: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    ds = DATASETS[ds_name]
    tag = _clean_tag(name, model)
    out_dir = REPO / "outputs" / ds.name / tag
    lead = REPO / "outputs" / ds.name / "_leaderboard.csv"
    existing_done = {
        r.get("tag", "")
        for r in read_leaderboard(lead)
        if r.get("status") == "done"
    }
    if tag in existing_done and not args.force:
        print(f"[replay] skip done tag={tag}", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ds.name,
        "train_root": ds.train,
        "eval_root": ds.eval,
        "tag": tag,
        "variant": "checkpoint_replay",
        "LS": "",
        "n_groups": "",
        "cmp_ls": "",
        "cutmix_p": "",
        "seed": "",
        "extra": name,
        "eval_n_per_class": str(args.eval_n_per_class),
        "train_eval_n_per_class": str(args.train_eval_n_per_class),
        "train_diag_cap": str(args.train_diag_cap),
        "eval_diag_cap": str(args.eval_diag_cap),
        "epochs": "",
        "batch": "",
        "accum": "",
        "status": "start",
        "ckpt": "external",
        "model": str(model),
        "out_dir": str(out_dir),
    }
    print(f"[replay] START tag={tag}", flush=True)
    print(f"[replay] model={model}", flush=True)
    print(f"[replay] train={ds.train}", flush=True)
    print(f"[replay] eval ={ds.eval}", flush=True)

    eval_log = out_dir / "eval_external.log"
    if not eval_log.exists() or not parse_eval(eval_log):
        rc = run_logged(_eval_cmd_replay(model, ds.eval, out_dir / "eval_external", args.eval_n_per_class, args), eval_log, env)
        if rc != 0:
            row["status"] = f"eval_fail_{rc}"
            append_leaderboard(lead, row)
            print(f"[replay] EVAL FAIL rc={rc} tag={tag}", flush=True)
            return
    em = parse_eval(eval_log)
    if not em:
        row["status"] = "eval_no_metrics"
        append_leaderboard(lead, row)
        return
    row.update(
        {
            "eval_cell": em.get("cell", ""),
            "eval_bit_F1": em.get("bit_F1", ""),
            "eval_NI_FAR": em.get("NI_FAR", ""),
            "eval_OOD_FAR": em.get("OOD_FAR", ""),
            "eval_Total_FAR": em.get("Total_FAR", ""),
            "eval_bb_F1": em.get("bb_F1", ""),
            "eval_fk_F1": em.get("fk_F1", ""),
            "eval_sc_F1": em.get("sc_F1", ""),
            "eval_sr_F1": em.get("sr_F1", ""),
        }
    )

    train_log = out_dir / "train_eval_external.log"
    if not train_log.exists() or not parse_eval(train_log):
        rc = run_logged(_eval_cmd_replay(model, ds.train, out_dir / "train_eval_external", args.train_eval_n_per_class, args), train_log, env)
        if rc != 0:
            row["status"] = f"train_eval_fail_{rc}"
            append_leaderboard(lead, row)
            return
    tm = parse_eval(train_log)
    row.update(
        {
            "train_cell": tm.get("cell", ""),
            "train_bit_F1": tm.get("bit_F1", ""),
            "train_NI_FAR": tm.get("NI_FAR", ""),
            "train_OOD_FAR": tm.get("OOD_FAR", ""),
            "train_Total_FAR": tm.get("Total_FAR", ""),
        }
    )

    train_diag_log = out_dir / "train_posneg_pcls_external.log"
    if not train_diag_log.exists() or not parse_diag(train_diag_log):
        rc = run_logged(diag_cmd(model, ds.train, f"{ds.name}:{tag}:train", args.train_diag_cap, args.diag_device), train_diag_log, env)
        if rc != 0:
            print(f"[replay] WARN train diag failed rc={rc} tag={tag}", flush=True)
    td = parse_diag(train_diag_log)
    train_rows = parse_pcls_rows(train_diag_log)
    if not train_rows:
        row["status"] = "train_diag_no_pcls"
        append_leaderboard(lead, row)
        return
    row["train_pos_prob"] = td.get("pos_prob", "")
    row["train_neg_prob"] = td.get("neg_prob", "")
    write_pcls_csv(train_rows, out_dir / "train_pcls.csv")
    train_report = out_dir / "train_pcls_report.md"
    write_pcls_report(train_rows, train_report, f"{tag} - TRAIN per-class 4-bit prob", td, "train", ds.train, tm)
    row["train_pcls_report"] = str(train_report)

    eval_diag_log = out_dir / "eval_posneg_pcls_external.log"
    if not eval_diag_log.exists() or not parse_diag(eval_diag_log):
        rc = run_logged(
            _eval_diag_cmd_from_existing_preds(
                model,
                ds.eval,
                f"{ds.name}:{tag}:eval",
                args.eval_diag_cap,
                args.diag_device,
                out_dir,
                em.get("cell", "T0__I10"),
            ),
            eval_diag_log,
            env,
        )
        if rc != 0:
            print(f"[replay] WARN eval diag failed rc={rc} tag={tag}", flush=True)
    ed = parse_diag(eval_diag_log)
    eval_rows = parse_pcls_rows(eval_diag_log)
    if not eval_rows:
        row["status"] = "eval_diag_no_pcls"
        append_leaderboard(lead, row)
        return
    row["eval_pos_prob"] = ed.get("pos_prob", "")
    row["eval_neg_prob"] = ed.get("neg_prob", "")
    row.update(pcls_gap_fields(eval_rows))
    write_pcls_csv(eval_rows, out_dir / "eval_pcls.csv")
    eval_report = out_dir / "eval_pcls_report.md"
    write_pcls_report(eval_rows, eval_report, f"{tag} - EVAL per-class 4-bit prob", ed, "eval", ds.eval, em)
    row["eval_pcls_report"] = str(eval_report)

    perf_report = out_dir / "performance_report.md"
    write_performance_report(perf_report, row, train_report, eval_report)
    row["performance_report"] = str(perf_report)
    if args.nb_calib_preds:
        calib_preds = Path(args.nb_calib_preds)
        eval_preds = _newest_stage1_preds(out_dir / "eval_external")
        if calib_preds.exists() and eval_preds and eval_preds.exists():
            nb_dir = out_dir / args.nb_reject_out_name
            nb_log = out_dir / "nb_reject_sweep_external.log"
            rc = run_logged(_nb_reject_sweep_cmd(calib_preds, eval_preds, nb_dir, em.get("cell", "T0__I10")), nb_log, env)
            if rc != 0:
                print(f"[replay] WARN nb reject sweep failed rc={rc} tag={tag}", flush=True)
    row["status"] = "done"
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    append_leaderboard(lead, row)
    print(
        f"[replay] DONE tag={tag} eval_bit_F1={row['eval_bit_F1']} "
        f"eval_Total_FAR={row['eval_Total_FAR']}%",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="frozen_original")
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--candidate-csv", default="")
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument("--eval-n-per-class", type=int, default=2000)
    ap.add_argument("--train-eval-n-per-class", type=int, default=2000)
    ap.add_argument("--diag-cap", type=int, default=2000)
    ap.add_argument("--train-diag-cap", type=int, default=0)
    ap.add_argument("--eval-diag-cap", type=int, default=0)
    ap.add_argument("--diag-device", default="cuda")
    ap.add_argument("--nb-calib-preds", default="")
    ap.add_argument("--nb-reject-out-name", default="nb_reject_sweep_calib_v15_n2000")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-invalid-heuristic", action="store_true")
    args = ap.parse_args()
    if args.train_diag_cap <= 0:
        args.train_diag_cap = args.diag_cap
    if args.eval_diag_cap <= 0:
        args.eval_diag_cap = args.diag_cap

    env = _env()
    models = _models(args)
    if not models:
        raise SystemExit("no existing checkpoints selected")
    for name, model in models:
        replay_one(args.dataset, name, model, args, env)


if __name__ == "__main__":
    main()
