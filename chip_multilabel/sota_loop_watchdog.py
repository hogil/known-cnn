# -*- coding: utf-8 -*-
"""Keep the frozen-original chip multilabel SOTA sweep alive.

This watcher is intentionally conservative:
- it starts `recipe_sweep --forever` only when no sweep parent is alive;
- it avoids starting a duplicate while a train/eval/diag child is still alive;
- it writes a small heartbeat log under outputs/frozen_original.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
SWEEP_DATASETS = os.environ.get("CHIP_SOTA_DATASETS", "frozen_iter116J_orig814_old_eval")
ACTIVE_DATASET = SWEEP_DATASETS.split(",")[0]
OUT_DIR = ROOT / "outputs" / ACTIVE_DATASET
SWEEP_MARKERS = ["chip_multilabel.recipe_sweep", "--datasets", SWEEP_DATASETS]
LEGACY_SWEEP_MARKERS = ["chip_multilabel.recipe_sweep", "--datasets", "frozen_original"]
WORK_MARKERS = [
    "chip_multilabel._train_chip_variant",
    "chip_multilabel.run_stage1",
    "chip_multilabel._posneg_prob_diag",
]


def _cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline())
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        return ""


def _matching(markers: list[str]) -> list[psutil.Process]:
    out: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name"]):
        name = (proc.info.get("name") or "").lower()
        if "python" not in name:
            continue
        cmd = _cmdline(proc)
        if all(marker in cmd for marker in markers):
            out.append(proc)
    return out


def _workers() -> list[psutil.Process]:
    out: list[psutil.Process] = []
    for proc in psutil.process_iter(["pid", "name"]):
        name = (proc.info.get("name") or "").lower()
        if "python" not in name:
            continue
        cmd = _cmdline(proc)
        if ACTIVE_DATASET in cmd and any(marker in cmd for marker in WORK_MARKERS):
            out.append(proc)
    return out


def _launch() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
            "TORCH_NUM_THREADS": "2",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    stdout = OUT_DIR / "_sota_loop_stdout.log"
    stderr = OUT_DIR / "_sota_loop_stderr.log"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "chip_multilabel.recipe_sweep",
        "--datasets",
        SWEEP_DATASETS,
        "--forever",
        "--diag-device",
        "cuda",
        "--eval-n-per-class",
        "2000",
        "--train-eval-n-per-class",
        "2000",
        "--diag-cap",
        "2000",
    ]
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


def _terminate(procs: list[psutil.Process], log) -> None:
    current = os.getpid()
    targets = [p for p in procs if p.pid != current]
    for proc in targets:
        try:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} terminate_legacy parent={proc.pid}\n")
            log.flush()
            proc.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    gone, alive = psutil.wait_procs(targets, timeout=10)
    for proc in alive:
        try:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} kill_legacy parent={proc.pid}\n")
            log.flush()
            proc.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass


def _refresh_status() -> None:
    _normalize_performance_reports()
    latest = OUT_DIR / "_latest_status.md"
    cmd = [
        sys.executable,
        "-m",
        "chip_multilabel.report_sota_status",
        "--dataset",
        ACTIVE_DATASET,
        "--top",
        "20",
        "--reports",
        "3",
    ]
    header = f"# {ACTIVE_DATASET} latest status ({time.strftime('%Y-%m-%d %H:%M:%S')})\n"
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    text = header + proc.stdout
    if proc.returncode != 0:
        text += "\nSTATUS_REFRESH_ERROR\n" + proc.stderr
    latest.write_text(text, encoding="utf-8")


def _normalize_performance_reports() -> None:
    lead = OUT_DIR / "_leaderboard.csv"
    if not lead.exists():
        return
    from chip_multilabel.recipe_sweep import write_performance_report

    with lead.open("r", encoding="utf-8", errors="replace", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row.get("status") != "done":
            continue
        perf = row.get("performance_report", "")
        train = row.get("train_pcls_report", "")
        eval_ = row.get("eval_pcls_report", "")
        if not perf or not train or not eval_:
            continue
        train_path = Path(train)
        eval_path = Path(eval_)
        if not train_path.exists() or not eval_path.exists():
            continue
        write_performance_report(Path(perf), row, train_path, eval_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=300)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "_sota_loop_watchdog.log"
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} watchdog_start interval={args.interval}\n")
        f.flush()
        while True:
            _refresh_status()
            parents = _matching(SWEEP_MARKERS)
            workers = _workers()
            if parents:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} alive "
                    f"parents={[p.pid for p in parents]} workers={[p.pid for p in workers]}\n"
                )
            elif workers:
                f.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} wait_parent_missing_worker_alive "
                    f"workers={[p.pid for p in workers]}\n"
                )
            else:
                legacy = [p for p in _matching(LEGACY_SWEEP_MARKERS) if p not in parents]
                if legacy:
                    _terminate(legacy, f)
                pid = _launch()
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} restarted parent={pid}\n")
            f.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
