#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""logs_chipgrid/<run>/ 의 모든 학습 결과를 markdown summary 로 출력.

사용:
    python _chipgrid_summary.py                # 전체 run 표 + 각 run 상세 → stdout
    python _chipgrid_summary.py > docs/chipgrid/RESULTS_AUTO.md
    python _chipgrid_summary.py --filter v3    # 이름에 'v3' 포함된 run 만

추출 항목 (각 run 마다):
- hparams (variant / n_per_class / obj_norm / target_id / chip_noise / seed / epochs)
- 데이터 분포: wafer class 별 train/val/test count, obj_id chip-object 분포
- BEST OVERALL: val/test acc/f1/P/R, best_epoch, total_epochs_run, early_stopped
- per-class (TEST + VAL): F1, FP, FN, support — weak class 만 강조
- BEST UPDATES SUMMARY: 매 best 갱신 epoch
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image


CHIPGRID_DIR = Path("D:/project/known-cnn/logs_chipgrid")
DATA_DIR = Path("D:/project/data/wm-811k/unknown")
OBJ_ID_DIR = Path("D:/project/data/wm-811k/obj_id_maps")
EXCLUDE_CLASSES = {"Normal", "classification", "classification_chips"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(CHIPGRID_DIR))
    p.add_argument("--filter", default=None, help="substring match on run dir name")
    p.add_argument("--no-data-dist", action="store_true",
                   help="skip wafer class data distribution section (faster)")
    p.add_argument("-o", "--output", default=None,
                   help="output file path (default: stdout). UTF-8 always.")
    return p.parse_args()


def collect_runs(root: Path, filt: str | None):
    runs = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if "_running" in d.name or "_ABORTED" in d.name or "_PAUSED" in d.name:
            continue
        # finished run dirs end with <test_f1>_<val_f1>
        if not re.search(r"_\d\.\d{2}_\d\.\d{2}$", d.name):
            continue
        if filt and filt.lower() not in d.name.lower():
            continue
        if (d / "best_history.txt").exists() and (d / "hparams.json").exists():
            runs.append(d)
    return runs


def parse_best_history(path: Path) -> dict:
    """Extract BEST OVERALL + per-class + updates summary from best_history.txt."""
    text = path.read_text(encoding="utf-8")
    out = {"raw": text}
    # BEST OVERALL line
    m = re.search(r"epoch (\d+)\s*\|\s*val F1 = (\d\.\d+)", text)
    if m:
        out["best_epoch"] = int(m.group(1))
        out["best_val_f1"] = float(m.group(2))
    # TEST acc/f1
    m = re.search(r"TEST\s+acc=\s*([\d.]+)%\s+f1=\s*([\d.]+)%", text)
    if m:
        out["test_acc"] = float(m.group(1))
        out["test_f1"] = float(m.group(2))
    # VAL acc/f1
    m = re.search(r"VAL\s+acc=\s*([\d.]+)%\s+f1=\s*([\d.]+)%", text)
    if m:
        out["val_acc"] = float(m.group(1))
        out["val_f1"] = float(m.group(2))
    # per-class — TEST then VAL sections. extract weak classes (F1 < 1.0)
    out["test_per_class"] = parse_per_class_section(text, "TEST")
    out["val_per_class"] = parse_per_class_section(text, "VAL")
    # BEST UPDATES SUMMARY rows
    out["best_updates"] = parse_best_updates(text)
    return out


def parse_per_class_section(text: str, split: str) -> list[dict]:
    """Find FINAL BEST per-class block for given split."""
    # try 2 possible headers (chipgrid script):
    #   "[1] FINAL BEST per-class (TEST)"  — chipgrid
    #   "[1b] FINAL BEST per-class (VAL)"  — chipgrid
    #   "[TEST per-class]"                 — compound/wafer
    patterns = [
        rf"FINAL BEST per-class \({split}\)\s*\n=+\n(.*?)(?:\n\n|\n\[)",
        rf"\[{split} per-class\]\s*\n(.*?)(?:\n\n|\n=)",
    ]
    block = None
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL)
        if m:
            block = m.group(1)
            break
    if not block:
        return []
    rows = []
    for line in block.split("\n"):
        line = line.strip()
        if not line or line.startswith("-") or line.startswith("Class") or line.startswith("=") \
                or line.startswith("macro") or line.startswith("weighted") or line.startswith("overall"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        # could be: class F1 P R Sup  OR  class F1 P R FP FN Sup
        # last is always Sup. detect: 5 tokens after class = chipgrid (no FP/FN), 6 = compound
        if len(parts) >= 7:  # compound style (class, F1, P, R, FP, FN, Sup)
            try:
                cls = parts[0]
                f1 = float(parts[1]); p = float(parts[2]); r = float(parts[3])
                fp = int(parts[4]); fn = int(parts[5]); sup = int(parts[6])
                rows.append({"class": cls, "f1": f1, "p": p, "r": r, "fp": fp, "fn": fn, "sup": sup})
            except (ValueError, IndexError):
                continue
        elif len(parts) >= 5:  # chipgrid style (class, F1, P, R, Sup)  — FP/FN derived
            try:
                cls = parts[0]
                f1 = float(parts[1]); p = float(parts[2]); r = float(parts[3])
                sup = int(parts[4])
                # derive FP/FN from P, R, sup:  TP = R * sup; FN = sup - TP; FP = TP/P - TP (if P>0)
                tp = r * sup
                fn_calc = round(sup - tp)
                fp_calc = round(tp / p - tp) if p > 0 else 0
                rows.append({"class": cls, "f1": f1, "p": p, "r": r,
                             "fp": int(fp_calc), "fn": int(fn_calc), "sup": sup})
            except (ValueError, IndexError):
                continue
    return rows


def parse_best_updates(text: str) -> list[dict]:
    """Extract 'BEST UPDATES SUMMARY' rows."""
    m = re.search(r"BEST UPDATES SUMMARY.*?\n(?:.*?\n)*?\s+ep\s+.*?\n(.*?)(?:\n\n|\n=|$)",
                  text, re.DOTALL)
    if not m:
        return []
    rows = []
    for line in m.group(1).split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            ep = int(parts[0])
            row = {"ep": ep}
            # remaining percentages
            row["val_f1"] = parts[1]
            row["val_acc"] = parts[2] if len(parts) > 2 else None
            if len(parts) >= 5:
                row["test_f1"] = parts[3]
                row["test_acc"] = parts[4]
            rows.append(row)
        except (ValueError, IndexError):
            continue
    return rows


def parse_history(path: Path) -> dict:
    """Read history.json: total epochs run, last epoch metrics."""
    if not path.exists():
        return {}
    h = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(h, list) or len(h) == 0:
        return {}
    return {
        "total_epochs": len(h),
        "last_val_f1": h[-1].get("val_f1"),
        "last_tr_loss": h[-1].get("tr_loss"),
    }


def build_data_composition(n_per_class: int) -> dict:
    """For each wafer class: how many samples are loaded (capped at n_per_class).

    Also computes obj_id chip-object distribution across the loaded samples (approximate —
    counts non-zero pixels per class id in the obj_id npys).
    """
    classes = sorted([d.name for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name not in EXCLUDE_CLASSES])
    per_class_counts = {}
    for c in classes:
        files = sorted([p for p in (DATA_DIR / c).glob("*.png")])
        per_class_counts[c] = min(len(files), n_per_class)
    total = sum(per_class_counts.values())
    # rough obj_id distribution: pick first sample of each class, count chip-id histogram
    # (이 정도면 충분 — full sample 합산은 시간 비싸다)
    obj_hist_total = Counter()
    npy_lookup = {}
    for p in OBJ_ID_DIR.rglob("*.npy"):
        if not p.name.startswith("_"):
            npy_lookup[p.stem.upper()] = p
    sampled = 0
    for c in classes[:5]:  # sample first 5 wafer classes
        files = sorted([p for p in (DATA_DIR / c).glob("*.png")])[:5]  # 5 wafer per class
        for png in files:
            npy_path = npy_lookup.get(png.stem.upper())
            if npy_path is None:
                continue
            arr = np.load(npy_path)
            for v in np.unique(arr):
                obj_hist_total[int(v)] += int((arr == v).sum())
            sampled += 1
    return {"per_class_counts": per_class_counts, "total": total,
            "obj_hist_sampled": dict(obj_hist_total), "obj_hist_n_wafers": sampled}


def fmt_data_dist(comp: dict) -> str:
    lines = [f"- 총 sample (capped): **{comp['total']}**, 80/10/10 split"]
    # show only classes where count differs from default
    counts = comp["per_class_counts"]
    if counts:
        max_c = max(counts.values())
        outliers = {k: v for k, v in counts.items() if v != max_c}
        if outliers:
            lines.append(f"- per-class count: 대부분 {max_c}, 예외: {outliers}")
        else:
            lines.append(f"- per-class count: 모두 {max_c} (균일)")
    if comp.get("obj_hist_sampled"):
        labels = {0: "none", 1: "bank_boundary", 2: "invalid_main",
                  3: "particle_blast", 4: "scratch", 5: "scratch_21deg"}
        h = comp["obj_hist_sampled"]
        total_pix = sum(h.values()) or 1
        nz = sum(v for k, v in h.items() if k > 0) or 1
        chip_pcts = []
        for k in sorted(h):
            pct = 100.0 * h[k] / nz if k > 0 else 0
            label = labels.get(k, f"id{k}")
            if k > 0:
                chip_pcts.append(f"{label} {pct:.1f}%")
        lines.append(f"- obj_id 분포 (chip 단위, n_wafers={comp['obj_hist_n_wafers']} 샘플): "
                     + " / ".join(chip_pcts))
    return "\n".join(lines)


def fmt_per_class_weak(rows: list[dict], threshold: float = 0.95) -> str:
    weak = [r for r in rows if r["f1"] < threshold]
    if not weak:
        return f"  - **모두 F1 ≥ {threshold:.2f}** (perfect)"
    lines = [f"  - weak class ({len(weak)}/{len(rows)}, F1 < {threshold:.2f}):"]
    for r in sorted(weak, key=lambda x: x["f1"]):
        lines.append(f"    - `{r['class']}`: F1={r['f1']:.3f}  FP={r['fp']}  FN={r['fn']}  Sup={r['sup']}")
    return "\n".join(lines)


def render_run(run_dir: Path, comp_cache: dict, no_data_dist: bool) -> str:
    hp = json.loads((run_dir / "hparams.json").read_text(encoding="utf-8"))
    bh = parse_best_history(run_dir / "best_history.txt")
    hist = parse_history(run_dir / "history.json")
    out = []
    out.append(f"\n### {run_dir.name}\n")
    out.append("**hparams**:")
    keys = ["variant", "n_per_class", "obj_norm", "target_id", "chip_noise",
            "chip_noise_eval", "seed", "epochs", "batch", "ema_decay"]
    hp_str = " | ".join(f"`{k}={hp.get(k)}`" for k in keys if hp.get(k) is not None)
    out.append(f"- {hp_str}")
    out.append(f"- in_ch=`{hp.get('in_ch')}`, n_classes=`{hp.get('n_classes')}`, "
               f"params=`{hp.get('n_params'):,}`")
    if not no_data_dist:
        n_per = hp.get("n_per_class", 30)
        if n_per not in comp_cache:
            comp_cache[n_per] = build_data_composition(n_per)
        out.append("\n**데이터**:")
        out.append(fmt_data_dist(comp_cache[n_per]))
    # BEST
    out.append("\n**BEST OVERALL**:")
    if bh.get("test_f1") is not None:
        out.append(f"- TEST  acc={bh.get('test_acc',0):.2f}%  f1=**{bh.get('test_f1',0):.2f}%**")
    if bh.get("val_f1") is not None:
        out.append(f"- VAL   acc={bh.get('val_acc',0):.2f}%  f1=**{bh.get('val_f1',0):.2f}%**")
    out.append(f"- best epoch = {bh.get('best_epoch')}, total epochs run = {hist.get('total_epochs', '?')}")
    if hist.get('total_epochs') and hp.get('epochs') and hist['total_epochs'] < hp['epochs']:
        out.append(f"  - **early stopped** (patience={hp.get('patience', 7)})")
    # weak class breakdown
    if bh.get("test_per_class"):
        out.append("\n**TEST per-class (weak)**:")
        out.append(fmt_per_class_weak(bh["test_per_class"]))
    if bh.get("val_per_class"):
        out.append("\n**VAL per-class (weak)**:")
        out.append(fmt_per_class_weak(bh["val_per_class"]))
    # epoch progression
    if bh.get("best_updates"):
        out.append("\n**BEST UPDATES** (매 best 갱신):")
        out.append("```")
        out.append(f"{'ep':>4}  {'val_f1':>8}  {'test_f1':>8}")
        for u in bh["best_updates"]:
            out.append(f"{u['ep']:>4}  {u.get('val_f1','?'):>8}  {u.get('test_f1','?'):>8}")
        out.append("```")
    return "\n".join(out)


def main():
    args = parse_args()
    runs = collect_runs(Path(args.root), args.filter)
    lines = []
    lines.append(f"# chipgrid 학습 결과 상세 (자동 생성)\n")
    lines.append(f"_생성 시각: `python _chipgrid_summary.py`{'(filtered: '+args.filter+')' if args.filter else ''}_")
    lines.append(f"\n총 {len(runs)} 개 run 분석\n")
    comp_cache = {}
    for run_dir in runs:
        try:
            lines.append(render_run(run_dir, comp_cache, args.no_data_dist))
        except Exception as e:
            lines.append(f"\n### {run_dir.name}")
            lines.append(f"_parse error: {e}_")
    text = "\n".join(lines)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"wrote {args.output} ({len(text)} chars)")
    else:
        # stdout: try utf-8 reconfigure (Python 3.7+), fallback to default
        try:
            import sys
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
        print(text)


if __name__ == "__main__":
    main()
