# -*- coding: utf-8 -*-
"""Build the 3-component ablation (FCM-PM + val-margin + nb-reject) for the
frozen_original champion, WITHOUT retraining or GPU re-evaluation.

The champion per-epoch/best checkpoints have been pruned (resource-manager keeps
logs+reports, deletes .pth), so the planned "re-evaluate saved checkpoints with
different (selection, acceptor)" cannot be executed for the champion.  What IS
recoverable from the logged artifacts of the champion run:

  cell (c) full method  = best_model.pth (val-margin, epoch 4) + I10 nb-reject
                          -> read directly from eval_best/.../bit_far_metrics.json
  cell (b) +val-margin  = the SAME ep4 checkpoint at raw sigmoid threshold 0.5
                          (no acceptor) -> reconstructed from the ep4 raw-threshold
                          pcls diagnostic (eval_pcls.csv), aggregated with the
                          tool's own micro chip-FAR formula: FAR = sum(fp)/sum(n).
  cell (a) FCM-PM only  = val-F1 selection (epoch 2) + raw threshold -> NOT
                          obtainable: the ep2 / best_f1_model.pth checkpoint was
                          pruned AND was never evaluated on the eval pool, so no
                          logged numbers exist.  Emitted as NA.

The (b)->(c) delta isolates the nb-reject/acceptor component on identical weights.
The (a)->(b) val-margin-selection component is NOT cleanly isolable from the
existing artifacts (documented, not fabricated).

Usage:
  python -m chip_multilabel._ablation_3component <champion_run_dir> [--out <csv>]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

NI_CLASSES = {"Normal", "Invalid"}


def _read_perf_report(run_dir: Path):
    """Parse train/eval bit_F1 + pos_prob/neg_prob from performance_report.md."""
    txt = (run_dir / "performance_report.md").read_text(encoding="utf-8", errors="ignore")
    out = {}
    for split in ("train", "eval"):
        m = re.search(
            rf"^{split} bit_F1=([0-9.]+).*?pos_prob=([0-9.]+)\s+neg_prob=([0-9.]+)",
            txt, re.MULTILINE)
        if m:
            out[split] = dict(bit_F1=float(m.group(1)),
                              pos=float(m.group(2)), neg=float(m.group(3)))
    return out


def _read_i10_metrics(run_dir: Path):
    """cell (c): read the logged I10 aggregate from eval_best bit_far_metrics.json."""
    cands = list((run_dir / "eval_best").rglob("bit_far_metrics.json"))
    if not cands:
        return None
    d = json.loads(cands[0].read_text(encoding="utf-8"))
    cell = d.get("T0__I10") or next(iter(d.values()))
    return dict(bit_F1=float(cell["bit_F1"]),
                ni_far=float(cell["NI_FAR"]),
                ood_far=float(cell["OOD_FAR"]),
                total_far=float(cell["Total_FAR"]))


def _read_raw_threshold_from_pcls(run_dir: Path):
    """cell (b): reconstruct raw-threshold-0.5 aggregate from eval_pcls.csv.

    POS rows carry metric==bit_F1 (raw-threshold per-class F1); NEG rows carry
    metric==FAR (raw-threshold per-class chip-FAR) + n.  Aggregate NEG with the
    tool's micro formula FAR = sum(FAR_c * n_c) / sum(n_c), split into NI vs OOD.
    """
    rows = list(csv.DictReader(open(run_dir / "eval_pcls.csv", encoding="utf-8")))
    pos_f1 = []
    ni_fp = ni_n = ood_fp = ood_n = 0.0
    for r in rows:
        if r["metric"] == "bit_F1":                       # POS class
            pos_f1.append(float(r["metric_value"]))
        elif r["metric"] == "FAR":                        # NEG class
            n = float(r["n"]); far = float(r["metric_value"]); fp = far * n
            if r["class"] in NI_CLASSES:
                ni_fp += fp; ni_n += n
            else:
                ood_fp += fp; ood_n += n
    return dict(
        bit_F1_macro_posclass=sum(pos_f1) / len(pos_f1),
        ni_far=ni_fp / ni_n if ni_n else float("nan"),
        ood_far=ood_fp / ood_n if ood_n else float("nan"),
        total_far=(ni_fp + ood_fp) / (ni_n + ood_n),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run_dir)
    tag = run.name

    perf = _read_perf_report(run)
    c = _read_i10_metrics(run)          # full method (val-margin ep4 + I10)
    b = _read_raw_threshold_from_pcls(run)  # val-margin ep4 + raw threshold

    trn = perf.get("train", {})
    evl = perf.get("eval", {})          # pos/neg identical for (b) and (c): same weights

    def pct(x):
        return "" if x != x else f"{100.0 * x:.2f}"

    header = ["config", "trn_bit_F1", "trn_pos", "trn_neg",
              "evl_bit_F1", "evl_pos", "evl_neg",
              "NI-FAR", "OOD-FAR", "Total-FAR", "delta-vs-prev", "note"]

    row_a = ["(a) FCM-PM only [val-F1 sel ep2 + raw-thresh]",
             "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA",
             "baseline missing",
             "ep2/best_f1_model.pth pruned AND never evaluated on eval pool -> not obtainable"]

    row_b = ["(b) FCM-PM + val-margin [val-margin sel ep4 + raw-thresh]",
             f"{trn.get('bit_F1', float('nan')):.4f}",
             f"{trn.get('pos', float('nan')):.4f}",
             f"{trn.get('neg', float('nan')):.4f}",
             f"{b['bit_F1_macro_posclass']:.4f}",
             f"{evl.get('pos', float('nan')):.4f}",
             f"{evl.get('neg', float('nan')):.4f}",
             pct(b["ni_far"]), pct(b["ood_far"]), pct(b["total_far"]),
             "(a)=NA",
             "raw sigmoid@0.5, no acceptor; reconstructed from ep4 pcls (bit_F1=POS-class macro)"]

    d_bit = c["bit_F1"] - b["bit_F1_macro_posclass"]
    d_ni = 100 * (c["ni_far"] - b["ni_far"])
    d_ood = 100 * (c["ood_far"] - b["ood_far"])
    d_tot = 100 * (c["total_far"] - b["total_far"])
    row_c = ["(c) FCM-PM + val-margin + nb-reject [val-margin sel ep4 + I10]",
             f"{trn.get('bit_F1', float('nan')):.4f}",
             f"{trn.get('pos', float('nan')):.4f}",
             f"{trn.get('neg', float('nan')):.4f}",
             f"{c['bit_F1']:.4f}",
             f"{evl.get('pos', float('nan')):.4f}",
             f"{evl.get('neg', float('nan')):.4f}",
             pct(c["ni_far"]), pct(c["ood_far"]), pct(c["total_far"]),
             f"bitF1 {d_bit:+.4f}; NI {d_ni:+.2f}pp; OOD {d_ood:+.2f}pp; Total {d_tot:+.2f}pp",
             "I10 = val-fitted joint-macro-F1 thresh + entropy-Normal reject gate (== champion)"]

    out = Path(a.out) if a.out else run.parent / f"_ablation_3component_{tag}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["# 3-component ablation for", tag])
        w.writerow(["# eval_pool", "chip_multilabel_v15direct_n2000"])
        w.writerow(["# NOTE",
                    "champion checkpoints pruned; (c) read from bit_far_metrics.json (I10); "
                    "(b) reconstructed from ep4 raw-threshold pcls diagnostic via micro chip-FAR; "
                    "(a) not obtainable (ep2/val-F1 ckpt pruned + never eval'd)"])
        w.writerow(header)
        w.writerow(row_a)
        w.writerow(row_b)
        w.writerow(row_c)
    print(f"[ablation] wrote {out}")
    # echo table to stdout
    for r in (header, row_a, row_b, row_c):
        print(" | ".join(str(x) for x in r[:11]))


if __name__ == "__main__":
    main()
