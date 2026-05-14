"""Aggregate all eval_n2000_pred/ results into a single CSV + ranking table.

Scans: outputs/<cell>/<inner_run>/eval_n2000_pred/stage1_*/preds_chip.parquet
Computes per (cell, inference variant):
  bit_F1     = positive (single+combo) macro-F1
  total_far  = (Normal_fp + Invalid_fp + OOD_fp) / N_total_negative
  ni_far     = (Normal_fp + Invalid_fp) / (N_Normal + N_Invalid)
  ood_far    = OOD_fp / N_OOD
  n_pos, n_ni, n_ood

Absolute rule 260512.
"""
from __future__ import annotations
import ast
import glob
import os
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score


SINGLES = {"bank_boundary", "fork", "scratch", "scratch_rot"}
COMBOS_2 = {
    "bank_boundary+fork", "bank_boundary+scratch", "bank_boundary+scratch_rot",
    "fork+scratch", "fork+scratch_rot", "scratch+scratch_rot",
}
POSITIVE = SINGLES | COMBOS_2
NEG_NI = {"Normal", "Invalid"}
NEG_OOD = {"CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst"}


def evaluate(df_var: pd.DataFrame) -> dict:
    pos = df_var[df_var["class_key"].isin(POSITIVE)].copy()
    if len(pos):
        f1 = f1_score(pos["true_labels"].tolist(), pos["pred_labels"].tolist(),
                      average="macro", zero_division=0)
    else:
        f1 = 0.0
    neg = df_var[df_var["class_key"].isin(NEG_NI | NEG_OOD)].copy()
    ni = neg[neg["class_key"].isin(NEG_NI)]
    ood = neg[neg["class_key"].isin(NEG_OOD)]
    fp_decision = ["single", "combo", "combo_collapsed"]
    if len(neg):
        fp = neg["decision_type"].isin(fp_decision).sum()
        ni_fp = ni["decision_type"].isin(fp_decision).sum() if len(ni) else 0
        ood_fp = ood["decision_type"].isin(fp_decision).sum() if len(ood) else 0
        total_far = 100.0 * fp / len(neg)
        ni_far = 100.0 * ni_fp / max(1, len(ni))
        ood_far = 100.0 * ood_fp / max(1, len(ood))
    else:
        total_far = ni_far = ood_far = 0.0
    return {
        "bit_f1": f1,
        "total_far": total_far,
        "ni_far": ni_far,
        "ood_far": ood_far,
        "n_pos": len(pos),
        "n_ni": len(ni),
        "n_ood": len(ood),
    }


def main():
    rows = []
    pq_files = sorted(glob.glob("outputs/*/*/eval_n2000_pred/stage1_*/preds_chip.parquet"))
    print(f"Found {len(pq_files)} eval_n2000_pred parquets")

    for pq in pq_files:
        parts = pq.split(os.sep)
        # outputs/<cell>/<inner>/eval_n2000_pred/stage1_<TS>/preds_chip.parquet
        cell = parts[1]
        df = pd.read_parquet(pq)
        for var in sorted(df["cell_id"].unique()):
            sub = df[df["cell_id"] == var]
            m = evaluate(sub)
            rows.append({"cell": cell, "variant": var, **m})

    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("No data, exiting")
        return

    # Sort by bit_F1 desc, then FAR asc — main ranking
    out["combined"] = out["bit_f1"] - out["total_far"] / 100.0
    out["dual_gate"] = (out["bit_f1"] >= 0.99) & (out["total_far"] <= 0.5)
    out = out.sort_values(["dual_gate", "combined"], ascending=[False, False])

    print(f"\nCells evaluated: {out['cell'].nunique()}")
    print(f"Dual-gate pass (bit_F1 ≥ 0.99 ∧ FAR ≤ 0.5%): {out['dual_gate'].sum()}")
    print()

    cols = ["cell", "variant", "bit_f1", "total_far", "ni_far", "ood_far", "n_pos", "dual_gate"]
    print("TOP 20:")
    print(out[cols].head(20).to_string(index=False, formatters={
        "bit_f1": "{:.4f}".format,
        "total_far": "{:.2f}".format,
        "ni_far": "{:.2f}".format,
        "ood_far": "{:.2f}".format,
    }))

    csv = "outputs/_pred_n2000_aggregate.csv"
    out.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
