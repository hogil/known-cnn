"""W2 sweep aggregator → bit_F1 + Total_FAR per (cell, variant).

bit_F1 = macro-F1 over POSITIVE class_keys (single defect + 2-combo, scratch+scratch_rot OK).
Total_FAR = FP-rate over {Normal, Invalid, OOD wafer-patterns}.

Absolute rule 260512:
  - positive = single (bb/fork/scratch/scratch_rot) + 2-combo (bb+fork, bb+scratch,
    bb+scratch_rot, fork+scratch, fork+scratch_rot, scratch+scratch_rot)
  - Normal+Invalid = NI group → FP if pred != Normal
  - OOD wafer-patterns (CenterDonut/CrossScratch/DiagonalSmear/Starburst) → FP if pred != Normal
"""
from __future__ import annotations
import glob, os, json, sys
import pandas as pd
from sklearn.metrics import f1_score

SINGLES = {"bank_boundary", "fork", "scratch", "scratch_rot"}
COMBOS_2 = {
    "bank_boundary+fork", "bank_boundary+scratch", "bank_boundary+scratch_rot",
    "fork+scratch", "fork+scratch_rot", "scratch+scratch_rot",
}
POSITIVE = SINGLES | COMBOS_2
NEG_NI = {"Normal", "Invalid"}
NEG_OOD = {"CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst",
           "CenterCircle", "CrescentArc", "RingDots", "Row"}


def evaluate(df_var: pd.DataFrame) -> dict:
    pos = df_var[df_var["class_key"].isin(POSITIVE)].copy()
    if len(pos):
        f1 = f1_score(pos["true_labels"].tolist(), pos["pred_labels"].tolist(),
                      average="macro", zero_division=0)
    else:
        f1 = 0.0
    neg = df_var[df_var["class_key"].isin(NEG_NI | NEG_OOD)].copy()
    if len(neg):
        # FP if prediction is NOT Normal-decision (i.e. predicted some defect)
        fp = neg["decision_type"].isin(["single", "combo", "combo_collapsed"]).sum()
        ni = neg[neg["class_key"].isin(NEG_NI)]
        ood = neg[neg["class_key"].isin(NEG_OOD)]
        ni_fp = ni["decision_type"].isin(["single", "combo", "combo_collapsed"]).sum() if len(ni) else 0
        ood_fp = ood["decision_type"].isin(["single", "combo", "combo_collapsed"]).sum() if len(ood) else 0
        total_far = 100.0 * fp / len(neg)
        ni_far = 100.0 * ni_fp / max(1, len(ni))
        ood_far = 100.0 * ood_fp / max(1, len(ood))
    else:
        total_far = ni_far = ood_far = 0.0
    return {"bit_f1": f1, "total_far": total_far, "ni_far": ni_far, "ood_far": ood_far,
            "n_pos": len(pos), "n_neg": len(neg)}


def main():
    rows = []
    for pq in sorted(glob.glob("outputs/W2_pt*/T*/eval_v15direct/stage1_*/preds_chip.parquet")):
        parts = pq.split(os.sep)
        cell = parts[1]
        df = pd.read_parquet(pq)
        for var in sorted(df["cell_id"].unique()):
            sub = df[df["cell_id"] == var]
            m = evaluate(sub)
            rows.append({"cell": cell, "variant": var, **m})

    out = pd.DataFrame(rows)
    out["dual_gate"] = (out["bit_f1"] >= 0.99) & (out["total_far"] <= 0.5)
    out = out.sort_values(["dual_gate", "bit_f1", "total_far"], ascending=[False, False, True])

    print(f"Cells evaluated: {out['cell'].nunique()} / 49")
    print(f"Dual-gate pass (≥0.99 ∧ ≤0.5%): {out['dual_gate'].sum()}")
    print()
    print("TOP 20 by bit_F1:")
    cols = ["cell", "variant", "bit_f1", "total_far", "ni_far", "ood_far", "dual_gate"]
    print(out[cols].head(20).to_string(index=False, formatters={
        "bit_f1": "{:.4f}".format, "total_far": "{:.2f}".format,
        "ni_far": "{:.2f}".format, "ood_far": "{:.2f}".format,
    }))
    print()
    print("DUAL-GATE PASS:")
    g = out[out["dual_gate"]]
    if len(g):
        print(g[cols].to_string(index=False, formatters={
            "bit_f1": "{:.4f}".format, "total_far": "{:.2f}".format,
            "ni_far": "{:.2f}".format, "ood_far": "{:.2f}".format,
        }))
    else:
        print("(none)")

    csv = "outputs/_W2_aggregate.csv"
    out.to_csv(csv, index=False)
    print(f"\nSaved: {csv}")


if __name__ == "__main__":
    main()
