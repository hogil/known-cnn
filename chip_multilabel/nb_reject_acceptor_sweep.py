#!/usr/bin/env python3
"""Sweep POS-pattern NB acceptors for chip multilabel prediction rejects.

This is a diagnostic sidecar: fit only known POS probability patterns
(single/combo), then reject any sample that does not match those patterns.
Normal/Invalid/OOD rows are never used for fitting.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


BITS = ["bank_boundary", "fork", "scratch", "scratch_rot"]
PROB_COLS = [f"prob_{b}" for b in BITS]
SINGLE = BITS[:]
COMBO = [
    "bank_boundary+fork",
    "bank_boundary+scratch",
    "bank_boundary+scratch_rot",
    "fork+scratch",
    "fork+scratch_rot",
    "scratch+scratch_rot",
]
POS = SINGLE + COMBO


def _parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_labels(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text or text in {"[]", "None", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = text.split("+")
    if isinstance(parsed, str):
        return [parsed] if parsed else []
    return [str(x) for x in parsed]


def labels_from_class_key(class_key: object) -> list[str]:
    key = str(class_key)
    if key in POS:
        return key.split("+")
    return []


def labels_to_vec(labels: Iterable[str]) -> np.ndarray:
    present = set(labels)
    return np.array([1 if b in present else 0 for b in BITS], dtype=np.int32)


def bit_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for i, bit in enumerate(BITS):
        yt = y_true[:, i].astype(bool)
        yp = y_pred[:, i].astype(bool)
        tp = int(np.logical_and(yt, yp).sum())
        fp = int(np.logical_and(~yt, yp).sum())
        fn = int(np.logical_and(yt, ~yp).sum())
        denom = 2 * tp + fp + fn
        scores[bit] = 0.0 if denom == 0 else (2 * tp / denom)
    return float(np.mean(list(scores.values()))), scores


def read_preds(path: Path, cell: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in ["class_key", "pred_labels", *PROB_COLS] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")
    if cell and "cell_id" in df.columns:
        df = df[df["cell_id"] == cell].copy()
    elif "cell_id" in df.columns and df["cell_id"].nunique() > 1:
        first = str(df["cell_id"].iloc[0])
        df = df[df["cell_id"] == first].copy()
    if df.empty:
        raise SystemExit(f"{path} has no rows after cell filter")
    return df.reset_index(drop=True)


def split_pos_calib_eval(
    df: pd.DataFrame, calib_per_pos_class: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    calib_idx: list[int] = []
    eval_idx: list[int] = []
    class_key = df["class_key"].astype(str)
    for cls in POS:
        idx = df.index[class_key == cls].to_numpy()
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        n = min(calib_per_pos_class, max(1, len(idx) // 2))
        calib_idx.extend(idx[:n])
        eval_idx.extend(idx[n:])
    eval_idx.extend(df.index[~class_key.isin(POS)].to_numpy())
    return df.loc[calib_idx].reset_index(drop=True), df.loc[eval_idx].reset_index(drop=True)


def _x(df: pd.DataFrame) -> np.ndarray:
    return np.clip(df.loc[:, PROB_COLS].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)


def fit_gnb(df: pd.DataFrame, classes: list[str], var_smoothing: float) -> GaussianNB:
    mask = df["class_key"].astype(str).isin(classes)
    if int(mask.sum()) < 2:
        raise SystemExit(f"not enough calibration rows for {classes}")
    return GaussianNB(var_smoothing=var_smoothing).fit(
        _x(df.loc[mask]), df.loc[mask, "class_key"].astype(str).to_numpy()
    )


def joint_ll(nb: GaussianNB, df: pd.DataFrame, include_prior: bool) -> np.ndarray:
    x = _x(df)
    # sklearn returns log prior + class-conditional Gaussian LL. For acceptors
    # the prior can be undesirable, so allow a no-prior variant.
    ll = nb._joint_log_likelihood(x)
    if not include_prior:
        ll = ll - np.log(nb.class_prior_.reshape(1, -1))
    return ll


def class_thresholds(
    nb: GaussianNB, calib: pd.DataFrame, classes: list[str], q: float, include_prior: bool
) -> dict[str, float]:
    out: dict[str, float] = {}
    ll = joint_ll(nb, calib, include_prior)
    class_to_col = {str(c): i for i, c in enumerate(nb.classes_)}
    keys = calib["class_key"].astype(str).to_numpy()
    for cls in classes:
        if cls not in class_to_col:
            continue
        vals = ll[keys == cls, class_to_col[cls]]
        if len(vals):
            out[cls] = float(np.quantile(vals, q))
    return out


def accept_by_class_threshold(
    nb: GaussianNB, df: pd.DataFrame, thresholds: dict[str, float], include_prior: bool
) -> np.ndarray:
    ll = joint_ll(nb, df, include_prior)
    accepted = np.zeros(len(df), dtype=bool)
    for cls, tau in thresholds.items():
        col = list(nb.classes_).index(cls)
        accepted |= ll[:, col] >= tau
    return accepted


def pred_labels_to_class_key(value: object) -> str:
    labels = [x for x in parse_labels(value) if x in BITS]
    if not labels:
        return ""
    labels = sorted(set(labels), key=BITS.index)
    if len(labels) > 2:
        return ""
    key = "+".join(labels)
    return key if key in POS else ""


def accept_predicted_class_threshold(
    nb: GaussianNB, df: pd.DataFrame, thresholds: dict[str, float], include_prior: bool
) -> np.ndarray:
    ll = joint_ll(nb, df, include_prior)
    class_to_col = {str(c): i for i, c in enumerate(nb.classes_)}
    accepted = np.zeros(len(df), dtype=bool)
    pred_classes = [pred_labels_to_class_key(v) for v in df["pred_labels"]]
    for i, cls in enumerate(pred_classes):
        if cls not in thresholds or cls not in class_to_col:
            continue
        accepted[i] = bool(ll[i, class_to_col[cls]] >= thresholds[cls])
    return accepted


def accept_predicted_class_split_threshold(
    single_nb: GaussianNB,
    combo_nb: GaussianNB,
    df: pd.DataFrame,
    single_thresholds: dict[str, float],
    combo_thresholds: dict[str, float],
    include_prior: bool,
) -> np.ndarray:
    single_ll = joint_ll(single_nb, df, include_prior)
    combo_ll = joint_ll(combo_nb, df, include_prior)
    single_cols = {str(c): i for i, c in enumerate(single_nb.classes_)}
    combo_cols = {str(c): i for i, c in enumerate(combo_nb.classes_)}
    accepted = np.zeros(len(df), dtype=bool)
    pred_classes = [pred_labels_to_class_key(v) for v in df["pred_labels"]]
    for i, cls in enumerate(pred_classes):
        if cls in single_thresholds and cls in single_cols:
            accepted[i] = bool(single_ll[i, single_cols[cls]] >= single_thresholds[cls])
        elif cls in combo_thresholds and cls in combo_cols:
            accepted[i] = bool(combo_ll[i, combo_cols[cls]] >= combo_thresholds[cls])
    return accepted


def candidate_classes_from_pred(value: object) -> list[str]:
    labels = [x for x in parse_labels(value) if x in BITS]
    labels = sorted(set(labels), key=BITS.index)
    if not labels:
        return []
    if len(labels) == 1:
        bit = labels[0]
        # If raw says one bit, it may be a true single or an OOD/weak combo
        # tail. Check the single plus all valid two-bit patterns containing it.
        return [bit] + [c for c in COMBO if bit in c.split("+")]
    if len(labels) == 2:
        exact = "+".join(labels)
        # If raw says two bits, check exact combo and the two component singles.
        # This keeps valid weak-one-bit samples from being rejected too eagerly.
        out = []
        if exact in COMBO:
            out.append(exact)
        out.extend(labels)
        return out
    # Three/four predicted bits are invalid for this task. Let reject handle them.
    return []


def accept_candidate_threshold(
    nb: GaussianNB, df: pd.DataFrame, thresholds: dict[str, float], include_prior: bool
) -> np.ndarray:
    ll = joint_ll(nb, df, include_prior)
    class_to_col = {str(c): i for i, c in enumerate(nb.classes_)}
    accepted = np.zeros(len(df), dtype=bool)
    for i, pred_value in enumerate(df["pred_labels"]):
        for cls in candidate_classes_from_pred(pred_value):
            if cls not in thresholds or cls not in class_to_col:
                continue
            if ll[i, class_to_col[cls]] >= thresholds[cls]:
                accepted[i] = True
                break
    return accepted


def accept_candidate_split_threshold(
    single_nb: GaussianNB,
    combo_nb: GaussianNB,
    df: pd.DataFrame,
    single_thresholds: dict[str, float],
    combo_thresholds: dict[str, float],
    include_prior: bool,
) -> np.ndarray:
    single_ll = joint_ll(single_nb, df, include_prior)
    combo_ll = joint_ll(combo_nb, df, include_prior)
    single_cols = {str(c): i for i, c in enumerate(single_nb.classes_)}
    combo_cols = {str(c): i for i, c in enumerate(combo_nb.classes_)}
    accepted = np.zeros(len(df), dtype=bool)
    for i, pred_value in enumerate(df["pred_labels"]):
        for cls in candidate_classes_from_pred(pred_value):
            if cls in single_thresholds and cls in single_cols and single_ll[i, single_cols[cls]] >= single_thresholds[cls]:
                accepted[i] = True
                break
            if cls in combo_thresholds and cls in combo_cols and combo_ll[i, combo_cols[cls]] >= combo_thresholds[cls]:
                accepted[i] = True
                break
    return accepted


def accept_global(nb: GaussianNB, calib: pd.DataFrame, eval_df: pd.DataFrame, q: float, include_prior: bool) -> np.ndarray:
    calib_ll = joint_ll(nb, calib, include_prior).max(axis=1)
    tau = float(np.quantile(calib_ll, q))
    return joint_ll(nb, eval_df, include_prior).max(axis=1) >= tau


def summarize(df: pd.DataFrame, accepted: np.ndarray) -> dict[str, object]:
    true = np.stack([labels_to_vec(labels_from_class_key(k)) for k in df["class_key"]])
    pred = np.stack([labels_to_vec(parse_labels(v)) for v in df["pred_labels"]])
    pred_reject = pred.copy()
    pred_reject[~accepted, :] = 0
    is_pos = true.sum(axis=1) > 0
    is_neg = ~is_pos
    is_single = df["class_key"].astype(str).isin(SINGLE).to_numpy()
    is_combo = df["class_key"].astype(str).isin(COMBO).to_numpy()
    pred_any = pred.sum(axis=1) > 0
    pred_any_reject = pred_reject.sum(axis=1) > 0
    f1_all, per_bit = bit_f1(true, pred_reject)
    if int(accepted.sum()) > 0:
        f1_accept, _ = bit_f1(true[accepted], pred[accepted])
    else:
        f1_accept = 0.0
    class_rows = []
    keys = df["class_key"].astype(str).to_numpy()
    for cls in sorted(set(keys)):
        m = keys == cls
        class_rows.append(
            {
                "class": cls,
                "n": int(m.sum()),
                "accepted": int(np.logical_and(m, accepted).sum()),
                "rejected": int(np.logical_and(m, ~accepted).sum()),
                "coverage": float(accepted[m].mean()),
                "false_reject_pos": int(np.logical_and(m, np.logical_and(is_pos, ~accepted)).sum()),
                "false_accept_neg": int(
                    np.logical_and(m, np.logical_and(is_neg, np.logical_and(accepted, pred_any))).sum()
                ),
            }
        )
    return {
        "bit_F1_reject": f1_all,
        "bit_F1_accept": f1_accept,
        "post_FAR": float(100.0 * pred_any_reject[is_neg].mean()) if int(is_neg.sum()) else 0.0,
        "accepted_neg_FAR": float(
            100.0
            * np.logical_and(np.logical_and(is_neg, accepted), pred_any).sum()
            / max(1, int(np.logical_and(is_neg, accepted).sum()))
        ),
        "pos_cov": float(accepted[is_pos].mean()) if int(is_pos.sum()) else 0.0,
        "single_cov": float(accepted[is_single].mean()) if int(is_single.sum()) else 0.0,
        "combo_cov": float(accepted[is_combo].mean()) if int(is_combo.sum()) else 0.0,
        "neg_cov": float(accepted[is_neg].mean()) if int(is_neg.sum()) else 0.0,
        "false_reject_pos": int(np.logical_and(is_pos, ~accepted).sum()),
        "false_accept_neg": int(np.logical_and(is_neg, np.logical_and(accepted, pred_any)).sum()),
        "per_bit": per_bit,
        "class_rows": class_rows,
    }


def raw_metrics(df: pd.DataFrame) -> dict[str, float]:
    true = np.stack([labels_to_vec(labels_from_class_key(k)) for k in df["class_key"]])
    pred = np.stack([labels_to_vec(parse_labels(v)) for v in df["pred_labels"]])
    is_neg = true.sum(axis=1) == 0
    f1, _ = bit_f1(true, pred)
    return {
        "raw_allrow_bit_F1": f1,
        "raw_FAR": float(100.0 * (pred.sum(axis=1) > 0)[is_neg].mean()) if int(is_neg.sum()) else 0.0,
    }


def run(args: argparse.Namespace) -> None:
    df = read_preds(args.preds, args.cell)
    calib, eval_df = split_pos_calib_eval(df, args.calib_per_pos_class, args.seed)
    qs = _parse_floats(args.quantiles)
    smoothings = _parse_floats(args.var_smoothing)
    rows: list[dict[str, object]] = []
    raw = raw_metrics(eval_df)

    for vs in smoothings:
        nb10 = fit_gnb(calib, POS, vs)
        nbs = fit_gnb(calib, SINGLE, vs)
        nbc = fit_gnb(calib, COMBO, vs)
        for include_prior in [False, True]:
            prior_tag = "with_prior" if include_prior else "no_prior"
            for q in qs:
                for mode, accepted in [
                    (
                        "ten_class_global",
                        accept_global(nb10, calib[calib["class_key"].astype(str).isin(POS)], eval_df, q, include_prior),
                    ),
                    (
                        "ten_class_class_tau",
                        accept_by_class_threshold(
                            nb10, eval_df, class_thresholds(nb10, calib, POS, q, include_prior), include_prior
                        ),
                    ),
                    (
                        "single_or_combo_global",
                        np.logical_or(
                            accept_global(nbs, calib[calib["class_key"].astype(str).isin(SINGLE)], eval_df, q, include_prior),
                            accept_global(nbc, calib[calib["class_key"].astype(str).isin(COMBO)], eval_df, q, include_prior),
                        ),
                    ),
                    (
                        "single_or_combo_class_tau",
                        np.logical_or(
                            accept_by_class_threshold(
                                nbs, eval_df, class_thresholds(nbs, calib, SINGLE, q, include_prior), include_prior
                            ),
                            accept_by_class_threshold(
                                nbc, eval_df, class_thresholds(nbc, calib, COMBO, q, include_prior), include_prior
                            ),
                        ),
                    ),
                    (
                        "ten_class_pred_tau",
                        accept_predicted_class_threshold(
                            nb10, eval_df, class_thresholds(nb10, calib, POS, q, include_prior), include_prior
                        ),
                    ),
                    (
                        "single_or_combo_pred_tau",
                        accept_predicted_class_split_threshold(
                            nbs,
                            nbc,
                            eval_df,
                            class_thresholds(nbs, calib, SINGLE, q, include_prior),
                            class_thresholds(nbc, calib, COMBO, q, include_prior),
                            include_prior,
                        ),
                    ),
                    (
                        "ten_class_candidate_tau",
                        accept_candidate_threshold(
                            nb10, eval_df, class_thresholds(nb10, calib, POS, q, include_prior), include_prior
                        ),
                    ),
                    (
                        "single_or_combo_candidate_tau",
                        accept_candidate_split_threshold(
                            nbs,
                            nbc,
                            eval_df,
                            class_thresholds(nbs, calib, SINGLE, q, include_prior),
                            class_thresholds(nbc, calib, COMBO, q, include_prior),
                            include_prior,
                        ),
                    ),
                ]:
                    s = summarize(eval_df, accepted)
                    rows.append(
                        {
                            "mode": mode,
                            "prior": prior_tag,
                            "q": q,
                            "var_smoothing": vs,
                            **raw,
                            **{k: v for k, v in s.items() if k not in {"class_rows", "per_bit"}},
                            "bb_F1": s["per_bit"]["bank_boundary"],
                            "fk_F1": s["per_bit"]["fork"],
                            "sc_F1": s["per_bit"]["scratch"],
                            "sr_F1": s["per_bit"]["scratch_rot"],
                        }
                    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(out_dir / "nb_acceptor_sweep.csv", index=False)
    # Rank by operational goal: FAR first, then F1, then low false reject.
    ranked = df_rows.sort_values(
        ["post_FAR", "bit_F1_reject", "false_reject_pos"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    ranked.head(args.top).to_csv(out_dir / "nb_acceptor_top.csv", index=False)

    best = ranked.iloc[0].to_dict()
    # Recompute best class rows for the report.
    vs = float(best["var_smoothing"])
    q = float(best["q"])
    include_prior = best["prior"] == "with_prior"
    nb10 = fit_gnb(calib, POS, vs)
    nbs = fit_gnb(calib, SINGLE, vs)
    nbc = fit_gnb(calib, COMBO, vs)
    mode = str(best["mode"])
    if mode == "ten_class_global":
        accepted = accept_global(nb10, calib[calib["class_key"].astype(str).isin(POS)], eval_df, q, include_prior)
    elif mode == "ten_class_class_tau":
        accepted = accept_by_class_threshold(nb10, eval_df, class_thresholds(nb10, calib, POS, q, include_prior), include_prior)
    elif mode == "single_or_combo_global":
        accepted = np.logical_or(
            accept_global(nbs, calib[calib["class_key"].astype(str).isin(SINGLE)], eval_df, q, include_prior),
            accept_global(nbc, calib[calib["class_key"].astype(str).isin(COMBO)], eval_df, q, include_prior),
        )
    elif mode == "single_or_combo_class_tau":
        accepted = np.logical_or(
            accept_by_class_threshold(nbs, eval_df, class_thresholds(nbs, calib, SINGLE, q, include_prior), include_prior),
            accept_by_class_threshold(nbc, eval_df, class_thresholds(nbc, calib, COMBO, q, include_prior), include_prior),
        )
    elif mode == "ten_class_pred_tau":
        accepted = accept_predicted_class_threshold(
            nb10, eval_df, class_thresholds(nb10, calib, POS, q, include_prior), include_prior
        )
    elif mode == "single_or_combo_pred_tau":
        accepted = accept_predicted_class_split_threshold(
            nbs,
            nbc,
            eval_df,
            class_thresholds(nbs, calib, SINGLE, q, include_prior),
            class_thresholds(nbc, calib, COMBO, q, include_prior),
            include_prior,
        )
    elif mode == "ten_class_candidate_tau":
        accepted = accept_candidate_threshold(
            nb10, eval_df, class_thresholds(nb10, calib, POS, q, include_prior), include_prior
        )
    else:
        accepted = accept_candidate_split_threshold(
            nbs,
            nbc,
            eval_df,
            class_thresholds(nbs, calib, SINGLE, q, include_prior),
            class_thresholds(nbc, calib, COMBO, q, include_prior),
            include_prior,
        )
    class_rows = pd.DataFrame(summarize(eval_df, accepted)["class_rows"])
    class_rows.to_csv(out_dir / "nb_acceptor_best_class_rows.csv", index=False)

    report = [
        "# NB POS-Pattern Acceptor Sweep",
        "",
        f"preds={args.preds}",
        f"cell={args.cell}",
        f"calib_per_pos_class={args.calib_per_pos_class}",
        f"seed={args.seed}",
        "",
        "## Best",
        "",
        "| mode | prior | q | var_smoothing | NB bit_F1 | NB FAR | pos_cov | single_cov | combo_cov | false_reject_pos | false_accept_neg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {best['mode']} | {best['prior']} | {float(best['q']):g} | {float(best['var_smoothing']):.0e} "
            f"| {float(best['bit_F1_reject']):.4f} | {float(best['post_FAR']):.2f}% "
            f"| {float(best['pos_cov']):.4f} | {float(best['single_cov']):.4f} "
            f"| {float(best['combo_cov']):.4f} | {int(best['false_reject_pos'])} | {int(best['false_accept_neg'])} |"
        ),
        "",
        "## Top Candidates",
        "",
        "| rank | mode | prior | q | var | F1 | FAR | pos_cov | combo_cov | false_reject_pos | false_accept_neg |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, row in ranked.head(args.top).iterrows():
        report.append(
            f"| {i+1} | {row['mode']} | {row['prior']} | {float(row['q']):g} "
            f"| {float(row['var_smoothing']):.0e} | {float(row['bit_F1_reject']):.4f} "
            f"| {float(row['post_FAR']):.2f}% | {float(row['pos_cov']):.4f} "
            f"| {float(row['combo_cov']):.4f} | {int(row['false_reject_pos'])} | {int(row['false_accept_neg'])} |"
        )
    report.extend(
        [
            "",
            "## Best Class Rows",
            "",
            "| class | n | accepted | rejected | coverage | false_reject_pos | false_accept_neg |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in class_rows.sort_values(["false_reject_pos", "false_accept_neg"], ascending=[False, False]).iterrows():
        report.append(
            f"| {row['class']} | {int(row['n'])} | {int(row['accepted'])} | {int(row['rejected'])} "
            f"| {float(row['coverage']):.4f} | {int(row['false_reject_pos'])} | {int(row['false_accept_neg'])} |"
        )
    (out_dir / "nb_acceptor_report.md").write_text("\n".join(report), encoding="utf-8")
    print(out_dir / "nb_acceptor_report.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cell", default="T0__I10")
    ap.add_argument("--calib-per-pos-class", type=int, default=800)
    ap.add_argument("--seed", type=int, default=20260608)
    ap.add_argument("--quantiles", default="0.0001,0.0005,0.001,0.002,0.005,0.01,0.015,0.02,0.03,0.05,0.075,0.1")
    ap.add_argument("--var-smoothing", default="1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3")
    ap.add_argument("--top", type=int, default=20)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
