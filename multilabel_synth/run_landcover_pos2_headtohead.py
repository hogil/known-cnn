"""Land-cover (DLRSD) pos2-only head-to-head -- the PARTITION-domain gate.
On the exactly-2-class real multi-region tiles, does the partition operator
(FCM-PM analog) beat single-only / cutmix / mixup / overlay on the standard
metric (iso-FAR bit-F1 @ NORMAL-FAR 1/5/10%)? Reports pos2 AND all-orders so
the pos2-scoping effect is visible. Reuses the committed operator_match
land-cover pipeline (same data, synthesis, resnet18, protocol).
"""
import argparse, os, collections
import numpy as np
from .datasets.dlrsd import prepare_landcover
from .synthesis import landcover_ops as ops
from .run_operator_match_landcover import build_train_for_arm, train_model, _predict
from .metrics import bit_f1, compute_map

ARMS = ["single_only", "partition", "cutmix", "mixup", "overlay", "oracle"]
SUBSET = ["buildings", "pavement", "trees", "grass", "baresoil"]

def isofar(P, Y, nP):
    nm = nP.max(axis=1); out = {}
    for al in (0.01, 0.05, 0.10):
        tau = float(np.quantile(nm, 1.0 - al))
        out[f"F1@FAR{int(al*100)}"] = float(bit_f1(P, Y, thr=tau))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/landcover_pos2_headtohead.csv")
    a = ap.parse_args()
    import torch
    if a.device == "cuda" and not torch.cuda.is_available(): a.device = "cpu"

    cell, grid = 64, 2
    data = prepare_landcover("E:/data/dlrsd_extracted/DLRSD", subset=SUBSET, cell=cell,
                             canvas=cell*grid, n_eval=a.n_eval, n_oracle=1000,
                             per_class_cap=140, min_side=48, purity=0.6, seed=0)
    pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y, names, meta = data
    nc = len(names)
    evX = ops.real_to_chw(eval_imgs)
    nX, _ = ops.build_normal(400, cell, grid, nc)
    pos2 = eval_Y.sum(axis=1) == 2
    print(f"device={a.device} eval_all={len(eval_Y)} eval_pos2={int(pos2.sum())} classes={names}", flush=True)

    import csv as csvmod
    rows = []
    for arm in a.arms:
        for s in a.seeds:
            trX, trY = build_train_for_arm(arm, pool_imgs, pool_labels, oracle_imgs, oracle_Y,
                                           140, 2000, s, cell, grid, nc, (2,))
            model = train_model(trX, trY, a.epochs, a.bs, a.lr, a.device, s, nc, not a.no_pretrained)
            P = _predict(model, evX, a.bs, a.device); nP = _predict(model, nX, a.bs, a.device)
            for scope, mask in (("all", np.ones(len(eval_Y), bool)), ("pos2", pos2)):
                Pm, Ym = P[mask], eval_Y[mask]
                r = {"arm": arm, "seed": s, "scope": scope, "n_eval": int(mask.sum()),
                     "raw_bitF1": round(float(bit_f1(Pm, Ym)), 4),
                     "mAP": round(float(compute_map(Pm, Ym)), 4),
                     "normal_FAR": round(float((nP >= 0.5).any(1).mean()), 4)}
                r.update({k: round(v, 4) for k, v in isofar(Pm, Ym, nP).items()})
                rows.append(r)
                print(f"[{arm:11s} s{s} {scope:4s}] rawF1={r['raw_bitF1']:.3f} mAP={r['mAP']:.3f} "
                      f"F1@1/5/10={r['F1@FAR1']:.3f}/{r['F1@FAR5']:.3f}/{r['F1@FAR10']:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[OUT] {os.path.abspath(a.out_csv)}", flush=True)

    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        agg[(r["scope"], r["arm"])]["f5"].append(r["F1@FAR5"]); agg[(r["scope"], r["arm"])]["map"].append(r["mAP"])
    for scope in ("pos2", "all"):
        print(f"\n=== land-cover {scope} eval | F1@FAR5 + mAP, {len(a.seeds)}-seed mean ===")
        sc = [(arm, np.mean(agg[(scope, arm)]["f5"]), np.mean(agg[(scope, arm)]["map"])) for arm in a.arms]
        for arm, f5, mp in sorted(sc, key=lambda x: -x[1]):
            tag = " <- ours" if arm == "partition" else (" (upper ref)" if arm == "oracle" else "")
            print(f"  {arm:11s} F1@FAR5={f5:.4f}  mAP={mp:.4f}{tag}")
        adm = [x for x in sc if x[0] != "oracle"]; best = max(adm, key=lambda x: x[1])
        p = [x for x in adm if x[0] == "partition"][0]
        print(f"  => best admissible: {best[0]} ({best[1]:.4f}); partition {'WINS' if best[0]=='partition' else 'LOSES to '+best[0]} (gap {p[1]-best[1]:+.4f})")

if __name__ == "__main__":
    main()
