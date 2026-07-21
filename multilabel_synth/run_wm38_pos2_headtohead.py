"""WM38 pos2-only head-to-head at the protocol anchor (g=3, 9x9).
Decisive gate: on the pos2 (exactly-2-defect) eval, does FCM-PM beat
single-only / cutmix / mixup / summation(overlay) on the STANDARD metric
(iso-FAR bit-F1 @ NORMAL-FAR 1/5/10%)? Also reports all-orders eval so the
effect of pos2-scoping is visible. oracle = real-mixed upper reference.

Same protocol as run_wm38_margin (synth_wm38 / SmallCNN / val-margin pick /
uniform neg-target). No test-set config selection: g/grid fixed at protocol
anchor for every arm. Self-contained.
"""
import argparse, copy, csv as csvmod, os, collections
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .datasets.mixedwm38 import load_wm38, split_groups
from .synthesis.wm38_arms import synth_wm38
from .models.small_cnn import SmallCNN
from .metrics import bit_f1, pos_neg_prob

ARMS = ["single_only", "fcm_pm", "cutmix", "mixup", "overlay", "oracle"]

def _loader(X, Y, bs, sh): return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)), batch_size=bs, shuffle=sh)
def _predict(m, X, dev, bs=128):
    m.eval(); P=[]
    with torch.no_grad():
        for xb, in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=bs):
            P.append(torch.sigmoid(m(xb.to(dev))).cpu().numpy())
    return np.concatenate(P)

def build_train(arm, sX, sY, oX, oY, a, seed):
    r = np.random.default_rng(seed)
    if arm == "oracle":
        pk = r.choice(len(oX), size=min(a.n_train, len(oX)), replace=False); bX, bY = oX[pk], oY[pk]
    elif arm == "fcm_pm":
        bX, bY = synth_wm38("fcm", sX, sY, a.n_train, seed, grid=9, n_groups=3, pair_mask=True, cell_layout="checkerboard")
    else:
        bX, bY = synth_wm38(arm, sX, sY, a.n_train, seed, grid=9, n_groups=3)
    aug = r.choice(len(sX), size=min(a.n_single_aug, len(sX)), replace=False)
    trX = np.concatenate([bX, sX[aug]]); trY = np.concatenate([bY, sY[aug]])
    if a.n_synth_normal > 0:
        np_ = r.choice(len(sX), size=min(a.n_synth_normal, len(sX)), replace=False)
        nX = np.minimum(sX[np_], 0.5)
        trX = np.concatenate([trX, nX]); trY = np.concatenate([trY, np.zeros((len(nX), sY.shape[1]), np.float32)])
    if a.neg_target > 0: trY = trY + (1.0 - trY) * a.neg_target
    return trX.astype(np.float32), trY.astype(np.float32)

def build_val(sX, sY, seed):
    r = np.random.default_rng(seed + 4242)
    vs = r.choice(len(sX), 300, replace=False); vn = r.choice(len(sX), 300, replace=False)
    vX1, vY1 = synth_wm38("overlay", sX, sY, 600, seed + 777)
    vaX = np.concatenate([vX1, sX[vs], np.minimum(sX[vn], 0.5)])
    vaY = np.concatenate([vY1, sY[vs], np.zeros((300, sY.shape[1]), np.float32)])
    return vaX.astype(np.float32), vaY.astype(np.float32)

def isofar(teP, teY, nrmP):
    nm = nrmP.max(axis=1); out = {}
    for al in (0.01, 0.05, 0.10):
        tau = float(np.quantile(nm, 1.0 - al))
        out[f"F1@FAR{int(al*100)}"] = float(bit_f1(teP, teY, thr=tau))
    return out

def run_arm(arm, sX, sY, oX, oY, evals, nrmX, a, seed):
    trX, trY = build_train(arm, sX, sY, oX, oY, a, seed)
    vaX, vaY = build_val(sX, sY, seed)
    torch.manual_seed(seed)
    m = SmallCNN(num_classes=sY.shape[1], in_ch=1).to(a.device)
    opt = torch.optim.Adam(m.parameters(), lr=a.lr); lf = nn.BCEWithLogitsLoss()
    best, st = -1e9, None
    for ep in range(1, a.epochs + 1):
        m.train()
        for xb, yb in _loader(trX, trY, a.bs, True):
            xb, yb = xb.to(a.device), yb.to(a.device); opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
        P = _predict(m, vaX, a.device); pos, neg = pos_neg_prob(P, vaY)
        if pos - neg > best: best, st = pos - neg, copy.deepcopy(m.state_dict())
    m.load_state_dict(st)
    nrmP = _predict(m, nrmX, a.device)
    rows = []
    for scope, (teX, teY) in evals.items():
        teP = _predict(m, teX, a.device)
        row = {"arm": arm, "seed": seed, "scope": scope, "n_eval": len(teX),
               "raw_bitF1": float(bit_f1(teP, teY)),
               "normal_FAR": float((nrmP >= 0.5).any(axis=1).mean())}
        row.update(isofar(teP, teY, nrmP))
        rows.append(row)
        print(f"[{arm:10s} s{seed} {scope:8s}] rawF1={row['raw_bitF1']:.3f} FAR={row['normal_FAR']:.3f} "
              f"F1@1/5/10={row['F1@FAR1']:.3f}/{row['F1@FAR5']:.3f}/{row['F1@FAR10']:.3f}", flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-single-aug", type=int, default=2000)
    ap.add_argument("--n-synth-normal", type=int, default=2000)
    ap.add_argument("--neg-target", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/wm38_pos2_headtohead.csv")
    a = ap.parse_args()

    X, Y = load_wm38(); g = split_groups(Y)
    rng = np.random.default_rng(12345)
    mixed = rng.permutation(g["mixed"]); half = len(mixed) // 2
    test_pool = mixed[half:]
    ti = rng.choice(test_pool, size=min(a.n_test, len(test_pool)), replace=False)
    teX, teY = X[ti], Y[ti]
    pos2 = teY.sum(axis=1) == 2
    evals = {"all": (teX, teY), "pos2": (teX[pos2], teY[pos2])}
    oX, oY = X[mixed[:half]], Y[mixed[:half]]
    nrmX = X[g["normal"]]; sX, sY = X[g["single"]], Y[g["single"]]
    print(f"device={a.device} singles={len(sX)} test_all={len(teX)} test_pos2={int(pos2.sum())} "
          f"oracle={len(oX)} normals={len(nrmX)}", flush=True)

    rows = []
    for arm in a.arms:
        for s in a.seeds:
            rows += run_arm(arm, sX, sY, oX, oY, evals, nrmX, a, s)
    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[OUT] {os.path.abspath(a.out_csv)}", flush=True)

    # summary: mean over seeds, F1@FAR5 as headline standard metric
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows: agg[(r["scope"], r["arm"])]["f5"].append(r["F1@FAR5"]); agg[(r["scope"], r["arm"])]["raw"].append(r["raw_bitF1"])
    for scope in ("pos2", "all"):
        print(f"\n=== WM38 {scope} eval | F1@FAR5% (standard) + rawF1, {len(a.seeds)}-seed mean ===")
        arm_scores = [(arm, np.mean(agg[(scope, arm)]["f5"]), np.mean(agg[(scope, arm)]["raw"])) for arm in a.arms]
        for arm, f5, raw in sorted(arm_scores, key=lambda x: -x[1]):
            tag = " <- ours" if arm == "fcm_pm" else (" (upper ref)" if arm == "oracle" else "")
            print(f"  {arm:10s} F1@FAR5={f5:.4f}  rawF1={raw:.4f}{tag}")
        adm = [x for x in arm_scores if x[0] not in ("oracle",)]
        best = max(adm, key=lambda x: x[1]); fcm = [x for x in adm if x[0] == "fcm_pm"][0]
        print(f"  => best admissible @F1FAR5: {best[0]} ({best[1]:.4f}); FCM-PM {'WINS' if best[0]=='fcm_pm' else 'LOSES to '+best[0]} (gap {fcm[1]-best[1]:+.4f})")

if __name__ == "__main__":
    main()
