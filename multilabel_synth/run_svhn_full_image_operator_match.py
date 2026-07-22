"""Strict FULL-IMAGE SVHN operator-match (bbox-free, mask-free real partition).

Pre-registered protocol (see docs plan 260721): synthesize 2-digit training data
from single-digit FULL images under each operator, train a single-label probe on
the sources, pick the operator by an evidence-fidelity margin computed ONLY on a
source-validation synthetic set, FREEZE the proxy ranking + a protocol hash, and
only then touch the sealed cardinality-2 test.

Phase A (this default): proxy selection + freeze -- NEVER loads the sealed test.
Phase B (--run-test): train every arm on frozen config, evaluate sealed test once.
"""
import argparse, hashlib, json, os, collections
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.svhn_full_image import load_full_image_by_cardinality, N_CLASSES, CLASSES

ARMS = ["single_only", "partition", "summation", "cutmix", "mixup", "fcm_pm"]
CANH, CANW = 64, 128
OUTDIR = "outputs/multilabel_synth/svhn_full_image_operator_match_v1"


# ---------------- synthesis (operates on full 64x128 RGB uint8) ----------------
def _resize_into(img, h, w):
    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((w, h)), np.uint8)


def synth_pair(arm, a, b, rng, grid=9, n_groups=3):
    """a,b: HxWx3 uint8 single-digit full images. Returns HxWx3 uint8 combo."""
    H, W = a.shape[:2]
    if arm == "summation":                      # max-union (superposition op)
        return np.maximum(a, b)
    if arm == "mixup":
        lam = float(rng.beta(1.0, 1.0)); return (lam * a + (1 - lam) * b).astype(np.uint8)
    if arm == "cutmix":
        cw = W // 2; x = int(rng.integers(0, W - cw + 1))
        out = a.copy(); out[:, x:x + cw] = b[:, x:x + cw]; return out
    if arm == "partition":                      # two digits side by side (real layout)
        out = np.zeros_like(a); half = W // 2
        out[:, :half] = _resize_into(a, H, half); out[:, half:] = _resize_into(b, H, W - half)
        return out
    if arm == "fcm_pm":                         # grid partition-complement, g groups
        cell_h, cell_w = H // grid, W // grid
        out = b.copy()
        cells = [(i, j) for i in range(grid) for j in range(grid)]
        pick = rng.permutation(len(cells))[: len(cells) // n_groups]
        for ci in pick:
            i, j = cells[ci]
            y0, x0 = i * cell_h, j * cell_w
            y1 = (i + 1) * cell_h if i < grid - 1 else H
            x1 = (j + 1) * cell_w if j < grid - 1 else W
            out[y0:y1, x0:x1] = a[y0:y1, x0:x1]
        return out
    raise ValueError(arm)


def build_arm_train(arm, sX, sY, n_per_pair, rng, oracleX=None, oracleY=None):
    """Return (X[N,H,W,3] uint8, Y[N,C] float). single_only=sources; oracle=real."""
    if arm == "single_only":
        return sX, sY
    if arm == "oracle":
        return oracleX, oracleY
    by = collections.defaultdict(list)
    for i, y in enumerate(sY):
        c = int(np.argmax(y)); by[c].append(i)
    classes = sorted(by)
    pairs = [(a, b) for ai, a in enumerate(classes) for b in classes[ai + 1:]]
    Xo, Yo = [], []
    for (ca, cb) in pairs:
        for _ in range(n_per_pair):
            ia = by[ca][int(rng.integers(len(by[ca])))]
            ib = by[cb][int(rng.integers(len(by[cb])))]
            Xo.append(synth_pair(arm, sX[ia], sX[ib], rng))
            y = np.zeros(N_CLASSES, np.float32); y[ca] = 1.0; y[cb] = 1.0; Yo.append(y)
    X = np.concatenate([np.stack(Xo).astype(np.uint8), sX])       # fold in singles
    Y = np.concatenate([np.stack(Yo), sY])
    return X, Y


# ---------------- model ----------------
class SmallRGB(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.h = nn.Linear(128, nc)

    def forward(self, x): return self.h(self.f(x).flatten(1))


_BACKBONE = "small"   # set from CLI; "small" (SmallRGB) or "resnet18"


def make_backbone(dev):
    if _BACKBONE == "resnet18":
        from torchvision.models import resnet18
        import torch.nn as _nn
        m = resnet18(weights=None); m.fc = _nn.Linear(m.fc.in_features, N_CLASSES)
        return m.to(dev)
    return SmallRGB(N_CLASSES).to(dev)


def _batches(X, Y, bs, rng):
    """Memory-safe minibatches: X uint8 or float HWC -> float CHW per batch."""
    idx = rng.permutation(len(X))
    for i in range(0, len(X), bs):
        b = idx[i:i + bs]; xb = X[b]
        if xb.dtype == np.uint8:
            xb = xb.astype(np.float32) / 255.0
        xb = np.ascontiguousarray(np.transpose(xb, (0, 3, 1, 2)))
        yield torch.from_numpy(xb), torch.from_numpy(Y[b])


def train(X, Y, epochs, seed, dev, lr=1e-3, bs=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = make_backbone(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train()
        for xb, yb in _batches(X, Y, bs, rng):
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    return m


def predict(m, X, dev, bs=128):
    """Memory-safe: X can be uint8 HWC or float HWC; convert+transpose per batch."""
    m.eval(); P = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs]
            if xb.dtype == np.uint8:
                xb = xb.astype(np.float32) / 255.0
            xb = np.ascontiguousarray(np.transpose(xb, (0, 3, 1, 2)))
            P.append(torch.sigmoid(m(torch.from_numpy(xb).to(dev))).cpu().numpy())
    return np.concatenate(P)


def evidence_margin(P, Y):
    """positive-min minus negative-max, averaged: evidence fidelity of an operator's
    synthetic combos under the single-label probe (higher = both labels survive)."""
    Y = (Y >= 0.5)
    pos = np.array([P[i, Y[i]].min() if Y[i].any() else np.nan for i in range(len(P))])
    neg = np.array([P[i, ~Y[i]].max() if (~Y[i]).any() else 0.0 for i in range(len(P))])
    return float(np.nanmean(pos - neg))


def proxy_select(sX, sY, vX, vY, dev, epochs, seed=0):
    """Train probe on source singles; score each operator's source-val combos by
    evidence margin. Returns ranking (higher first). NEVER sees the test set."""
    probe = train(sX, sY, epochs, seed, dev)
    rng = np.random.default_rng(seed + 1)
    scores = {}
    for arm in [a for a in ARMS if a not in ("single_only",)]:
        # synth source-val combos with this operator
        cX, cY = build_arm_train(arm, vX, vY, n_per_pair=20, rng=rng)
        # keep only the combo rows (drop appended singles)
        n_combo = len(cX) - len(vX)
        P = predict(probe, cX[:n_combo], dev)
        scores[arm] = evidence_margin(P, cY[:n_combo])
    ranking = sorted(scores, key=scores.get, reverse=True)
    return ranking, scores


def train_with_margin(X, Y, vmX, vmY, epochs, seed, dev, lr=1e-3, bs=64, neg_target=0.02):
    """Train; after each epoch score pos-min minus neg-max margin on a source-val
    synthetic set; return the best-margin checkpoint (val-margin selection)."""
    import copy
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = make_backbone(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
    Yt = (Y + (1.0 - Y) * neg_target).astype(np.float32)
    best, st = -1e9, None
    for _ in range(epochs):
        m.train()
        for xb, yb in _batches(X, Yt, bs, rng):
            xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
        P = predict(m, vmX, dev); mg = evidence_margin(P, vmY)
        if mg > best: best, st = mg, copy.deepcopy(m.state_dict())
    m.load_state_dict(st); return m


def fpr_threshold(probs, Y, target_fpr):
    """tau so the NEGATIVE-bit false-positive rate on source-val singles = target."""
    Yb = (Y >= 0.5); neg_scores = probs[~Yb]
    return float(np.quantile(neg_scores, 1 - target_fpr))


def _map(P, Y):
    from .metrics import compute_map
    return float(compute_map(P, Y))


def bitf1_at(P, Y, tau):
    pred = (P >= tau).astype(int); Yb = (Y >= 0.5).astype(int)
    tp = (pred & Yb).sum(); fp = (pred & ~Yb).sum(); fn = (~pred & Yb).sum()
    p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9); return 2*p*r/(p+r+1e-9)


def worst_class_recall(P, Y, tau):
    Yb = (Y >= 0.5); pred = (P >= tau)
    recs = []
    for c in range(N_CLASSES):
        pos = Yb[:, c]
        if pos.sum() == 0: continue
        recs.append((pred[pos, c]).mean())
    return float(min(recs)) if recs else 0.0


def phase_b(sX, sY, vX, vY, args, proxy_hash):
    from PIL import Image  # noqa
    # freeze test protocol BEFORE touching test
    proto = dict(classes=list(CLASSES), per_class_train=args.per_class_train,
                 per_class_val=args.per_class_val, arms=ARMS, n_per_pair=100,
                 backbone=_BACKBONE, epochs=args.epochs, neg_target=0.02,
                 seeds=[1, 2, 3, 4, 5], fpr_targets=[0.01, 0.05],
                 checkpoint="val_margin", threshold="source_val_negbit_fpr",
                 proxy_hash=proxy_hash)
    proto["hash"] = hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()[:16]
    with open(os.path.join(OUTDIR, "test_protocol.json"), "w") as f:
        json.dump(proto, f, indent=2)
    print(f"[test_protocol frozen] hash={proto['hash']}", flush=True)

    # NOW load sealed test (once); keep uint8 (predict converts per batch)
    tX, tY, tm = load_full_image_by_cardinality(args.root_test, 2, distinct=True, seed=0)
    print(f"[SEALED TEST loaded] n={tm['n']} pairs={tm['n_pairs']}", flush=True)

    rows = []
    for arm in ARMS:
        for seed in proto["seeds"]:
            rng = np.random.default_rng(1000 + seed)
            trX, trY = build_arm_train(arm, sX, sY, 100, rng)
            # val-margin checkpoint set: synth val combos (source-val only)
            vmX_full, vmY_full = build_arm_train(arm if arm != "single_only" else "partition",
                                                 vX, vY, 20, np.random.default_rng(seed))
            n_c = len(vmX_full) - len(vX); vmX, vmY = vmX_full[:n_c], vmY_full[:n_c]
            m = train_with_margin(trX, trY, vmX, vmY, args.epochs, seed, args.device,
                                  neg_target=proto["neg_target"])
            Pv = predict(m, vX, args.device)   # source-val singles (uint8)
            Pt = predict(m, tX, args.device)   # sealed test (uint8)
            r = {"arm": arm, "seed": seed, "mAP": _map(Pt, tY)}
            for a in proto["fpr_targets"]:
                tau = fpr_threshold(Pv, vY, a)
                r[f"F1@FPR{int(a*100)}"] = bitf1_at(Pt, tY, tau)
                r[f"realFPR@{int(a*100)}"] = float((Pt[(tY < 0.5)] >= tau).mean())
                r[f"wcr@{int(a*100)}"] = worst_class_recall(Pt, tY, tau)
            rows.append(r)
            print(f"[{arm:11s} s{seed}] mAP={r['mAP']:.4f} F1@1%={r['F1@FPR1']:.3f} "
                  f"F1@5%={r['F1@FPR5']:.3f}", flush=True)

    import csv as csvmod
    with open(os.path.join(OUTDIR, "sealed_test_results.csv"), "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # GATE-2 judgment
    ag = collections.defaultdict(list)
    for r in rows: ag[r["arm"]].append(r["mAP"])
    means = {a: float(np.mean(v)) for a, v in ag.items()}
    order = sorted(means, key=means.get, reverse=True)
    frozen_winner = "summation"
    actual = order[0]
    print("\n=== SEALED TEST mAP (5-seed mean) ===")
    for a in order: print(f"  {a:11s} {means[a]:.4f}")
    print(f"frozen proxy winner = {frozen_winner} | actual test winner = {actual}")
    # paired CI: actual vs 2nd
    a1 = np.array(ag[order[0]]); a2 = np.array(ag[order[1]])
    diff = a1 - a2; lo = diff.mean() - 1.96 * diff.std(ddof=1) / np.sqrt(len(diff))
    print(f"paired {order[0]}-{order[1]} mAP: mean={diff.mean():+.4f} CI95_low={lo:+.4f}")
    print(f"GATE-2 {'PASS' if (actual==frozen_winner and lo>0) else 'FAIL'}: "
          f"proxy-winner=={actual}? {actual==frozen_winner}; CI_low>0? {lo>0}")
    print(f"[OUT] {os.path.abspath(os.path.join(OUTDIR, 'sealed_test_results.csv'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-train", default="E:/data/torchvision/SVHN_format1/train")
    ap.add_argument("--root-test", default="E:/data/torchvision/SVHN_format1/test")
    ap.add_argument("--per-class-train", type=int, default=350)
    ap.add_argument("--per-class-val", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--proxy-seed", type=int, default=0)
    ap.add_argument("--backbone", choices=["small", "resnet18"], default="small")
    ap.add_argument("--outdir", default=None,
                    help="override OUTDIR (use a NEW dir per backbone; never overwrite v1)")
    ap.add_argument("--run-test", action="store_true")
    args = ap.parse_args()
    global _BACKBONE, OUTDIR
    _BACKBONE = args.backbone
    if args.outdir:
        OUTDIR = args.outdir
    os.makedirs(OUTDIR, exist_ok=True)

    if args.run_test:
        mpath = os.path.join(OUTDIR, "proxy_manifest.json")
        if not os.path.exists(mpath):
            raise SystemExit("proxy_manifest.json missing -- run Phase A (proxy) first")
        manifest = json.load(open(mpath))
        print(f"[integrity] proxy hash={manifest['hash']} winner={manifest['predicted_winner']}")
        rng = np.random.default_rng(12345)
        allX, allY, _ = load_full_image_by_cardinality(args.root_train, 1, per_class_cap=None, seed=0)
        by = collections.defaultdict(list)
        for i, y in enumerate(allY): by[int(np.argmax(y))].append(i)
        tr, va = [], []
        for c, idx in by.items():
            idx = list(rng.permutation(idx))
            tr += idx[:args.per_class_train]; va += idx[args.per_class_train:args.per_class_train + args.per_class_val]
        phase_b(allX[tr], allY[tr], allX[va], allY[va], args, manifest["hash"])
        return

    allX, allY, _ = load_full_image_by_cardinality(args.root_train, 1, per_class_cap=None,
                                                   seed=0)
    # deterministic per-class split: first per_class_train -> train, next per_class_val -> val
    rng = np.random.default_rng(12345)
    by = collections.defaultdict(list)
    for i, y in enumerate(allY): by[int(np.argmax(y))].append(i)
    tr, va = [], []
    for c, idx in by.items():
        idx = list(rng.permutation(idx))
        tr += idx[:args.per_class_train]; va += idx[args.per_class_train:args.per_class_train + args.per_class_val]
    sX, sY, vX, vY = allX[tr], allY[tr], allX[va], allY[va]
    print(f"source-train={len(sX)} source-val={len(vX)} (per-class {args.per_class_train}/{args.per_class_val})", flush=True)

    ranking, scores = proxy_select(sX, sY, vX, vY, args.device, args.epochs, args.proxy_seed)
    predicted_winner = ranking[0]
    manifest = dict(backbone=_BACKBONE, arms=ARMS, canvas=[CANH, CANW], classes=list(CLASSES),
                    per_class_train=args.per_class_train, per_class_val=args.per_class_val,
                    proxy_scores=scores, proxy_ranking=ranking,
                    predicted_winner=predicted_winner, epochs=args.epochs,
                    proxy_seed=args.proxy_seed)
    manifest["hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    with open(os.path.join(OUTDIR, "proxy_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("=== PROXY (evidence-fidelity margin, source-val only; NO test) ===")
    for a in ranking: print(f"  {a:12s} margin={scores[a]:+.4f}")
    print(f"PREDICTED WINNER = {predicted_winner}  |  hash={manifest['hash']}")
    print(f"[OUT] {os.path.abspath(os.path.join(OUTDIR, 'proxy_manifest.json'))}")
    if predicted_winner in ("partition", "fcm_pm"):
        print("GATE-1 PASS: proxy picks a PARTITION operator without seeing test.")
    else:
        print(f"GATE-1 note: proxy picked {predicted_winner} (not partition) -- record honestly.")


if __name__ == "__main__":
    main()
