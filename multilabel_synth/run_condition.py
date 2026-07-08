"""Generic condition-type multi-label runner over a preloaded
{single,multi,normal} split (works for ChestX-ray14 gray and Plant RGB).
Arms are content-blind: overlay(max), cutmix, mixup, single_only, oracle
(real multi minus held-out combos). synthetic-normal + neg-target optional.
Reports TRAIN + EVAL bit_F1/FAR/exact/pos/neg + real-NORMAL FAR.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .models.small_cnn import SmallCNN
from .metrics import bit_f1, far, exact_match, pos_neg_prob


def _pairs(Y, rng, n):
    lab = Y.argmax(1)
    out, ii, jj = [], rng.integers(0, len(Y), n * 4), rng.integers(0, len(Y), n * 4)
    for a, b in zip(ii, jj):
        if lab[a] != lab[b]:
            out.append((int(a), int(b)))
            if len(out) >= n:
                break
    return out


def synth(arm, X, Y, n, seed, frac=0.25, alpha=1.0):
    rng = np.random.default_rng(seed)
    if arm == "single_only":
        idx = rng.integers(0, len(X), n)
        return X[idx].copy(), Y[idx].copy()
    ox, oy = [], []
    S = X.shape[-1]
    for a, b in _pairs(Y, rng, n):
        ca, cb = X[a], X[b]
        if arm == "overlay":
            img = np.maximum(ca, cb); y = np.maximum(Y[a], Y[b])
        elif arm == "cutmix":
            s = max(1, int(round(S * frac ** 0.5)))
            y0, x0 = int(rng.integers(0, S - s + 1)), int(rng.integers(0, S - s + 1))
            img = ca.copy(); img[:, y0:y0 + s, x0:x0 + s] = cb[:, y0:y0 + s, x0:x0 + s]
            y = np.maximum(Y[a], Y[b])
        elif arm == "mixup":
            lam = float(rng.beta(alpha, alpha))
            img = lam * ca + (1 - lam) * cb; y = lam * Y[a] + (1 - lam) * Y[b]
        else:
            raise ValueError(arm)
        ox.append(img); oy.append(y)
    return np.stack(ox).astype(np.float32), np.stack(oy).astype(np.float32)


def _loader(X, Y, bs, sh):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
                      batch_size=bs, shuffle=sh)


def _pred(m, X, bs, dev):
    m.eval(); P = []
    with torch.no_grad():
        for xb, in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=bs):
            P.append(torch.sigmoid(m(xb.to(dev))).cpu().numpy())
    return np.concatenate(P)


def run(split, arms, seeds, in_ch, K, n_train=3000, n_syn_normal=2000,
        neg_target=0.03, epochs=20, bs=64, lr=1e-3, device="cpu", tag=""):
    spX, spY = split["single"]
    mX, mY = split["multi"]
    nX, nY = split["normal"]
    rngg = np.random.default_rng(999)
    te = rngg.permutation(len(mX))
    teX, teY = mX[te[:len(te) // 2]], mY[te[:len(te) // 2]]
    orX, orY = mX[te[len(te) // 2:]], mY[te[len(te) // 2:]]   # oracle real-multi pool
    rows = []
    for arm in arms:
        for seed in seeds:
            r2 = np.random.default_rng(seed)
            if arm == "oracle":
                pk = r2.integers(0, len(orX), min(n_train, len(orX)))
                bX, bY = orX[pk], orY[pk]
            elif arm == "single_only":
                bX = np.empty((0, in_ch) + spX.shape[2:], np.float32)
                bY = np.empty((0, K), np.float32)
            else:
                bX, bY = synth(arm, spX, spY, n_train, seed)
            aug = r2.choice(len(spX), min(2000, len(spX)), replace=False)
            trX, trY = np.concatenate([bX, spX[aug]]), np.concatenate([bY, spY[aug]])
            if arm != "oracle" and n_syn_normal > 0:
                npk = r2.choice(len(spX), min(n_syn_normal, len(spX)), replace=False)
                snX = np.minimum(spX[npk], np.median(spX))     # erase-toward-baseline normals
                trX = np.concatenate([trX, snX])
                trY = np.concatenate([trY, np.zeros((len(snX), K), np.float32)])
            if neg_target > 0 and arm != "oracle":
                trY = trY + (1 - trY) * neg_target
            torch.manual_seed(seed)
            m = SmallCNN(num_classes=K, in_ch=in_ch).to(device)
            opt = torch.optim.Adam(m.parameters(), lr=lr); lf = nn.BCEWithLogitsLoss()
            for _ in range(epochs):
                m.train()
                for xb, yb in _loader(trX, trY, bs, True):
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
            teP, nP = _pred(m, teX, bs, device), _pred(m, nX, bs, device)
            pos, neg = pos_neg_prob(teP, teY)
            nfar = float((nP >= 0.5).any(1).mean())
            print(f"[{tag}] {arm:12s} s{seed} | EVAL bitF1={bit_f1(teP, teY):.4f} "
                  f"FAR={far(teP, teY):.4f} exact={exact_match(teP, teY):.4f} "
                  f"pos={pos:.4f} neg={neg:.4f} | NORMAL_FAR={nfar:.4f}", flush=True)
            rows.append((arm, seed, bit_f1(teP, teY), nfar))
    return rows
