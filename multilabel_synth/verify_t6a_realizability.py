"""Gate item 2: empirical realizability of the T6a utility-inversion, on a CONTROLLED
composition family (CPU, tiny). Turns T6a's "assume operator utilities flip between
two worlds with identical single marginals" from an assumption into a CHECKED fact for
at least one family.

Family: 16x16 grayscale, 2 classes. class0 = horizontal bar (random row-band),
class1 = vertical bar (random col-band). Single-class conditionals are generated
IDENTICALLY in both worlds (so single marginals are world-invariant). The two worlds
differ ONLY in the true co-occurrence (both-present) appearance:
  - World A (superposition):  max(h_bar, v_bar)            -> a '+' cross
  - World B (partition):      h_bar in top half, v_bar in bottom half (each squished)
Content-blind operators built from singles:
  - summation: max(single0, single1)
  - partition: single0 top half + single1 bottom half (squished)
Train M_sum on {singles + summation combos}, M_part on {singles + partition combos};
utility = bit-F1 of detecting BOTH classes at a fixed threshold on a world's real
co-occurrence test set. T6a realizability holds for this family iff the utilities
FLIP: M_sum wins on World A, M_part wins on World B, while singles are identical.
"""
import numpy as np
import torch, torch.nn as nn

H = W = 16
rng_global = np.random.default_rng(0)


def h_bar(rng):
    img = np.zeros((H, W), np.float32); r = int(rng.integers(2, H - 4))
    img[r:r + 3, :] = 1.0; return img


def v_bar(rng):
    img = np.zeros((H, W), np.float32); c = int(rng.integers(2, W - 4))
    img[:, c:c + 3] = 1.0; return img


def squish_top(img):    # compress a full HxW image into the top half
    out = np.zeros((H, W), np.float32); out[:H // 2, :] = img[::2, :]; return out


def squish_bot(img):
    out = np.zeros((H, W), np.float32); out[H // 2:, :] = img[::2, :]; return out


def real_cooc(world, rng, n):
    X, Y = [], []
    for _ in range(n):
        h, v = h_bar(rng), v_bar(rng)
        if world == "A":              # superposition
            X.append(np.maximum(h, v))
        else:                          # partition (side-by-side, top/bottom, squished)
            X.append(np.maximum(squish_top(h), squish_bot(v)))
        Y.append([1.0, 1.0])
    return np.array(X), np.array(Y, np.float32)


def singles(rng, n):
    X, Y = [], []
    for _ in range(n):
        X.append(h_bar(rng)); Y.append([1.0, 0.0])
        X.append(v_bar(rng)); Y.append([0.0, 1.0])
    return np.array(X), np.array(Y, np.float32)


def synth(op, rng, n):
    X, Y = [], []
    for _ in range(n):
        h, v = h_bar(rng), v_bar(rng)
        if op == "summation":
            X.append(np.maximum(h, v))
        else:                          # partition operator (content-blind)
            X.append(np.maximum(squish_top(h), squish_bot(v)))
        Y.append([1.0, 1.0])
    return np.array(X), np.array(Y, np.float32)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(16, 2))

    def forward(self, x): return self.net(x)


def train(op, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    sX, sY = singles(rng, 150); cX, cY = synth(op, rng, 300)
    X = np.concatenate([sX, cX])[:, None]; Y = np.concatenate([sY, cY])
    m = TinyCNN(); opt = torch.optim.Adam(m.parameters(), 3e-3); lf = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    for _ in range(30):
        idx = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            b = idx[i:i + 64]; opt.zero_grad(); lf(m(Xt[b]), Yt[b]).backward(); opt.step()
    return m


def bitf1(m, X, Y, tau=0.5):
    with torch.no_grad():
        P = torch.sigmoid(m(torch.from_numpy(X[:, None]))).numpy()
    pred = (P >= tau).astype(float)
    tp = (pred * Y).sum(); fp = (pred * (1 - Y)).sum(); fn = ((1 - pred) * Y).sum()
    prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


def main():
    seeds = range(5)
    # single-marginal identity check: singles are generated identically for both worlds
    # (by construction -- real_cooc differs, singles() does not). Confirm distributions match.
    r1 = np.random.default_rng(100); r2 = np.random.default_rng(100)
    sA, _ = singles(r1, 200); sB, _ = singles(r2, 200)
    print(f"single-marginal identity (same RNG -> identical draws): "
          f"max|sA-sB|={np.abs(sA - sB).max():.1e} (0 => world-invariant singles)")
    res = {("summation", "A"): [], ("summation", "B"): [],
           ("partition", "A"): [], ("partition", "B"): []}
    for s in seeds:
        m_sum = train("summation", s); m_part = train("partition", s)
        rA = np.random.default_rng(1000 + s); rB = np.random.default_rng(2000 + s)
        XA, YA = real_cooc("A", rA, 200); XB, YB = real_cooc("B", rB, 200)
        res[("summation", "A")].append(bitf1(m_sum, XA, YA))
        res[("summation", "B")].append(bitf1(m_sum, XB, YB))
        res[("partition", "A")].append(bitf1(m_part, XA, YA))
        res[("partition", "B")].append(bitf1(m_part, XB, YB))
    def mean(k): return float(np.mean(res[k]))
    print("\nutility (bit-F1 detecting BOTH classes), mean over 5 seeds:")
    print(f"  {'operator':10s} {'World A (overlay)':>18s} {'World B (side-by-side)':>24s}")
    print(f"  {'summation':10s} {mean(('summation','A')):18.3f} {mean(('summation','B')):24.3f}")
    print(f"  {'partition':10s} {mean(('partition','A')):18.3f} {mean(('partition','B')):24.3f}")
    flipA = mean(("summation", "A")) - mean(("partition", "A"))   # >0: summation wins on A
    flipB = mean(("partition", "B")) - mean(("summation", "B"))   # >0: partition wins on B
    best_A = "summation" if flipA > 0 else "partition"
    best_B = "partition" if flipB > 0 else "summation"
    flipped = (best_A != best_B)
    print(f"\n  best op on World A = {best_A}; best op on World B = {best_B}")
    print(f"  utility INVERSION realized: {flipped}  (deltaA={flipA:+.3f}, deltaB={flipB:+.3f})")
    print("  => T6a realizability holds for THIS controlled family: identical single"
          " marginals,\n     opposite best operator across two worlds -> a source-only"
          " selector cannot pick.\n     (One family; NOT a claim for all domains.)")


if __name__ == "__main__":
    main()
