import numpy as np

# Natural RGB images are not sparse, so MNIST max-overlay / fill do NOT port
# (max of two photos = bright garbage; there is no empty background to fill).
# The content-blind methods that transfer are cutmix (region paste) and mixup
# (blend). 'single' and 'oracle' are the floor/ceiling references.


def _diff_label_pairs(Y, rng, n):
    """Sample n index pairs (a, b) from the single pool with different labels."""
    labels = Y.argmax(1)
    N = len(Y)
    pairs = []
    ii = rng.integers(0, N, size=n * 4)
    jj = rng.integers(0, N, size=n * 4)
    for a, b in zip(ii, jj):
        if labels[a] != labels[b]:
            pairs.append((int(a), int(b)))
            if len(pairs) >= n:
                break
    return pairs


def synth_single(X, Y, n, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), size=n)
    return X[idx].copy(), Y[idx].copy()


def synth_cutmix(X, Y, n, seed, frac=0.3):
    rng = np.random.default_rng(seed)
    S = X.shape[-1]
    out_x, out_y = [], []
    for a, b in _diff_label_pairs(Y, rng, n):
        img = X[a].copy()
        side = max(1, min(S, int(round(S * frac ** 0.5))))
        y = int(rng.integers(0, S - side + 1))
        x = int(rng.integers(0, S - side + 1))
        img[:, y:y + side, x:x + side] = X[b][:, y:y + side, x:x + side]
        out_x.append(img)
        out_y.append(np.maximum(Y[a], Y[b]))   # hard {a, b}
    return np.stack(out_x), np.stack(out_y)


def synth_mixup(X, Y, n, seed, alpha=1.0):
    rng = np.random.default_rng(seed)
    out_x, out_y = [], []
    for a, b in _diff_label_pairs(Y, rng, n):
        lam = float(rng.beta(alpha, alpha))
        out_x.append(lam * X[a] + (1.0 - lam) * X[b])
        out_y.append(lam * Y[a] + (1.0 - lam) * Y[b])   # soft
    return np.stack(out_x).astype(np.float32), np.stack(out_y).astype(np.float32)


def synth_arm(arm, X, Y, n, seed, frac=0.3, alpha=1.0):
    if arm == "single_only":
        return synth_single(X, Y, n, seed)
    if arm == "cutmix":
        return synth_cutmix(X, Y, n, seed, frac=frac)
    if arm == "mixup":
        return synth_mixup(X, Y, n, seed, alpha=alpha)
    raise ValueError(f"unknown VOC arm: {arm}")
