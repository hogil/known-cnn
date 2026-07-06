import numpy as np

# Synthesis arms for MixedWM38: combine two real SINGLE wafer maps into a
# synthetic mixed map. Inputs are X [N,1,52,52] in [0,1] (0=bg, 0.5=normal die,
# 1.0=defect die) and Y [N,8] one-hot singles. All arms are content-blind.


def _pair_indices(Y, rng, n):
    lab = Y.argmax(1)
    out = []
    ii = rng.integers(0, len(Y), size=n * 4)
    jj = rng.integers(0, len(Y), size=n * 4)
    for a, b in zip(ii, jj):
        if lab[a] != lab[b]:
            out.append((int(a), int(b)))
            if len(out) >= n:
                break
    return out


def _complement_mix(ca, cb, grid, n_groups, rng):
    """Faithful FCM-PM: B base + A overwrites 1/n_groups random-scattered cells."""
    _, H, W = ca.shape
    ch, cw = H // grid, W // grid
    n_cells = grid * grid
    perm = rng.permutation(n_cells)
    a_cells = perm[: n_cells // n_groups]
    img = cb.copy()
    for ci in a_cells:
        gi, gj = int(ci) // grid, int(ci) % grid
        y0, x0 = gi * ch, gj * cw
        y1 = (gi + 1) * ch if gi < grid - 1 else H
        x1 = (gj + 1) * cw if gj < grid - 1 else W
        img[:, y0:y1, x0:x1] = ca[:, y0:y1, x0:x1]
    return img


def synth_wm38(arm, X, Y, n, seed, grid=9, n_groups=3, cutmix_frac=0.33,
               mixup_alpha=1.0):
    """Return (X_synth, Y_synth) built from single-label wafers only."""
    rng = np.random.default_rng(seed)
    if arm == "single_only":
        idx = rng.integers(0, len(X), size=n)
        return X[idx].copy(), Y[idx].copy()

    out_x, out_y = [], []
    for a, b in _pair_indices(Y, rng, n):
        ca, cb = X[a], X[b]
        if arm == "overlay":
            # max: defect(1.0) wins over normal die(0.5); bg stays bg.
            img = np.maximum(ca, cb)
        elif arm == "fcm_pm":
            img = _complement_mix(ca, cb, grid, n_groups, rng)
        elif arm == "cutmix":
            _, H, W = ca.shape
            side = max(1, min(H, int(round(H * cutmix_frac ** 0.5))))
            y = int(rng.integers(0, H - side + 1))
            x = int(rng.integers(0, W - side + 1))
            img = ca.copy()
            img[:, y:y + side, x:x + side] = cb[:, y:y + side, x:x + side]
        elif arm == "mixup":
            lam = float(rng.beta(mixup_alpha, mixup_alpha))
            img = lam * ca + (1.0 - lam) * cb
            out_x.append(img)
            out_y.append(lam * Y[a] + (1.0 - lam) * Y[b])   # soft
            continue
        else:
            raise ValueError(f"unknown wm38 arm: {arm}")
        out_x.append(img)
        out_y.append(np.maximum(Y[a], Y[b]))                # hard {a,b}
    return np.stack(out_x).astype(np.float32), np.stack(out_y).astype(np.float32)
