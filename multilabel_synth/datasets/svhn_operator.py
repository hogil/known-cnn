"""Content-blind combo-synthesis arms for SVHN, from a REAL single-digit pool.

The single-digit pool is a set of real SVHN digit crops (svhn_format1.
load_single_digit_pool). Every arm builds two-label training canvases from that
SAME shared pool; the ONLY thing that differs between arms is the combination
operator applied to two single crops -- exactly the operator-match design.

House numbers read left-to-right, so the canvas is a horizontal 1xG grid of
DIGIT_SIZE x DIGIT_SIZE cells (default G=2 -> 32 tall x 64 wide). The REAL
multi-digit eval (svhn_format1.load_multidigit) crops the union bbox of a
2-digit house number and resizes it to the same 32x64 canvas, so a clean
partition tiling is the layout matched to the real house-number law.

  partition  : digit A in one cell, digit B in the other (disjoint) -> matched
  overlay    : max-union both digits into ONE cell (superposition)  -> mismatched
  cutmix     : random-rectangle paste of B's single canvas onto A's -> content-blind
  mixup      : 0.5/0.5 pixel blend of A's and B's single canvases   -> content-blind
  single_only: no combos (floor)
"""
import numpy as np


def _index_pool(labels, n_classes=10):
    return {c: np.where(labels == c)[0] for c in range(n_classes)}


def canvas_shape(digit_size=32, grid=2):
    return digit_size, digit_size * grid          # (H, W)


def _cell_origin(cell, digit_size):
    return 0, cell * digit_size                    # (y0, x0), single row


def build_singles(pool_imgs, pool_labels, per_class, seed,
                  digit_size=32, grid=2, n_classes=10):
    """Single-label pool: one real digit crop placed in one random cell."""
    H, W = canvas_shape(digit_size, grid)
    rng = np.random.default_rng(seed)
    by = _index_pool(pool_labels, n_classes)
    X, Y = [], []
    for c in range(n_classes):
        pool = by[c]
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        for i in pick:
            img = np.zeros((H, W), dtype=np.float32)
            k = int(rng.integers(0, grid))
            y0, x0 = _cell_origin(k, digit_size)
            img[y0:y0 + digit_size, x0:x0 + digit_size] = \
                pool_imgs[i].astype(np.float32)
            t = np.zeros(n_classes, dtype=np.float32); t[c] = 1.0
            X.append(img); Y.append(t)
    X = np.stack(X)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(Y)


def build_multi(pool_imgs, pool_labels, n, seed, allowed_pairs, law,
                digit_size=32, grid=2, n_classes=10):
    """Two-label canvases under a combination law.

    law='partition'     : two digits in two distinct cells (disjoint).
    law='superposition' : two digits max-unioned in ONE random cell.
    """
    H, W = canvas_shape(digit_size, grid)
    rng = np.random.default_rng(seed)
    by = _index_pool(pool_labels, n_classes)
    pairs = list(allowed_pairs)
    X, Y = [], []
    for _ in range(n):
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        da = pool_imgs[int(rng.choice(by[a]))].astype(np.float32)
        db = pool_imgs[int(rng.choice(by[b]))].astype(np.float32)
        img = np.zeros((H, W), dtype=np.float32)
        if law == "partition":
            ka, kb = rng.choice(grid, size=2, replace=False)
            y0, x0 = _cell_origin(int(ka), digit_size)
            img[y0:y0 + digit_size, x0:x0 + digit_size] = da
            y0, x0 = _cell_origin(int(kb), digit_size)
            img[y0:y0 + digit_size, x0:x0 + digit_size] = db
        elif law == "superposition":
            k = int(rng.integers(0, grid))
            y0, x0 = _cell_origin(k, digit_size)
            img[y0:y0 + digit_size, x0:x0 + digit_size] = np.maximum(da, db)
        else:
            raise ValueError("law must be 'partition' or 'superposition'")
        t = np.zeros(n_classes, dtype=np.float32); t[a] = 1.0; t[b] = 1.0
        X.append(img); Y.append(t)
    X = np.stack(X)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(Y)


def build_multi_baseline(pool_imgs, pool_labels, n, seed, allowed_pairs, mode,
                         digit_size=32, grid=2, n_classes=10):
    """Content-blind combo baselines (no partition/overlay law knowledge).

    Each source digit is first rendered on its own single-style canvas (one
    random cell). Then the two single canvases are combined:
      mode='cutmix': paste a random rectangular patch of B's canvas onto A's.
      mode='mixup' : 0.5/0.5 pixel average.
    Multi-hot union label {a, b}.
    """
    H, W = canvas_shape(digit_size, grid)
    rng = np.random.default_rng(seed)
    by = _index_pool(pool_labels, n_classes)
    pairs = list(allowed_pairs)
    X, Y = [], []
    for _ in range(n):
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        da = pool_imgs[int(rng.choice(by[a]))].astype(np.float32)
        db = pool_imgs[int(rng.choice(by[b]))].astype(np.float32)
        ca = np.zeros((H, W), dtype=np.float32)
        cb = np.zeros((H, W), dtype=np.float32)
        y0, x0 = _cell_origin(int(rng.integers(0, grid)), digit_size)
        ca[y0:y0 + digit_size, x0:x0 + digit_size] = da
        y0, x0 = _cell_origin(int(rng.integers(0, grid)), digit_size)
        cb[y0:y0 + digit_size, x0:x0 + digit_size] = db
        if mode == "cutmix":
            img = ca.copy()
            lam = float(rng.random())
            cut_h = int(round(H * np.sqrt(1.0 - lam)))
            cut_w = int(round(W * np.sqrt(1.0 - lam)))
            if cut_h > 0 and cut_w > 0:
                cy = int(rng.integers(0, H)); cx = int(rng.integers(0, W))
                y1 = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)
                x1 = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
                img[y1:y2, x1:x2] = cb[y1:y2, x1:x2]
        elif mode == "mixup":
            img = 0.5 * ca + 0.5 * cb
        else:
            raise ValueError("mode must be 'cutmix' or 'mixup'")
        t = np.zeros(n_classes, dtype=np.float32); t[a] = 1.0; t[b] = 1.0
        X.append(img); Y.append(t)
    X = np.stack(X)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(Y)


def build_normal(n, digit_size=32, grid=2, n_classes=10):
    """All-negative (blank canvas) pool for a false-alarm-rate probe."""
    H, W = canvas_shape(digit_size, grid)
    X = np.zeros((n, 1, H, W), dtype=np.float32)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    return X, Y
