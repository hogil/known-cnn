"""Color combination operators for the DLRSD land-cover operator-match study.

Same shared single-class crop pool feeds every arm; the ONLY thing that differs
is the combination operator applied to the single-class crops -- the operator-match
design, here on real aerial land-cover instead of SVHN digits.

Canvas = a GRID x GRID tiling of cell x cell quadrants (default 2x2 -> 128x128),
because land cover occupies DISJOINT 2D spatial regions.

  partition   : k distinct-class crops in k DISJOINT quadrants  -> MATCHED law
  overlay     : k distinct-class crops per-pixel-MAX into ONE quadrant (superpose)
  cutmix      : random-rectangle paste of crop B's canvas onto crop A's (blind)
  mixup       : 0.5/0.5 pixel blend of A's and B's single canvases (blind)
  single_only : one crop in one random quadrant (floor)

Outputs are [N, 3, H, W] float32 in [0,1]; the runner applies ImageNet norm.
"""
import numpy as np


def _index_pool(labels, n_classes):
    return {c: np.where(labels == c)[0] for c in range(n_classes)}


def canvas_shape(cell, grid):
    return cell * grid, cell * grid                      # (H, W)


def _quad_origin(q, cell, grid):
    return (q // grid) * cell, (q % grid) * cell         # (y0, x0)


def _place(canvas, crop, q, cell, grid):
    y0, x0 = _quad_origin(q, cell, grid)
    canvas[y0:y0 + cell, x0:x0 + cell, :] = crop


def build_singles(pool_imgs, pool_labels, per_class, seed, cell, grid,
                  n_classes):
    H, W = canvas_shape(cell, grid)
    rng = np.random.default_rng(seed)
    by = _index_pool(pool_labels, n_classes)
    nq = grid * grid
    X, Y = [], []
    for c in range(n_classes):
        pool = by[c]
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        for i in pick:
            img = np.zeros((H, W, 3), dtype=np.float32)
            _place(img, pool_imgs[i].astype(np.float32), int(rng.integers(0, nq)),
                   cell, grid)
            t = np.zeros(n_classes, dtype=np.float32); t[c] = 1.0
            X.append(img); Y.append(t)
    X = np.stack(X).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return X, np.stack(Y)


def _pick_k_distinct(rng, n_classes, k_choices):
    k = int(rng.choice(k_choices))
    k = min(k, n_classes)
    return rng.choice(n_classes, size=k, replace=False)


def build_multi(pool_imgs, pool_labels, n, seed, law, cell, grid, n_classes,
                k_choices=(2,)):
    """law='partition'     : k crops in k distinct quadrants (disjoint).
       law='superposition' : k crops per-pixel-max into ONE random quadrant."""
    H, W = canvas_shape(cell, grid)
    rng = np.random.default_rng(seed)
    by = _index_pool(pool_labels, n_classes)
    nq = grid * grid
    X, Y = [], []
    for _ in range(n):
        classes = _pick_k_distinct(rng, n_classes, k_choices)
        crops = [pool_imgs[int(rng.choice(by[c]))].astype(np.float32) for c in classes]
        img = np.zeros((H, W, 3), dtype=np.float32)
        if law == "partition":
            quads = rng.choice(nq, size=len(classes), replace=False)
            for crop, q in zip(crops, quads):
                _place(img, crop, int(q), cell, grid)
        elif law == "superposition":
            q = int(rng.integers(0, nq))
            y0, x0 = _quad_origin(q, cell, grid)
            acc = crops[0]
            for cr in crops[1:]:
                acc = np.maximum(acc, cr)
            img[y0:y0 + cell, x0:x0 + cell, :] = acc
        else:
            raise ValueError("law must be 'partition' or 'superposition'")
        t = np.zeros(n_classes, dtype=np.float32)
        for c in classes:
            t[c] = 1.0
        X.append(img); Y.append(t)
    X = np.stack(X).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return X, np.stack(Y)


def build_multi_baseline(pool_imgs, pool_labels, n, seed, mode, cell, grid,
                         n_classes):
    """Content-blind pair baselines (no partition/overlay law knowledge)."""
    H, W = canvas_shape(cell, grid)
    rng = np.random.default_rng(seed)
    by = _index_pool(pool_labels, n_classes)
    nq = grid * grid
    X, Y = [], []
    for _ in range(n):
        a, b = rng.choice(n_classes, size=2, replace=False)
        da = pool_imgs[int(rng.choice(by[a]))].astype(np.float32)
        db = pool_imgs[int(rng.choice(by[b]))].astype(np.float32)
        ca = np.zeros((H, W, 3), dtype=np.float32)
        cb = np.zeros((H, W, 3), dtype=np.float32)
        _place(ca, da, int(rng.integers(0, nq)), cell, grid)
        _place(cb, db, int(rng.integers(0, nq)), cell, grid)
        if mode == "cutmix":
            img = ca.copy()
            lam = float(rng.random())
            cut_h = int(round(H * np.sqrt(1.0 - lam)))
            cut_w = int(round(W * np.sqrt(1.0 - lam)))
            if cut_h > 0 and cut_w > 0:
                cy = int(rng.integers(0, H)); cx = int(rng.integers(0, W))
                y1 = max(0, cy - cut_h // 2); y2 = min(H, cy + cut_h // 2)
                x1 = max(0, cx - cut_w // 2); x2 = min(W, cx + cut_w // 2)
                img[y1:y2, x1:x2, :] = cb[y1:y2, x1:x2, :]
        elif mode == "mixup":
            img = 0.5 * ca + 0.5 * cb
        else:
            raise ValueError("mode must be 'cutmix' or 'mixup'")
        t = np.zeros(n_classes, dtype=np.float32); t[a] = 1.0; t[b] = 1.0
        X.append(img); Y.append(t)
    X = np.stack(X).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return X, np.stack(Y)


def real_to_chw(imgs_uint8):
    """[N,H,W,3] uint8 -> [N,3,H,W] float32 in [0,1]."""
    return imgs_uint8.transpose(0, 3, 1, 2).astype(np.float32) / 255.0


def build_normal(n, cell, grid, n_classes):
    """All-negative (blank) canvases for a false-alarm-rate probe."""
    H, W = canvas_shape(cell, grid)
    X = np.zeros((n, 3, H, W), dtype=np.float32)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    return X, Y
