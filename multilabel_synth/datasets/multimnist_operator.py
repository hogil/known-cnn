"""Operator-match control on public MNIST.

Two multi-label domains that differ in ONE variable: the true combination law
by which two source digits form a two-label example.

  - SUPERPOSITION law: the two digits share a single grid cell and are combined
    by pixelwise max (ink-superposition / overlapping strokes).
  - PARTITION law: the two digits occupy two DISTINCT grid cells (disjoint
    regions, no overlap).

Both laws use the identical canvas geometry (a grid of digit-sized cells), so
the only thing that changes between the two domains is same-cell (overlap) vs
distinct-cell (disjoint) placement.

The MNIST analogue of the chip synthesis operators:
  - OVERLAY operator  == max-union of two digits in one cell  (superposition).
  - PARTITION operator == place each digit into a distinct cell (FCM-PM style).

Training digits are drawn from the MNIST TRAIN split; evaluation multi-label
examples are drawn from the MNIST TEST split, so no digit instance leaks from
train into eval.
"""
import numpy as np

from .multimnist import _index_by_class


def canvas_size(grid=2, cell=28):
    return grid * cell


def _cell_origin(cell_idx, grid, cell):
    gi, gj = cell_idx // grid, cell_idx % grid
    return gi * cell, gj * cell


def build_singles(imgs, labels, per_class, seed, grid=2, cell=28, n_classes=10):
    """Single-label pool: one digit placed in one random cell. 1-hot target."""
    canvas = canvas_size(grid, cell)
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    n_cell = grid * grid
    X, Y = [], []
    for c in range(n_classes):
        pool = by[c]
        pick = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        for i in pick:
            img = np.zeros((canvas, canvas), dtype=np.float32)
            k = int(rng.integers(0, n_cell))
            y0, x0 = _cell_origin(k, grid, cell)
            img[y0:y0 + cell, x0:x0 + cell] = imgs[i].astype(np.float32)
            t = np.zeros(n_classes, dtype=np.float32)
            t[c] = 1.0
            X.append(img)
            Y.append(t)
    X = np.stack(X)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(Y)


def build_multi(imgs, labels, n, seed, allowed_pairs, law,
                grid=2, cell=28, n_classes=10):
    """Two-label examples under a combination law.

    law='superposition': two digits in ONE random cell, pixelwise max.
    law='partition'    : two digits in TWO distinct random cells.
    """
    canvas = canvas_size(grid, cell)
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    n_cell = grid * grid
    pairs = list(allowed_pairs)
    X, Y = [], []
    for _ in range(n):
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        da = imgs[int(rng.choice(by[a]))].astype(np.float32)
        db = imgs[int(rng.choice(by[b]))].astype(np.float32)
        img = np.zeros((canvas, canvas), dtype=np.float32)
        if law == "superposition":
            k = int(rng.integers(0, n_cell))
            y0, x0 = _cell_origin(k, grid, cell)
            img[y0:y0 + cell, x0:x0 + cell] = np.maximum(da, db)
        elif law == "partition":
            ka, kb = rng.choice(n_cell, size=2, replace=False)
            y0, x0 = _cell_origin(int(ka), grid, cell)
            img[y0:y0 + cell, x0:x0 + cell] = da
            y0, x0 = _cell_origin(int(kb), grid, cell)
            img[y0:y0 + cell, x0:x0 + cell] = db
        else:
            raise ValueError("law must be 'superposition' or 'partition'")
        t = np.zeros(n_classes, dtype=np.float32)
        t[a] = 1.0
        t[b] = 1.0
        X.append(img)
        Y.append(t)
    X = np.stack(X)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(Y)


def build_multi_baseline(imgs, labels, n, seed, allowed_pairs, mode,
                         grid=2, cell=28, n_classes=10):
    """Content-blind combo-synthesis baselines (no true-law knowledge).

    Both baselines start from the SAME single sources as build_singles/build_multi:
    each of the two source digits is first rendered on its own single-style canvas
    (one digit in one random grid cell). They then combine those two single
    canvases WITHOUT any knowledge of the domain's true combination law:

      mode='cutmix': paste a random rectangular patch cropped from the second
        single canvas onto the first (standard CutMix box, size ~sqrt(1-lam)).
        The only spatial "knowledge" is a random rectangle -- no cell/partition
        awareness. Multi-hot union label {a, b}.
      mode='mixup' : alpha-blend the two single canvases by a 0.5/0.5 pixel
        average. Multi-hot union label {a, b}.

    Because neither matches a clean partition (disjoint cells) nor a clean
    superposition (single-cell max-union), both should UNDER-perform the operator
    matched to the domain's true law in either regime.
    """
    canvas = canvas_size(grid, cell)
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    n_cell = grid * grid
    pairs = list(allowed_pairs)
    X, Y = [], []
    for _ in range(n):
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        da = imgs[int(rng.choice(by[a]))].astype(np.float32)
        db = imgs[int(rng.choice(by[b]))].astype(np.float32)
        # render each source digit on its own single-style canvas (one random cell)
        ca = np.zeros((canvas, canvas), dtype=np.float32)
        cb = np.zeros((canvas, canvas), dtype=np.float32)
        ka = int(rng.integers(0, n_cell))
        y0, x0 = _cell_origin(ka, grid, cell)
        ca[y0:y0 + cell, x0:x0 + cell] = da
        kb = int(rng.integers(0, n_cell))
        y0, x0 = _cell_origin(kb, grid, cell)
        cb[y0:y0 + cell, x0:x0 + cell] = db
        if mode == "cutmix":
            img = ca.copy()
            lam = float(rng.random())                 # ~U(0,1) -> Beta(1,1)
            cut = int(round(canvas * np.sqrt(1.0 - lam)))
            if cut > 0:
                cx = int(rng.integers(0, canvas))
                cy = int(rng.integers(0, canvas))
                x1 = max(0, cx - cut // 2); x2 = min(canvas, cx + cut // 2)
                y1 = max(0, cy - cut // 2); y2 = min(canvas, cy + cut // 2)
                img[y1:y2, x1:x2] = cb[y1:y2, x1:x2]
        elif mode == "mixup":
            img = 0.5 * ca + 0.5 * cb                  # 0.5/0.5 pixel average
        else:
            raise ValueError("mode must be 'cutmix' or 'mixup'")
        t = np.zeros(n_classes, dtype=np.float32)
        t[a] = 1.0
        t[b] = 1.0
        X.append(img)
        Y.append(t)
    X = np.stack(X)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(Y)


def build_normal(n, grid=2, cell=28, n_classes=10):
    """All-negative (blank canvas) pool for a false-alarm-rate probe."""
    canvas = canvas_size(grid, cell)
    X = np.zeros((n, 1, canvas, canvas), dtype=np.float32)
    Y = np.zeros((n, n_classes), dtype=np.float32)
    return X, Y
