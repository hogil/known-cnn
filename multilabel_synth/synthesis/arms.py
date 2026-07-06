import numpy as np
from ..datasets.multimnist import _index_by_class, _place, synthesize_multi


def _grid_mask(canvas, grid):
    m = np.zeros((canvas, canvas), dtype=bool)
    cell = canvas // grid
    for i in range(grid):
        for j in range(grid):
            if (i + j) % 2 == 0:
                m[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = True
    return m


def synthesize_arm(arm, imgs, labels, n, seed, allowed_pairs,
                   canvas=40, n_classes=10, grid=4):
    if arm == "oracle":
        return synthesize_multi(imgs, labels, n, seed, allowed_pairs, canvas, n_classes)

    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    pairs = list(allowed_pairs)
    out_imgs, out_tgt = [], []

    for _ in range(n):
        t = np.zeros(n_classes, dtype=np.float32)

        if arm == "single_only":
            c = int(rng.integers(0, n_classes))
            img = _place(imgs[int(rng.choice(by[c]))], canvas, rng).astype(np.float32)
            t[c] = 1.0
            out_imgs.append(img); out_tgt.append(t)
            continue

        a, b = pairs[int(rng.integers(0, len(pairs)))]
        ca = _place(imgs[int(rng.choice(by[a]))], canvas, rng).astype(np.float32)
        cb = _place(imgs[int(rng.choice(by[b]))], canvas, rng).astype(np.float32)

        if arm == "mixup":
            lam = float(rng.beta(1.0, 1.0))
            img = lam * ca + (1.0 - lam) * cb
            t[a] = lam; t[b] = 1.0 - lam
        elif arm == "copy_paste":
            img = np.zeros((canvas, canvas), dtype=np.float32)
            half = canvas // 2
            img[:, :half] = ca[:, :half]
            img[:, half:] = cb[:, half:]
            t[a] = 1.0; t[b] = 1.0
        elif arm == "cutmix":
            img = ca.copy()
            ch = cw = canvas // 2
            y = int(rng.integers(0, canvas - ch))
            x = int(rng.integers(0, canvas - cw))
            img[y:y + ch, x:x + cw] = cb[y:y + ch, x:x + cw]
            t[a] = 1.0; t[b] = 1.0
        elif arm == "fcm_pm":
            mask = _grid_mask(canvas, grid)
            img = np.where(mask, ca, cb)
            t[a] = 1.0; t[b] = 1.0
        else:
            raise ValueError(f"unknown arm: {arm}")

        out_imgs.append(img); out_tgt.append(t)

    X = np.stack(out_imgs)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(out_tgt)
