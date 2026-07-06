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
                   canvas=40, n_classes=10, grid=4, fcm_mode="checker",
                   cutmix_frac=0.5, mixup_alpha=1.0):
    if arm in ("oracle", "overlay"):
        # max-overlay of two whole single digits: blind (no location knowledge),
        # both objects preserved whole, full-cover -> label-honest. This is the
        # MNIST analog of the chip min-blend (defect wins over normal).
        # 'oracle' vs 'overlay' differ only in allowed_pairs (caller restricts
        # oracle to train pairs; overlay may use all pairs).
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
            lam = float(rng.beta(mixup_alpha, mixup_alpha))
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
            side = max(1, min(canvas, int(round(canvas * (cutmix_frac ** 0.5)))))
            y = int(rng.integers(0, canvas - side + 1))
            x = int(rng.integers(0, canvas - side + 1))
            img[y:y + side, x:x + side] = cb[y:y + side, x:x + side]
            t[a] = 1.0; t[b] = 1.0
        elif arm == "fcm_pm":
            if fcm_mode == "fill":
                # filling-complement: keep a whole, fill b into a's empty background
                img = ca.copy()
                bg = ca == 0
                img[bg] = cb[bg]
            elif fcm_mode == "strip":
                img = np.zeros((canvas, canvas), dtype=np.float32)
                cell = canvas // grid
                for r in range(grid):
                    src = ca if r % 2 == 0 else cb
                    img[r * cell:(r + 1) * cell, :] = src[r * cell:(r + 1) * cell, :]
            else:  # "checker"
                mask = _grid_mask(canvas, grid)
                img = np.where(mask, ca, cb)
            t[a] = 1.0; t[b] = 1.0
        else:
            raise ValueError(f"unknown arm: {arm}")

        out_imgs.append(img); out_tgt.append(t)

    X = np.stack(out_imgs)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(out_tgt)
