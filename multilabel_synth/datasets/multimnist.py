import numpy as np
from itertools import combinations


def all_pairs(n_classes=10):
    return list(combinations(range(n_classes), 2))


def split_holdout(pairs, n_holdout, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    hold = {pairs[i] for i in idx[:n_holdout]}
    train = [p for p in pairs if p not in hold]
    return train, sorted(hold)


def load_mnist(root="E:/data/torchvision", train=True):
    from torchvision.datasets import MNIST
    ds = MNIST(root, train=train, download=True)
    imgs = ds.data.numpy().astype(np.uint8)     # [N,28,28]
    labels = ds.targets.numpy().astype(int)     # [N]
    return imgs, labels


def _index_by_class(labels, n_classes=10):
    return {c: np.where(labels == c)[0] for c in range(n_classes)}


def _place(digit28, canvas, rng):
    c = np.zeros((canvas, canvas), dtype=np.uint8)
    off = canvas - 28
    y = int(rng.integers(0, off + 1))
    x = int(rng.integers(0, off + 1))
    c[y:y + 28, x:x + 28] = digit28
    return c


def build_single_pool(imgs, labels, per_class, seed, canvas=40, n_classes=10):
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    out_imgs, out_tgt = [], []
    for c in range(n_classes):
        pool = by[c]
        pick = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        for i in pick:
            out_imgs.append(_place(imgs[i], canvas, rng))
            t = np.zeros(n_classes, dtype=np.float32); t[c] = 1.0
            out_tgt.append(t)
    X = np.stack(out_imgs)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(out_tgt)


def synthesize_multi(imgs, labels, n, seed, allowed_pairs, canvas=40, n_classes=10):
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    pairs = list(allowed_pairs)
    out_imgs, out_tgt = [], []
    for _ in range(n):
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        ca = _place(imgs[int(rng.choice(by[a]))], canvas, rng)
        cb = _place(imgs[int(rng.choice(by[b]))], canvas, rng)
        merged = np.maximum(ca, cb)          # overlay; may overlap
        out_imgs.append(merged)
        t = np.zeros(n_classes, dtype=np.float32); t[a] = 1.0; t[b] = 1.0
        out_tgt.append(t)
    X = np.stack(out_imgs)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(out_tgt)
