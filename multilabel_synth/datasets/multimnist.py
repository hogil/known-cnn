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


def load_torchvision(name, root="E:/data/torchvision", train=True, split=None):
    """Generalized 28x28-grayscale torchvision loader.

    Returns (imgs [N,28,28] uint8, labels [N] int in [0,n_classes), n_classes).
    The operator-match builders are dataset-agnostic: they only need
    (image, class-label) pairs plus the class count, so any of these public
    single-object 28x28 grayscale sources is a drop-in source domain.

    Supported names: mnist, fashionmnist, kmnist, emnist (split= letters /
    balanced / digits / ...). EMNIST is stored transposed relative to display
    orientation and 'letters' labels are 1..26, both corrected here.
    """
    import torchvision.datasets as D
    key = name.lower()
    if key == "mnist":
        ds = D.MNIST(root, train=train, download=True)
        return ds.data.numpy().astype(np.uint8), ds.targets.numpy().astype(int), 10
    if key == "fashionmnist":
        ds = D.FashionMNIST(root, train=train, download=True)
        return ds.data.numpy().astype(np.uint8), ds.targets.numpy().astype(int), 10
    if key == "kmnist":
        ds = D.KMNIST(root, train=train, download=True)
        return ds.data.numpy().astype(np.uint8), ds.targets.numpy().astype(int), 10
    if key == "emnist":
        sp = split or "letters"
        ds = D.EMNIST(root, split=sp, train=train, download=True)
        imgs = np.transpose(ds.data.numpy().astype(np.uint8), (0, 2, 1))
        labels = ds.targets.numpy().astype(int)
        if sp == "letters":
            labels = labels - 1          # 1..26 -> 0..25
            n_classes = 26
        elif sp == "balanced":
            n_classes = 47
        elif sp in ("digits", "mnist"):
            n_classes = 10
        else:
            n_classes = int(labels.max()) + 1
        return imgs, labels, n_classes
    raise ValueError("unknown torchvision dataset: %r" % name)


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
