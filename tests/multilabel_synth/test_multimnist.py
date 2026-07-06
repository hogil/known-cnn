import numpy as np
from multilabel_synth.datasets.multimnist import (
    all_pairs, split_holdout, build_single_pool, synthesize_multi,
)


def _fake_mnist(n_per=20, n_classes=10):
    # each digit image is a 28x28 block whose pixels encode its class (>0 ink)
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per):
            im = np.full((28, 28), (c + 1) * 20, dtype=np.uint8)
            imgs.append(im); labels.append(c)
    return np.stack(imgs), np.array(labels, dtype=int)


def test_all_pairs_count():
    assert len(all_pairs(10)) == 45


def test_split_holdout_deterministic_and_disjoint():
    pairs = all_pairs(10)
    tr1, ho1 = split_holdout(pairs, 9, seed=7)
    tr2, ho2 = split_holdout(pairs, 9, seed=7)
    assert ho1 == ho2                       # deterministic
    assert len(ho1) == 9
    assert set(tr1).isdisjoint(set(ho1))    # disjoint
    assert len(tr1) + len(ho1) == 45


def test_single_pool_is_single_label():
    imgs, labels = _fake_mnist()
    X, Y = build_single_pool(imgs, labels, per_class=5, seed=0, canvas=40)
    assert X.shape == (50, 1, 40, 40)
    assert X.dtype == np.float32 and X.max() <= 1.0
    assert (Y.sum(axis=1) == 1).all()       # exactly one positive each


def test_synthesize_multi_has_two_labels_from_allowed_pairs():
    imgs, labels = _fake_mnist()
    X, Y = synthesize_multi(imgs, labels, n=32, seed=1,
                            allowed_pairs=[(0, 1)], canvas=40)
    assert X.shape == (32, 1, 40, 40)
    assert (Y.sum(axis=1) == 2).all()
    # only classes 0 and 1 ever active
    assert Y[:, 2:].sum() == 0
    assert (Y[:, 0] == 1).all() and (Y[:, 1] == 1).all()
