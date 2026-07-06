import os
import numpy as np
import pytest
from multilabel_synth.train import train_one


def _mnist_available():
    return os.path.exists("E:/data/torchvision/MNIST/raw/train-images-idx3-ubyte")


def test_train_one_returns_finite_metrics():
    # 3 classes, tiny separable data: class k has a bright band in row-block k
    def make(n):
        X = np.zeros((n, 1, 40, 40), dtype=np.float32)
        Y = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            k = i % 3
            X[i, 0, k * 5:k * 5 + 4, :] = 1.0
            Y[i, k] = 1.0
        return X, Y
    trX, trY = make(60)
    teX, teY = make(30)
    out = train_one(trX, trY, teX, teY, epochs=2, bs=16, device="cpu", seed=0)
    for key in ("mAP", "exact_match", "pos_prob", "neg_prob"):
        assert key in out
        assert np.isfinite(out[key])
    assert 0.0 <= out["mAP"] <= 1.0


@pytest.mark.skipif(not _mnist_available(), reason="MNIST not cached at E:/data/torchvision")
def test_learns_single_digit_mnist():
    # Regression guard for the prior-collapse bug: the old GAP(1) SmallCNN got
    # single-digit mAP ~0.28 @ 15 epochs; the fixed model reaches ~0.95.
    from multilabel_synth.datasets.multimnist import load_mnist, build_single_pool
    tr_i, tr_l = load_mnist(train=True)
    te_i, te_l = load_mnist(train=False)
    trX, trY = build_single_pool(tr_i, tr_l, per_class=200, seed=0)
    teX, teY = build_single_pool(te_i, te_l, per_class=80, seed=1)
    out = train_one(trX, trY, teX, teY, epochs=15, bs=64, device="cpu", seed=0)
    assert out["mAP"] > 0.85
    assert out["pos_prob"] - out["neg_prob"] > 0.3
