import numpy as np
from multilabel_synth.train import train_one


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
