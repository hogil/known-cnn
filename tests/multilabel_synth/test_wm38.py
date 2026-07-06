import numpy as np
from multilabel_synth.synthesis.wm38_arms import synth_wm38
from multilabel_synth.datasets.mixedwm38 import split_holdout_combos


def _fake_singles(n_per=30, n_bits=8):
    X, Y = [], []
    for c in range(n_bits):
        for _ in range(n_per):
            img = np.full((1, 52, 52), 0.5, np.float32)
            img[0, c * 6:c * 6 + 4, :] = 1.0        # class-coded defect band
            X.append(img)
            y = np.zeros(n_bits, np.float32); y[c] = 1.0
            Y.append(y)
    return np.stack(X), np.stack(Y)


def test_hard_arms_two_bits():
    X, Y = _fake_singles()
    for arm in ["overlay", "fcm_pm", "cutmix"]:
        sX, sY = synth_wm38(arm, X, Y, 16, seed=0)
        assert sX.shape == (16, 1, 52, 52), arm
        assert (sY.sum(1) == 2).all(), arm


def test_mixup_soft():
    X, Y = _fake_singles()
    _, sY = synth_wm38("mixup", X, Y, 16, seed=1)
    assert (np.abs(sY.sum(1) - 1.0) < 1e-5).all()


def test_single_only_one_bit():
    X, Y = _fake_singles()
    _, sY = synth_wm38("single_only", X, Y, 16, seed=2)
    assert (sY.sum(1) == 1).all()


def test_holdout_combos_disjoint():
    combos = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    tr, ho = split_holdout_combos(combos, 2, seed=3)
    assert len(ho) == 2 and set(tr).isdisjoint(set(ho))
