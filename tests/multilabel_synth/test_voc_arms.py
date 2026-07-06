import numpy as np
from multilabel_synth.synthesis.voc_arms import synth_arm


def _fake(n=40):
    X = np.random.default_rng(0).random((n, 3, 32, 32)).astype(np.float32)
    Y = np.zeros((n, 20), np.float32)
    for i in range(n):
        Y[i, i % 20] = 1.0
    return X, Y


def test_cutmix_two_labels():
    X, Y = _fake()
    sX, sY = synth_arm("cutmix", X, Y, 16, seed=0)
    assert sX.shape == (16, 3, 32, 32)
    assert (sY.sum(1) == 2).all()


def test_single_one_label():
    X, Y = _fake()
    sX, sY = synth_arm("single_only", X, Y, 16, seed=0)
    assert (sY.sum(1) == 1).all()


def test_mixup_soft_labels():
    X, Y = _fake()
    sX, sY = synth_arm("mixup", X, Y, 16, seed=0)
    assert (np.abs(sY.sum(1) - 1.0) < 1e-5).all()
    assert ((sY > 0).sum(1) == 2).all()


def test_unknown_arm_raises():
    X, Y = _fake()
    try:
        synth_arm("bogus", X, Y, 4, seed=0)
        assert False
    except ValueError:
        pass


def test_copypaste_two_labels():
    from multilabel_synth.synthesis.voc_arms import synth_copypaste
    bgX, bgY = _fake()
    crX, crY = _fake()
    sX, sY = synth_copypaste(bgX, bgY, crX, crY, 16, seed=0)
    assert sX.shape == (16, 3, 32, 32)
    assert (sY.sum(1) == 2).all()
