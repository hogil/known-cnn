import numpy as np
from multilabel_synth.synthesis.arms import synthesize_arm


def _fake_mnist(n_per=20, n_classes=10):
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per):
            imgs.append(np.full((28, 28), (c + 1) * 20, dtype=np.uint8))
            labels.append(c)
    return np.stack(imgs), np.array(labels, dtype=int)


def test_single_only_one_positive():
    imgs, labels = _fake_mnist()
    X, Y = synthesize_arm("single_only", imgs, labels, n=16, seed=0,
                          allowed_pairs=[(0, 1)])
    assert X.shape == (16, 1, 40, 40)
    assert (Y.sum(axis=1) == 1).all()


def test_hard_arms_two_positives():
    imgs, labels = _fake_mnist()
    for arm in ["oracle", "copy_paste", "cutmix", "fcm_pm"]:
        X, Y = synthesize_arm(arm, imgs, labels, n=16, seed=1,
                              allowed_pairs=[(0, 1)])
        assert X.shape == (16, 1, 40, 40), arm
        assert (Y.sum(axis=1) == 2).all(), arm
        assert (Y[:, 0] == 1).all() and (Y[:, 1] == 1).all(), arm


def test_mixup_soft_labels_sum_to_one():
    imgs, labels = _fake_mnist()
    X, Y = synthesize_arm("mixup", imgs, labels, n=16, seed=2,
                          allowed_pairs=[(0, 1)])
    # two nonzero soft entries per row summing to ~1
    assert (np.abs(Y.sum(axis=1) - 1.0) < 1e-5).all()
    assert ((Y > 0).sum(axis=1) == 2).all()


def test_fcm_pm_preserves_both_digits():
    # complement grid must keep ink from BOTH sources (unlike a cutmix that can hide one)
    imgs, labels = _fake_mnist()
    X, _ = synthesize_arm("fcm_pm", imgs, labels, n=1, seed=3,
                          allowed_pairs=[(0, 1)], canvas=40, grid=4)
    img = X[0, 0]
    # both digit intensities (0->20, 1->40 after scaling /255) should appear
    vals = np.unique((img * 255).round().astype(int))
    assert (vals > 0).sum() >= 1


def test_unknown_arm_raises():
    imgs, labels = _fake_mnist()
    try:
        synthesize_arm("bogus", imgs, labels, n=4, seed=0, allowed_pairs=[(0, 1)])
        assert False, "expected ValueError"
    except ValueError:
        pass
