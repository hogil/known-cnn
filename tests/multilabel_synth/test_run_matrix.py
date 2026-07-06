import os
import numpy as np
from multilabel_synth.run_matrix import run_matrix


def _fake_mnist(n_per=40, n_classes=10):
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per):
            imgs.append(np.full((28, 28), (c + 1) * 20, dtype=np.uint8))
            labels.append(c)
    return np.stack(imgs), np.array(labels, dtype=int)


def test_run_matrix_writes_rows(tmp_path):
    imgs, labels = _fake_mnist()
    csv = tmp_path / "res.csv"
    rows = run_matrix(
        imgs, labels,
        arms=["single_only", "fcm_pm"],
        seeds=[0],
        n_holdout=3,
        per_class_single=10,
        n_train=40, n_test=40, n_holdout_test=20,
        epochs=1, bs=16, device="cpu",
        out_csv=str(csv),
    )
    assert os.path.exists(csv)
    assert len(rows) == 2                      # 2 arms x 1 seed
    cols = {"arm", "seed", "mAP_full", "mAP_holdout", "exact_full",
            "pos_prob_full", "neg_prob_full"}
    assert cols.issubset(set(rows[0].keys()))
