import numpy as np
from multilabel_synth.metrics import compute_map, exact_match, pos_neg_prob


def test_perfect_predictions():
    targets = np.array([[1, 0, 0], [0, 1, 1]], dtype=int)
    probs = np.array([[0.9, 0.1, 0.2], [0.2, 0.8, 0.7]], dtype=float)
    assert compute_map(probs, targets) == 1.0
    assert exact_match(probs, targets) == 1.0


def test_pos_neg_prob_separates():
    targets = np.array([[1, 0], [0, 1]], dtype=int)
    probs = np.array([[0.8, 0.1], [0.2, 0.9]], dtype=float)
    pos, neg = pos_neg_prob(probs, targets)
    assert abs(pos - 0.85) < 1e-9   # mean of 0.8, 0.9
    assert abs(neg - 0.15) < 1e-9   # mean of 0.1, 0.2


def test_map_skips_all_negative_classes():
    # class 2 has no positives -> excluded from macro mAP
    targets = np.array([[1, 0, 0], [0, 1, 0]], dtype=int)
    probs = np.array([[0.9, 0.1, 0.5], [0.1, 0.9, 0.5]], dtype=float)
    assert compute_map(probs, targets) == 1.0
