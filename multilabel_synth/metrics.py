import numpy as np
from sklearn.metrics import average_precision_score


def compute_map(probs, targets):
    """Macro mAP over classes that have at least one positive."""
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    aps = []
    for c in range(targets.shape[1]):
        if targets[:, c].sum() == 0:
            continue
        aps.append(average_precision_score(targets[:, c], probs[:, c]))
    return float(np.mean(aps)) if aps else 0.0


def exact_match(probs, targets, thr=0.5):
    pred = (np.asarray(probs, dtype=float) >= thr).astype(int)
    tgt = np.asarray(targets, dtype=int)
    return float((pred == tgt).all(axis=1).mean())


def pos_neg_prob(probs, targets):
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    pos = probs[targets == 1]
    neg = probs[targets == 0]
    return (float(pos.mean()) if pos.size else float("nan"),
            float(neg.mean()) if neg.size else float("nan"))
