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


def bit_f1(probs, targets, thr=0.5):
    """Macro-F1 over class bits (chip-domain bit_F1 analog)."""
    pred = (np.asarray(probs, dtype=float) >= thr).astype(int)
    tgt = np.asarray(targets, dtype=int)
    f1s = []
    for c in range(tgt.shape[1]):
        tp = int(((pred[:, c] == 1) & (tgt[:, c] == 1)).sum())
        fp = int(((pred[:, c] == 1) & (tgt[:, c] == 0)).sum())
        fn = int(((pred[:, c] == 0) & (tgt[:, c] == 1)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def far(probs, targets, thr=0.5):
    """False-alarm rate on negative bits: FP / (# true-negative bit slots).

    Closed-set analog of chip FAR (no Normal/Invalid/OOD reject class in VOC).
    """
    pred = (np.asarray(probs, dtype=float) >= thr).astype(int)
    tgt = np.asarray(targets, dtype=int)
    neg = (tgt == 0)
    n_neg = int(neg.sum())
    fp = int(((pred == 1) & neg).sum())
    return float(fp / n_neg) if n_neg else 0.0


def pos_neg_prob(probs, targets):
    probs = np.asarray(probs, dtype=float)
    targets = np.asarray(targets, dtype=int)
    pos = probs[targets == 1]
    neg = probs[targets == 0]
    return (float(pos.mean()) if pos.size else float("nan"),
            float(neg.mean()) if neg.size else float("nan"))
