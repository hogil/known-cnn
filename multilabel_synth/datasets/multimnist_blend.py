"""Learnable-operator support for the operator-match control.

The operator-match harness (multimnist_operator.py) hand-selects one of two
KNOWN combination operators per domain:

  overlay   (superposition): two digits share ONE cell, combined by pixelwise max
  partition (FCM-PM)       : two digits occupy TWO distinct cells (disjoint)

This module makes the operator a LEARNABLE convex blend of those two primitives
so the operator can be learned from data instead of hand-picked. For a pair
(a, b) with digit images (da, db) and a shared first cell ka plus a distinct
second cell kb, we precompute two canvases:

  O (overlay canvas)   : cell ka = max(da, db); cell kb = 0
  Q (partition canvas) : cell ka = da;          cell kb = db

The blended operator with gate g in [0, 1] is the exact convex interpolation

  x(g) = g * O + (1 - g) * Q

  g = 1 -> cell ka = max(da, db), cell kb = 0            == pure superposition op
  g = 0 -> cell ka = da,          cell kb = db           == pure partition op

so the two ENDPOINTS coincide exactly with the two hand-selected operators of
the paper's harness (build_multi law='superposition' / 'partition'); a learned g
interpolates between them. g is a single scalar (or per-example) that a learner
can move without ever being told the domain's true combination law.

Single-label examples are represented with O == Q == single-canvas, so blending
leaves them g-independent (they carry no gradient signal for g), letting a mixed
single+combo pool feed one learner while only combos inform the operator.
"""
import numpy as np

from .multimnist import _index_by_class
from .multimnist_operator import canvas_size, _cell_origin, build_singles


def build_blend_pairs(imgs, labels, n, seed, allowed_pairs,
                      grid=2, cell=28, n_classes=10):
    """Paired (O, Q, Y) two-label canvases for the learnable-operator blend.

    O and Q are built from the SAME source digits and the SAME cell assignment
    (shared first cell ka, distinct second cell kb), so x(g)=g*O+(1-g)*Q is a
    well-defined interpolation whose endpoints are exactly the overlay and
    partition operators.
    """
    canvas = canvas_size(grid, cell)
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    n_cell = grid * grid
    pairs = list(allowed_pairs)
    O, Q, Y = [], [], []
    for _ in range(n):
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        da = imgs[int(rng.choice(by[a]))].astype(np.float32)
        db = imgs[int(rng.choice(by[b]))].astype(np.float32)
        ka, kb = rng.choice(n_cell, size=2, replace=False)
        ya, xa = _cell_origin(int(ka), grid, cell)
        yb, xb = _cell_origin(int(kb), grid, cell)
        o = np.zeros((canvas, canvas), dtype=np.float32)
        q = np.zeros((canvas, canvas), dtype=np.float32)
        o[ya:ya + cell, xa:xa + cell] = np.maximum(da, db)   # overlay in cell ka
        q[ya:ya + cell, xa:xa + cell] = da                   # partition: da in ka
        q[yb:yb + cell, xb:xb + cell] = db                   #            db in kb
        t = np.zeros(n_classes, dtype=np.float32)
        t[a] = 1.0
        t[b] = 1.0
        O.append(o)
        Q.append(q)
        Y.append(t)
    O = np.stack(O)[:, None, :, :].astype(np.float32) / 255.0
    Q = np.stack(Q)[:, None, :, :].astype(np.float32) / 255.0
    return O, Q, np.stack(Y)


def build_blend_pool(imgs, labels, n_combo, per_class_single, seed,
                     allowed_pairs, grid=2, cell=28, n_classes=10):
    """Mixed combo+single pool as (O, Q, Y).

    Combos have O != Q (the two operator canvases). Singles have O == Q ==
    single-canvas, so blending is g-independent for them. A meta-learner can
    sample uniformly from the pool; only combos carry an operator gradient.
    """
    O, Q, Y = build_blend_pairs(imgs, labels, n_combo, seed, allowed_pairs,
                                grid=grid, cell=cell, n_classes=n_classes)
    spX, spY = build_singles(imgs, labels, per_class_single, seed,
                             grid=grid, cell=cell, n_classes=n_classes)
    O = np.concatenate([O, spX])
    Q = np.concatenate([Q, spX])          # singles: O == Q
    Y = np.concatenate([Y, spY])
    return O, Q, Y
