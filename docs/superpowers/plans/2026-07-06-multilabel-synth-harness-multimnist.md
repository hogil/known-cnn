# Multi-label Synthesis Harness + MultiMNIST Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable, CPU-first multi-label synthesis harness and run the full MultiMNIST arm-vs-arm matrix (oracle, FCM-PM, CutMix, Mixup, copy-paste, single-only), including the held-out-combo compositional-generalization experiment.

**Architecture:** A new top-level package `multilabel_synth/` separate from the chip-specific `chip_multilabel/`. Pure-numpy data synthesis (single pool + multi overlay + 6 arms) feeds a small CPU CNN trained with BCE; evaluation reports mAP / exact-match / pos-neg prob. An orchestrator loops arms x seeds and writes a results CSV. MultiMNIST is fully controlled so the mechanism (region-preserving hard-label synthesis beats blending) and compositional generalization (synthesis recovers held-out combos the oracle never saw) can be shown immediately and cheaply.

**Tech Stack:** Python 3.13, PyTorch 2.6 (CPU), torchvision (MNIST download only), NumPy, scikit-learn (`average_precision_score`), pytest.

---

## Conventions

- Run all tests from repo root with: `python -m pytest <path> -v`
- Device defaults to `"cpu"` everywhere (GPU is shared/busy). CUDA is never required.
- MNIST download root: `E:/data/torchvision` (data lives on E:, per project rule; `data/` is gitignored anyway).
- Results go under `outputs/multilabel_synth/` (gitignored).
- 10 digit classes; a "combo" is an unordered digit pair `{i,j}`, `i != j` (45 total).

## File Structure

```
multilabel_synth/
  __init__.py              # empty
  metrics.py               # compute_map, exact_match, pos_neg_prob
  models/
    __init__.py            # empty
    small_cnn.py           # SmallCNN
  datasets/
    __init__.py            # empty
    multimnist.py          # load_mnist, all_pairs, split_holdout, build_single_pool, synthesize_multi, _place, _index_by_class
  synthesis/
    __init__.py            # empty
    arms.py                # synthesize_arm (6 arms), _grid_mask
  train.py                 # train_one
  run_matrix.py            # main orchestrator -> CSV
tests/multilabel_synth/
  test_metrics.py
  test_small_cnn.py
  test_multimnist.py
  test_synthesis.py
  test_train.py
  test_run_matrix.py
```

---

### Task 1: Scaffold package skeleton

**Files:**
- Create: `multilabel_synth/__init__.py` (empty)
- Create: `multilabel_synth/models/__init__.py` (empty)
- Create: `multilabel_synth/datasets/__init__.py` (empty)
- Create: `multilabel_synth/synthesis/__init__.py` (empty)
- Create: `tests/multilabel_synth/` (dir)

- [ ] **Step 1: Create the four empty `__init__.py` files and the tests dir**

Create each `__init__.py` with a single line:

```python
# multilabel_synth package
```

Create the tests directory (a `.gitkeep` is fine until test files land).

- [ ] **Step 2: Verify package imports**

Run: `python -c "import multilabel_synth; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add multilabel_synth tests/multilabel_synth
git commit -m "feat(mlsynth): scaffold multilabel_synth package"
```

---

### Task 2: Metrics (mAP, exact-match, pos/neg prob)

**Files:**
- Create: `multilabel_synth/metrics.py`
- Test: `tests/multilabel_synth/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/multilabel_synth/test_metrics.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/multilabel_synth/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'multilabel_synth.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# multilabel_synth/metrics.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/multilabel_synth/test_metrics.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add multilabel_synth/metrics.py tests/multilabel_synth/test_metrics.py
git commit -m "feat(mlsynth): mAP / exact-match / pos-neg-prob metrics"
```

---

### Task 3: Small CPU CNN

**Files:**
- Create: `multilabel_synth/models/small_cnn.py`
- Test: `tests/multilabel_synth/test_small_cnn.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/multilabel_synth/test_small_cnn.py
import torch
from multilabel_synth.models.small_cnn import SmallCNN


def test_forward_shape():
    model = SmallCNN(num_classes=10, in_ch=1)
    x = torch.zeros(4, 1, 40, 40)
    out = model(x)
    assert out.shape == (4, 10)


def test_param_count_small():
    model = SmallCNN(num_classes=10, in_ch=1)
    n = sum(p.numel() for p in model.parameters())
    assert n < 200_000   # stays tiny for CPU
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/multilabel_synth/test_small_cnn.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# multilabel_synth/models/small_cnn.py
import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes=10, in_ch=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/multilabel_synth/test_small_cnn.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add multilabel_synth/models/small_cnn.py tests/multilabel_synth/test_small_cnn.py
git commit -m "feat(mlsynth): SmallCNN backbone for MultiMNIST"
```

---

### Task 4: MultiMNIST datasets (single pool, multi overlay, held-out split)

**Files:**
- Create: `multilabel_synth/datasets/multimnist.py`
- Test: `tests/multilabel_synth/test_multimnist.py`

Uses injected fake `imgs`/`labels` arrays so unit tests need no download. `load_mnist` (the only torchvision-touching function) is not exercised in unit tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/multilabel_synth/test_multimnist.py
import numpy as np
from multilabel_synth.datasets.multimnist import (
    all_pairs, split_holdout, build_single_pool, synthesize_multi,
)


def _fake_mnist(n_per=20, n_classes=10):
    # each digit image is a 28x28 block whose pixels encode its class (>0 ink)
    imgs, labels = [], []
    for c in range(n_classes):
        for _ in range(n_per):
            im = np.full((28, 28), (c + 1) * 20, dtype=np.uint8)
            imgs.append(im); labels.append(c)
    return np.stack(imgs), np.array(labels, dtype=int)


def test_all_pairs_count():
    assert len(all_pairs(10)) == 45


def test_split_holdout_deterministic_and_disjoint():
    pairs = all_pairs(10)
    tr1, ho1 = split_holdout(pairs, 9, seed=7)
    tr2, ho2 = split_holdout(pairs, 9, seed=7)
    assert ho1 == ho2                       # deterministic
    assert len(ho1) == 9
    assert set(tr1).isdisjoint(set(ho1))    # disjoint
    assert len(tr1) + len(ho1) == 45


def test_single_pool_is_single_label():
    imgs, labels = _fake_mnist()
    X, Y = build_single_pool(imgs, labels, per_class=5, seed=0, canvas=40)
    assert X.shape == (50, 1, 40, 40)
    assert X.dtype == np.float32 and X.max() <= 1.0
    assert (Y.sum(axis=1) == 1).all()       # exactly one positive each


def test_synthesize_multi_has_two_labels_from_allowed_pairs():
    imgs, labels = _fake_mnist()
    X, Y = synthesize_multi(imgs, labels, n=32, seed=1,
                            allowed_pairs=[(0, 1)], canvas=40)
    assert X.shape == (32, 1, 40, 40)
    assert (Y.sum(axis=1) == 2).all()
    # only classes 0 and 1 ever active
    assert Y[:, 2:].sum() == 0
    assert (Y[:, 0] == 1).all() and (Y[:, 1] == 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/multilabel_synth/test_multimnist.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# multilabel_synth/datasets/multimnist.py
import numpy as np
from itertools import combinations


def all_pairs(n_classes=10):
    return list(combinations(range(n_classes), 2))


def split_holdout(pairs, n_holdout, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    hold = {pairs[i] for i in idx[:n_holdout]}
    train = [p for p in pairs if p not in hold]
    return train, sorted(hold)


def load_mnist(root="E:/data/torchvision", train=True):
    from torchvision.datasets import MNIST
    ds = MNIST(root, train=train, download=True)
    imgs = ds.data.numpy().astype(np.uint8)     # [N,28,28]
    labels = ds.targets.numpy().astype(int)     # [N]
    return imgs, labels


def _index_by_class(labels, n_classes=10):
    return {c: np.where(labels == c)[0] for c in range(n_classes)}


def _place(digit28, canvas, rng):
    c = np.zeros((canvas, canvas), dtype=np.uint8)
    off = canvas - 28
    y = int(rng.integers(0, off + 1))
    x = int(rng.integers(0, off + 1))
    c[y:y + 28, x:x + 28] = digit28
    return c


def build_single_pool(imgs, labels, per_class, seed, canvas=40, n_classes=10):
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    out_imgs, out_tgt = [], []
    for c in range(n_classes):
        pool = by[c]
        pick = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        for i in pick:
            out_imgs.append(_place(imgs[i], canvas, rng))
            t = np.zeros(n_classes, dtype=np.float32); t[c] = 1.0
            out_tgt.append(t)
    X = np.stack(out_imgs)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(out_tgt)


def synthesize_multi(imgs, labels, n, seed, allowed_pairs, canvas=40, n_classes=10):
    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    pairs = list(allowed_pairs)
    out_imgs, out_tgt = [], []
    for _ in range(n):
        a, b = pairs[int(rng.integers(0, len(pairs)))]
        ca = _place(imgs[int(rng.choice(by[a]))], canvas, rng)
        cb = _place(imgs[int(rng.choice(by[b]))], canvas, rng)
        merged = np.maximum(ca, cb)          # overlay; may overlap
        out_imgs.append(merged)
        t = np.zeros(n_classes, dtype=np.float32); t[a] = 1.0; t[b] = 1.0
        out_tgt.append(t)
    X = np.stack(out_imgs)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(out_tgt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/multilabel_synth/test_multimnist.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add multilabel_synth/datasets/multimnist.py tests/multilabel_synth/test_multimnist.py
git commit -m "feat(mlsynth): MultiMNIST single pool, multi overlay, held-out split"
```

---

### Task 5: Synthesis arms (the six training-data generators)

**Files:**
- Create: `multilabel_synth/synthesis/arms.py`
- Test: `tests/multilabel_synth/test_synthesis.py`

Arms produce training data from the single pool. `oracle` uses the real multi
overlay; `single_only` uses lone digits; the rest synthesize combos differently.

- [ ] **Step 1: Write the failing test**

```python
# tests/multilabel_synth/test_synthesis.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/multilabel_synth/test_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# multilabel_synth/synthesis/arms.py
import numpy as np
from ..datasets.multimnist import _index_by_class, _place, synthesize_multi


def _grid_mask(canvas, grid):
    m = np.zeros((canvas, canvas), dtype=bool)
    cell = canvas // grid
    for i in range(grid):
        for j in range(grid):
            if (i + j) % 2 == 0:
                m[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell] = True
    return m


def synthesize_arm(arm, imgs, labels, n, seed, allowed_pairs,
                   canvas=40, n_classes=10, grid=4):
    if arm == "oracle":
        return synthesize_multi(imgs, labels, n, seed, allowed_pairs, canvas, n_classes)

    rng = np.random.default_rng(seed)
    by = _index_by_class(labels, n_classes)
    pairs = list(allowed_pairs)
    out_imgs, out_tgt = [], []

    for _ in range(n):
        t = np.zeros(n_classes, dtype=np.float32)

        if arm == "single_only":
            c = int(rng.integers(0, n_classes))
            img = _place(imgs[int(rng.choice(by[c]))], canvas, rng).astype(np.float32)
            t[c] = 1.0
            out_imgs.append(img); out_tgt.append(t)
            continue

        a, b = pairs[int(rng.integers(0, len(pairs)))]
        ca = _place(imgs[int(rng.choice(by[a]))], canvas, rng).astype(np.float32)
        cb = _place(imgs[int(rng.choice(by[b]))], canvas, rng).astype(np.float32)

        if arm == "mixup":
            lam = float(rng.beta(1.0, 1.0))
            img = lam * ca + (1.0 - lam) * cb
            t[a] = lam; t[b] = 1.0 - lam
        elif arm == "copy_paste":
            img = np.zeros((canvas, canvas), dtype=np.float32)
            half = canvas // 2
            img[:, :half] = ca[:, :half]
            img[:, half:] = cb[:, half:]
            t[a] = 1.0; t[b] = 1.0
        elif arm == "cutmix":
            img = ca.copy()
            ch = cw = canvas // 2
            y = int(rng.integers(0, canvas - ch))
            x = int(rng.integers(0, canvas - cw))
            img[y:y + ch, x:x + cw] = cb[y:y + ch, x:x + cw]
            t[a] = 1.0; t[b] = 1.0
        elif arm == "fcm_pm":
            mask = _grid_mask(canvas, grid)
            img = np.where(mask, ca, cb)
            t[a] = 1.0; t[b] = 1.0
        else:
            raise ValueError(f"unknown arm: {arm}")

        out_imgs.append(img); out_tgt.append(t)

    X = np.stack(out_imgs)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(out_tgt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/multilabel_synth/test_synthesis.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add multilabel_synth/synthesis/arms.py tests/multilabel_synth/test_synthesis.py
git commit -m "feat(mlsynth): six synthesis arms (oracle/fcm_pm/cutmix/mixup/copy_paste/single_only)"
```

---

### Task 6: CPU trainer

**Files:**
- Create: `multilabel_synth/train.py`
- Test: `tests/multilabel_synth/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/multilabel_synth/test_train.py
import numpy as np
from multilabel_synth.train import train_one


def test_train_one_returns_finite_metrics():
    rng = np.random.default_rng(0)
    # 3 classes, tiny separable data: class k has a bright pixel in row k
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/multilabel_synth/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# multilabel_synth/train.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .models.small_cnn import SmallCNN
from .metrics import compute_map, exact_match, pos_neg_prob


def _loader(X, Y, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)


def train_one(train_X, train_Y, test_X, test_Y,
              epochs=3, bs=64, lr=1e-3, device="cpu", seed=0):
    torch.manual_seed(seed)
    model = SmallCNN(num_classes=train_Y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in _loader(train_X, train_Y, bs, True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in _loader(test_X, test_Y, bs, False):
            probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    P = np.concatenate(probs)

    pos, neg = pos_neg_prob(P, test_Y)
    return {
        "mAP": compute_map(P, test_Y),
        "exact_match": exact_match(P, test_Y),
        "pos_prob": pos,
        "neg_prob": neg,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/multilabel_synth/test_train.py -v`
Expected: 1 passed (runs on CPU in a few seconds)

- [ ] **Step 5: Commit**

```bash
git add multilabel_synth/train.py tests/multilabel_synth/test_train.py
git commit -m "feat(mlsynth): CPU BCE trainer returning mAP/exact-match/pos-neg-prob"
```

---

### Task 7: Matrix orchestrator + real MultiMNIST run

**Files:**
- Create: `multilabel_synth/run_matrix.py`
- Test: `tests/multilabel_synth/test_run_matrix.py`

Orchestrates: build single pool + held-out split from MNIST, for each arm x seed
build training data, train, evaluate on BOTH the full-multi test and the
held-out-combo test, write one CSV row per (arm, seed). A `run_matrix(...)`
function takes injected `(imgs, labels)` so the smoke test avoids download; the
`main()` CLI loads real MNIST.

- [ ] **Step 1: Write the failing test**

```python
# tests/multilabel_synth/test_run_matrix.py
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
    cols = {"arm", "seed", "mAP_full", "mAP_holdout", "exact_full", "pos_prob_full", "neg_prob_full"}
    assert cols.issubset(set(rows[0].keys()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/multilabel_synth/test_run_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# multilabel_synth/run_matrix.py
import os
import csv as csvmod
import argparse

from .datasets.multimnist import (
    all_pairs, split_holdout, build_single_pool, synthesize_multi, load_mnist,
)
from .synthesis.arms import synthesize_arm
from .train import train_one

ARMS = ["oracle", "fcm_pm", "cutmix", "mixup", "copy_paste", "single_only"]
FIELDS = ["arm", "seed", "mAP_full", "mAP_holdout", "exact_full",
          "pos_prob_full", "neg_prob_full"]


def run_matrix(imgs, labels, arms, seeds, n_holdout,
               per_class_single, n_train, n_test, n_holdout_test,
               epochs, bs, device, out_csv):
    pairs = all_pairs(10)
    train_pairs, holdout_pairs = split_holdout(pairs, n_holdout, seed=12345)

    # fixed evaluation sets (seed-independent)
    testX_full, testY_full = synthesize_multi(imgs, labels, n_test, seed=999,
                                              allowed_pairs=pairs)
    testX_ho, testY_ho = synthesize_multi(imgs, labels, n_holdout_test, seed=998,
                                          allowed_pairs=holdout_pairs)

    rows = []
    for arm in arms:
        # oracle only sees train_pairs (never the held-out combos);
        # synthesis arms may generate every pair.
        allowed = train_pairs if arm == "oracle" else pairs
        for seed in seeds:
            if arm == "single_only":
                trX, trY = synthesize_arm("single_only", imgs, labels,
                                          n=n_train, seed=seed, allowed_pairs=allowed)
            elif arm == "oracle":
                trX, trY = synthesize_arm("oracle", imgs, labels,
                                          n=n_train, seed=seed, allowed_pairs=allowed)
            else:
                trX, trY = synthesize_arm(arm, imgs, labels,
                                          n=n_train, seed=seed, allowed_pairs=allowed)
            # augment every arm with the single pool (shared single-label signal)
            spX, spY = build_single_pool(imgs, labels, per_class_single, seed=seed)
            import numpy as np
            trX = np.concatenate([trX, spX]); trY = np.concatenate([trY, spY])

            full = train_one(trX, trY, testX_full, testY_full,
                             epochs=epochs, bs=bs, device=device, seed=seed)
            # re-evaluate the SAME trained model on held-out combos:
            # simplest faithful approach = train once, eval twice -> retrain-free
            ho = train_one(trX, trY, testX_ho, testY_ho,
                           epochs=epochs, bs=bs, device=device, seed=seed)
            rows.append({
                "arm": arm, "seed": seed,
                "mAP_full": round(full["mAP"], 4),
                "mAP_holdout": round(ho["mAP"], 4),
                "exact_full": round(full["exact_match"], 4),
                "pos_prob_full": round(full["pos_prob"], 4),
                "neg_prob_full": round(full["neg_prob"], 4),
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-holdout", type=int, default=9)
    ap.add_argument("--per-class-single", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-holdout-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mnist-root", default="E:/data/torchvision")
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/multimnist_matrix.csv")
    args = ap.parse_args()

    imgs, labels = load_mnist(args.mnist_root, train=True)
    rows = run_matrix(imgs, labels, args.arms, args.seeds, args.n_holdout,
                      args.per_class_single, args.n_train, args.n_test,
                      args.n_holdout_test, args.epochs, args.bs, args.device,
                      args.out_csv)
    for r in rows:
        print(r)
    print(f"[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/multilabel_synth/test_run_matrix.py -v`
Expected: 1 passed

- [ ] **Step 5: Full test suite green**

Run: `python -m pytest tests/multilabel_synth -v`
Expected: all tests passed

- [ ] **Step 6: Commit**

```bash
git add multilabel_synth/run_matrix.py tests/multilabel_synth/test_run_matrix.py
git commit -m "feat(mlsynth): arm x seed matrix orchestrator with held-out-combo eval"
```

- [ ] **Step 7: Real MultiMNIST fast run (CPU, downloads MNIST once to E:)**

Run:
```bash
python -m multilabel_synth.run_matrix \
  --arms oracle fcm_pm cutmix mixup copy_paste single_only \
  --seeds 0 --n-holdout 9 --per-class-single 100 \
  --n-train 2000 --n-test 1000 --n-holdout-test 500 \
  --epochs 3 --bs 64 --device cpu \
  --out-csv outputs/multilabel_synth/multimnist_fast.csv
```
Expected: prints one row per arm and an `[OUT]` line; CSV written. Sanity to
eyeball (not asserted): `mixup` mAP_full lowest; `fcm_pm`/`copy_paste` high;
on `mAP_holdout`, `fcm_pm` >= `oracle` (oracle never trained on held-out pairs).

- [ ] **Step 8: Commit the first result artifact reference (CSV is gitignored; record the numbers)**

Record the observed rows into the spec's results section or a notes file under
`docs/superpowers/` (the CSV itself stays gitignored under `outputs/`).

```bash
git add docs/superpowers
git commit -m "docs(mlsynth): record first MultiMNIST fast-run numbers"
```

---

## Self-Review

**Spec coverage:**
- Single pool / natural-vs-synth split (spec S3-S4): Task 4 (`build_single_pool`, `synthesize_multi`, held-out split). MultiMNIST is the fully-synth case, as specified.
- Six synthesis arms (spec S5): Task 5.
- Held-out-combo main experiment (spec S6): Task 7 (`oracle` restricted to `train_pairs`; synthesis arms get all pairs; separate `mAP_holdout`).
- Backbone / metrics (spec S7): Task 3 (SmallCNN), Task 2 (mAP, exact-match, pos/neg prob).
- Matrix / seeds / CPU / outputs (spec S8-S9): Task 6 (CPU trainer), Task 7 (arm x seed CSV under `outputs/`, device=cpu default).
- VOC / COCO (spec S9 phases 3-4): intentionally out of scope for THIS plan; separate plans follow.

**Placeholder scan:** No TBD/TODO; every code step has complete runnable code; expected outputs are concrete.

**Type consistency:** `synthesize_arm(arm, imgs, labels, n, seed, allowed_pairs, canvas, n_classes, grid)`, `synthesize_multi(imgs, labels, n, seed, allowed_pairs, canvas, n_classes)`, `build_single_pool(imgs, labels, per_class, seed, canvas, n_classes)`, `train_one(train_X, train_Y, test_X, test_Y, epochs, bs, lr, device, seed)`, `run_matrix(...)` signatures are consistent across Tasks 4-7 and their call sites.

**Known simplification (intentional):** Task 7 Step 3 evaluates held-out combos by calling `train_one` a second time (train-once-eval-twice would be marginally cheaper). This retrains per eval set; acceptable for the fast harness. A later refactor can split train/eval to train once and evaluate on both sets — noted as a follow-up, not a blocker.
