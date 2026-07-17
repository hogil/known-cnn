"""DLRSD land-cover loader for the REAL-public operator-match study.

DLRSD (Dense Labeling Remote Sensing Dataset, Shao et al. 2018) is the UC-Merced
Land-Use archive (2100 aerial tiles, 256x256, 21 scene folders) re-annotated with
PIXEL-LEVEL masks for the SAME 17 land-cover classes as the Multi-label UC-Merced
label set (Chaudhuri et al. 2018). Pixel value v in {1..17} == land-cover class
index v-1 in the csv-header order below.

The partition combination law for remote sensing: an aerial tile's label set is
the UNION of the land-cover classes occupying DISJOINT spatial regions (buildings
here, trees there, grass there). That is exactly the partition law tested on chips
and on SVHN house numbers -- here on a REAL, public, non-constructed benchmark.

This loader provides:
  * a SINGLE-CLASS SOURCE pool: real pure single-class region crops harvested from
    the pixel masks (purity >= thresh), one label each -> the singles the synthesis
    arms combine.  (Land cover is intrinsically co-occurring, so image-level single
    -label tiles barely exist; pixel-mask region crops are the honest single sources.)
  * a REAL MULTI-LABEL pool: tiles with >= 2 of the chosen classes present, split
    disjointly into an oracle-train pool and a held-out eval set. Multi-hot targets
    over the chosen class subset (which classes present -- NOT pixel positions).

Eval images never donate single-class crops (no train/eval leakage).
"""
import os
import re
import time

import numpy as np
import pandas as pd
from PIL import Image

# csv-header order == pixel-value-minus-1 order
CLASS_NAMES_17 = ['airplane', 'baresoil', 'buildings', 'cars', 'chaparral',
                  'court', 'dock', 'field', 'grass', 'mobilehome', 'pavement',
                  'sand', 'sea', 'ship', 'tanks', 'trees', 'water']

# default clean subset: large, co-occurring urban/residential land-cover classes
# (all with >=90 pure single-class crops and heavy real co-occurrence)
DEFAULT_SUBSET = ['buildings', 'pavement', 'trees', 'grass', 'baresoil']


def _folder(name):
    return re.sub(r'\d+$', '', name)


def _square_crop_for_class(img, mask, c_pixval, min_side, purity):
    """Centered square crop over class c's mask bbox; return HxWx3 uint8 or None
    if the region is too small or not pure enough (fraction of c pixels < purity)."""
    ys, xs = np.where(mask == c_pixval)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    side = max(y1 - y0 + 1, x1 - x0 + 1)
    if side < min_side:
        return None
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    H, W = mask.shape
    half = side // 2
    yy0, yy1 = max(0, cy - half), min(H, cy + half + 1)
    xx0, xx1 = max(0, cx - half), min(W, cx + half + 1)
    sub_m = mask[yy0:yy1, xx0:xx1]
    if (sub_m == c_pixval).mean() < purity:
        return None
    return img[yy0:yy1, xx0:xx1]


def prepare_landcover(base, subset=None, cell=64, canvas=128, n_eval=400,
                      n_oracle=1000, per_class_cap=140, min_side=48,
                      purity=0.6, seed=0, cache=None):
    """Returns (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
    class_names, meta).

    pool_imgs   : [Np, cell, cell, 3] uint8  single-class region crops
    pool_labels : [Np] int   class index within `subset`
    oracle_imgs : [No, canvas, canvas, 3] uint8  real multi-label tiles
    eval_imgs   : [Ne, canvas, canvas, 3] uint8  real multi-label tiles (held out)
    *_Y         : multi-hot float32 over the subset
    """
    if subset is None:
        subset = list(DEFAULT_SUBSET)
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] loaded {cache}", flush=True)
        return (z["pool_imgs"], z["pool_labels"], z["oracle_imgs"], z["oracle_Y"],
                z["eval_imgs"], z["eval_Y"], list(z["class_names"]),
                z["meta"].item())

    t0 = time.time()
    sub_idx = [CLASS_NAMES_17.index(c) for c in subset]           # 0-based class ids
    sub_pixval = [i + 1 for i in sub_idx]                          # mask pixel values
    n_classes = len(subset)

    df = pd.read_csv(os.path.join(base, "multilabels.csv"))
    Msub = df[subset].values.astype(np.int64)                     # [2100, n_classes]
    k = Msub.sum(1)
    multi_mask = k >= 2
    multi_names = df['image'].values[multi_mask]
    multi_Y = Msub[multi_mask].astype(np.float32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(multi_names))
    eval_sel = perm[:n_eval]
    oracle_sel = perm[n_eval:n_eval + n_oracle]
    eval_names = set(multi_names[eval_sel])

    def _load_canvas(name):
        f = _folder(name)
        im = Image.open(os.path.join(base, "Images", f, name + ".tif")).convert("RGB")
        return np.asarray(im.resize((canvas, canvas), Image.BILINEAR), dtype=np.uint8)

    eval_imgs = np.stack([_load_canvas(multi_names[i]) for i in eval_sel])
    eval_Y = multi_Y[eval_sel]
    oracle_imgs = np.stack([_load_canvas(multi_names[i]) for i in oracle_sel])
    oracle_Y = multi_Y[oracle_sel]
    print(f"[prep] real multi eval {eval_imgs.shape} oracle {oracle_imgs.shape} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # single-class region crops from every NON-eval image
    crops = {ci: [] for ci in range(n_classes)}
    for name in df['image'].values:
        if name in eval_names:
            continue
        f = _folder(name)
        mask = np.asarray(Image.open(os.path.join(base, "Labels", f, name + ".png")))
        present = set(np.unique(mask).tolist())
        img = None
        for ci, pv in enumerate(sub_pixval):
            if pv not in present:
                continue
            if img is None:
                img = np.asarray(Image.open(
                    os.path.join(base, "Images", f, name + ".tif")).convert("RGB"),
                    dtype=np.uint8)
            crop = _square_crop_for_class(img, mask, pv, min_side, purity)
            if crop is not None:
                crops[ci].append(np.asarray(
                    Image.fromarray(crop).resize((cell, cell), Image.BILINEAR),
                    dtype=np.uint8))

    pool_imgs, pool_labels = [], []
    for ci in range(n_classes):
        arr = crops[ci]
        if len(arr) > per_class_cap:
            keep = rng.choice(len(arr), size=per_class_cap, replace=False)
            arr = [arr[i] for i in keep]
        for a in arr:
            pool_imgs.append(a)
            pool_labels.append(ci)
    pool_imgs = np.stack(pool_imgs)
    pool_labels = np.asarray(pool_labels, dtype=np.int64)
    per_class = {subset[ci]: int((pool_labels == ci).sum()) for ci in range(n_classes)}
    print(f"[prep] single-class crop pool {pool_imgs.shape} per_class={per_class} "
          f"({time.time()-t0:.0f}s)", flush=True)

    meta = dict(subset=list(subset), n_classes=n_classes, cell=cell, canvas=canvas,
                pool=list(pool_imgs.shape), per_class=per_class,
                oracle=list(oracle_imgs.shape), eval=list(eval_imgs.shape),
                purity=purity, min_side=min_side,
                eval_pos={subset[c]: int(eval_Y[:, c].sum()) for c in range(n_classes)})

    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, pool_imgs=pool_imgs, pool_labels=pool_labels,
                            oracle_imgs=oracle_imgs, oracle_Y=oracle_Y,
                            eval_imgs=eval_imgs, eval_Y=eval_Y,
                            class_names=np.array(subset, dtype=object),
                            meta=np.array(meta, dtype=object))
        print(f"[cache] saved {cache}", flush=True)
    return (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
            list(subset), meta)


def harvest_mask_templates(base, subset=None, canvas=128, n_eval=400,
                           min_frac=0.01, seed=0):
    """Real DLRSD pixel-mask LAYOUT templates for realistic partition synthesis.

    Returns a list of [canvas, canvas] int16 arrays whose values are the subset
    class id (0..n_classes-1) at each pixel, or -1 for a non-subset land-cover
    class.  Only tiles NOT in the held-out eval split donate templates (same
    seed=0 permutation as prepare_landcover -> identical eval split, no leakage).

    The template carries REAL region geometry + area statistics + class
    co-occurrence; the synthesis fills each class-region with that class's own
    single-source crop content (appearance stays synthetic -- so this isolates
    whether the sim-to-real gap is geometric or appearance-fundamental).

    A template is kept only if >= 2 subset classes each occupy >= min_frac of the
    tile (a genuine multi-label layout).
    """
    if subset is None:
        subset = list(DEFAULT_SUBSET)
    sub_pixval = [CLASS_NAMES_17.index(c) + 1 for c in subset]
    n_classes = len(subset)
    min_area = int(min_frac * canvas * canvas)

    df = pd.read_csv(os.path.join(base, "multilabels.csv"))
    Msub = df[subset].values.astype(np.int64)
    multi_mask = Msub.sum(1) >= 2
    multi_names = df['image'].values[multi_mask]

    # reproduce prepare_landcover's eval split exactly (fresh rng, first draw)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(multi_names))
    eval_names = set(multi_names[perm[:n_eval]])

    templates = []
    for name in multi_names:
        if name in eval_names:
            continue
        f = _folder(name)
        mask = np.asarray(Image.open(
            os.path.join(base, "Labels", f, name + ".png")).resize(
            (canvas, canvas), Image.NEAREST))
        tmpl = np.full((canvas, canvas), -1, dtype=np.int16)
        present = 0
        for ci, pv in enumerate(sub_pixval):
            region = (mask == pv)
            if region.sum() >= min_area:
                tmpl[region] = ci
                present += 1
        if present >= 2:
            templates.append(tmpl)
    print(f"[templates] {len(templates)} real multi-label mask layouts "
          f"(canvas={canvas}, min_frac={min_frac})", flush=True)
    return templates
