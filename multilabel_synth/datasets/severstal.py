"""Severstal Steel Defect Detection -- STRICT image-level multi-label loader.

The audit's confirmatory public industrial domain: it uniquely provides, in one
public dataset, a REAL normal class (defect-free images) + single-defect sources
+ multi-defect eval, derivable at the IMAGE LEVEL from the RLE annotations WITHOUT
ever reading pixel masks or bounding boxes -> strict, no-location.

train.csv format (Kaggle): rows of (ImageId, ClassId, EncodedPixels) where a
present defect class c in {1,2,3,4} yields one row with a non-empty
EncodedPixels; images with NO row (or all-empty) are DEFECT-FREE (normal). We use
ONLY the SET of present ClassIds per image (image-level multi-label); the RLE
pixels are never decoded.

Roles (from the public train split only; original test labels are private):
  - normal:  0 defect classes present  -> real known-good normal
  - single:  exactly 1 class present   -> single-defect source
  - multi:   >= 2 classes present       -> sealed multi-defect positive eval
"""
import os
import csv as csvmod
import collections
import numpy as np
from PIL import Image

N_CLASSES = 4                              # Severstal defect classes 1..4 -> idx 0..3
CANH, CANW = 128, 256                      # letterbox target (steel strips are wide)
LETTERBOX_FILL = 114                        # neutral gray (audit: black loses thin defects)


def parse_labels(train_csv):
    """train.csv -> {ImageId: set(class_idx present)}. Handles the two public
    layouts: long (ImageId,ClassId,EncodedPixels) and wide (ImageId_ClassId,
    EncodedPixels)."""
    present = collections.defaultdict(set)
    all_imgs = set()
    with open(train_csv, encoding="utf-8") as f:
        rd = csvmod.DictReader(f)
        cols = [c.strip() for c in rd.fieldnames]
        long_fmt = "ClassId" in cols
        for row in rd:
            if long_fmt:
                img = row["ImageId"].strip(); cid = int(row["ClassId"]); rle = row.get("EncodedPixels", "")
            else:                                   # wide: "ImageId_ClassId"
                key = row.get("ImageId_ClassId") or row.get("ImageId", "")
                img, cid = key.rsplit("_", 1); img = img.strip(); cid = int(cid)
                rle = row.get("EncodedPixels", "")
            all_imgs.add(img)
            if rle and rle.strip():
                present[img].add(cid - 1)           # 1..4 -> 0..3
    return present, all_imgs


def _letterbox(img, out_h, out_w, fill=LETTERBOX_FILL):
    w, h = img.size
    s = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = np.full((out_h, out_w, 3), fill, np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = np.asarray(r.convert("RGB"), np.uint8)[..., :3]
    return canvas


def metadata_gate(train_csv):
    """Audit eligibility gate (metadata only; no pixels). Returns (ok, report)."""
    present, all_imgs = parse_labels(train_csv)
    normal = [i for i in all_imgs if len(present.get(i, ())) == 0]
    single = [i for i in all_imgs if len(present.get(i, ())) == 1]
    multi = [i for i in all_imgs if len(present.get(i, ())) >= 2]
    per_class_single = collections.Counter()
    for i in single:
        per_class_single[next(iter(present[i]))] += 1
    pair = collections.Counter()
    for i in multi:
        cs = sorted(present[i])
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                pair[(cs[a], cs[b])] += 1
    report = dict(
        n_images=len(all_imgs), n_normal=len(normal), n_single=len(single),
        n_multi=len(multi), per_class_single=dict(per_class_single),
        n_pairs=len(pair), pairs_ge30=sum(1 for v in pair.values() if v >= 30))
    ok = (report["n_multi"] >= 300 and min(per_class_single.values() or [0]) >= 200
          and report["n_normal"] >= 1000 and report["pairs_ge30"] >= 3)
    report["GATE"] = "PASS" if ok else "FAIL"
    return ok, report


def load_split(root, seed=0):
    """Deterministic strict split of the PUBLIC train set (private test unused).
    single: 80% train / 20% source-val ; normal: 50% calib / 50% sealed FAR test ;
    multi: all sealed positive test. Returns dict of (ids, image-level Y)."""
    train_csv = os.path.join(root, "train.csv")
    present, all_imgs = parse_labels(train_csv)
    rng = np.random.default_rng(seed)

    def Y(ids):
        arr = np.zeros((len(ids), N_CLASSES), np.float32)
        for k, i in enumerate(ids):
            for c in present.get(i, ()):
                arr[k, c] = 1.0
        return arr

    normal = sorted(i for i in all_imgs if len(present.get(i, ())) == 0)
    single = sorted(i for i in all_imgs if len(present.get(i, ())) == 1)
    multi = sorted(i for i in all_imgs if len(present.get(i, ())) >= 2)
    rng.shuffle(normal); rng.shuffle(single); rng.shuffle(multi)
    ns = int(0.8 * len(single)); nn = len(normal) // 2
    return dict(
        single_train=(single[:ns], Y(single[:ns])),
        single_val=(single[ns:], Y(single[ns:])),
        normal_cal=(normal[:nn], Y(normal[:nn])),
        normal_test=(normal[nn:], Y(normal[nn:])),
        multi_test=(multi, Y(multi)),
        img_dir=os.path.join(root, "train_images"))


def load_images(ids, img_dir, canh=CANH, canw=CANW, fill=LETTERBOX_FILL):
    out = []
    for i in ids:
        p = os.path.join(img_dir, i)
        out.append(_letterbox(Image.open(p), canh, canw, fill=fill))
    return np.stack(out) if out else np.zeros((0, canh, canw, 3), np.uint8)
