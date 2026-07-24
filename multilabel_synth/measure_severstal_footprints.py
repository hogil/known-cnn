"""Gate item 3 (measured footprints): decode Severstal RLE masks and measure how many
9x9 grid cells each defect footprint occupies (r), at the DEPLOYED FCM grid
(grid=9 -> N=81 cells, n_groups=3 -> m=27 assigned to source A). Then plug the measured
r into the T6-FCM coherence formula P(both) = C(N-r_a-r_b, m-r_a)/C(N,m) to get the
REAL preservation probability for steel defects. Tests the mechanism hypothesis that
steel defects are 'extended' (large r -> low preservation). No training.

NOTE: this corrects an earlier doc error -- the deployed operator grid is 9x9 (N=81,
m=27), NOT 3x3; 'g3' denotes n_groups=3, not grid-size 3.
"""
import csv as csvmod
import collections
import numpy as np
from math import comb
from PIL import Image
import os, glob

ROOT = "E:/data/severstal"
GRID = 9          # deployed grid side -> N = 81 cells
N_CELLS = GRID * GRID
M_A = N_CELLS // 3  # n_groups=3 -> 27 cells to source A


def decode_rle(rle, h, w):
    """Column-major (Fortran) 1-indexed RLE -> boolean HxW mask."""
    mask = np.zeros(h * w, np.bool_)
    nums = list(map(int, rle.split()))
    for i in range(0, len(nums), 2):
        start = nums[i] - 1; length = nums[i + 1]
        mask[start:start + length] = True
    return mask.reshape((w, h)).T   # column-major -> (h,w)


def grid_r(mask):
    """Number of 9x9 grid cells the footprint touches."""
    h, w = mask.shape
    occ = 0
    for i in range(GRID):
        for j in range(GRID):
            y0, y1 = i * h // GRID, (i + 1) * h // GRID
            x0, x1 = j * w // GRID, (j + 1) * w // GRID
            if mask[y0:y1, x0:x1].any():
                occ += 1
    return occ


def main():
    train_csv = os.path.join(ROOT, "train.csv")
    # get one image's dims
    f0 = sorted(glob.glob(os.path.join(ROOT, "train_images", "*")))[0]
    W, H = Image.open(f0).size
    print(f"image dims WxH = {W}x{H}; grid {GRID}x{GRID} (N={N_CELLS}, m={M_A})")

    present = collections.defaultdict(dict)   # img -> {cls: rle}
    with open(train_csv, encoding="utf-8") as fh:
        rd = csvmod.DictReader(fh)
        for row in rd:
            key = row.get("ImageId_ClassId") or ""
            rle = (row.get("EncodedPixels") or "").strip()
            if not rle:
                continue
            img, cid = key.rsplit("_", 1)
            present[img][int(cid) - 1] = rle

    rs = collections.defaultdict(list)   # class -> list of r
    rs_multi = []                         # (r_a, r_b) for multi-defect images
    n_img = 0
    for img, d in present.items():
        n_img += 1
        rvals = {}
        for cls, rle in d.items():
            try:
                m = decode_rle(rle, H, W); r = grid_r(m)
            except Exception:
                continue
            rs[cls].append(r); rvals[cls] = r
        if len(rvals) >= 2:
            vs = sorted(rvals.values(), reverse=True)
            rs_multi.append((vs[0], vs[1]))
        if n_img >= 4000:
            break

    print(f"\nfootprint cell-count r (of {N_CELLS}) per defect class:")
    print(f"  {'class':6s} {'n':>5s} {'mean r':>7s} {'median':>7s} {'p10':>5s} {'p90':>5s}")
    allr = []
    for cls in sorted(rs):
        a = np.array(rs[cls]); allr += list(a)
        print(f"  {cls:6d} {len(a):5d} {a.mean():7.1f} {np.median(a):7.0f} "
              f"{np.percentile(a,10):5.0f} {np.percentile(a,90):5.0f}")
    allr = np.array(allr)
    print(f"  {'ALL':6s} {len(allr):5d} {allr.mean():7.1f} {np.median(allr):7.0f} "
          f"{np.percentile(allr,10):5.0f} {np.percentile(allr,90):5.0f}")

    # preservation probability at measured footprints
    def P(ra, rb):
        if M_A - ra < 0 or N_CELLS - ra - rb < M_A - ra:
            return 0.0
        return comb(N_CELLS - ra - rb, M_A - ra) / comb(N_CELLS, N_CELLS // 3 + (M_A - N_CELLS // 3))

    def Pboth(ra, rb):
        if M_A - ra < 0 or N_CELLS - ra - rb < M_A - ra:
            return 0.0
        return comb(N_CELLS - ra - rb, M_A - ra) / comb(N_CELLS, M_A)

    print(f"\nP(both footprints preserved) at measured r (N={N_CELLS}, m={M_A}):")
    if rs_multi:
        ras = np.array([x[0] for x in rs_multi]); rbs = np.array([x[1] for x in rs_multi])
        ps = np.array([Pboth(int(a), int(b)) for a, b in rs_multi])
        print(f"  multi-defect images: n={len(rs_multi)}, median (r_a,r_b)="
              f"({int(np.median(ras))},{int(np.median(rbs))})")
        print(f"  P(both preserved): mean={ps.mean():.4f} median={np.median(ps):.4f} "
              f"frac<0.05={np.mean(ps<0.05):.2f}")
    print("  reference small-r: P(1,1)={:.3f} P(2,2)={:.3f} P(3,3)={:.3f} P(5,5)={:.4f}"
          .format(Pboth(1, 1), Pboth(2, 2), Pboth(3, 3), Pboth(5, 5)))
    med = int(np.median(allr))
    print(f"  at median single-defect r={med}: P(both, r_a=r_b={med})={Pboth(med, med):.4f}")


if __name__ == "__main__":
    main()
