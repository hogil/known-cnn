"""Does GENERATIVE / SPATIAL appearance transfer close the land-cover appearance
floor that content-blind synthesis cannot -- where deep-CORAL global feature-moment
alignment FAILED (outputs/multilabel_synth/landcover_unlabeled_align.csv: baseline
~0.836 -> aligned 0.81-0.83, vs oracle ~0.937)?

CORAL matched only a single GLOBAL feature moment; the appearance floor is about
SPATIAL co-occurrence appearance -- the local texture and region-boundary look of
real multi-label tiles. This probe tests two STRONGER, spatially-aware mechanisms
that make content-blind synthetic combos actually LOOK like real multi-label tiles
at the pixel/texture level, using only UNLABELED real multi-label tiles (appearance
only -- labels stay SYNTHETIC, preserving the "no multi-label annotation" spirit;
the only relaxation is unlabeled real-multi IMAGES, identical to the CORAL arm).

ARMS (both: refine ONLY the synthetic combos, keep their SYNTHETIC labels; then
train the SAME resnet18 recipe on refined-combos + real singles; eval on the SAME
held-out REAL multi-label test):

  gen_refine   : PREFERRED generative mechanism. A lightweight residual U-Net
                 refiner trained UNPAIRED (source = content-blind synthetic combos,
                 target = U UNLABELED real multi-label tiles) with a PatchGAN
                 discriminator (LSGAN) that judges LOCAL patches (spatial texture)
                 + a low-frequency L1 content-preservation loss that pins the coarse
                 region layout so the synthetic multi-hot labels stay valid. This is
                 a real image-to-image generator learned from the unlabeled reals.

  pyramid_adain: DETERMINISTIC multi-scale spatial-statistics transfer (robustness
                 control, strictly stronger than CORAL's single global moment):
                 a Laplacian-pyramid texture transfer that matches, band-by-band
                 (every spatial frequency), the per-channel mean+std of each
                 synthetic combo to a randomly drawn real multi-label tile, keeping
                 the synthetic layout positions (labels) intact. No training -> can
                 never be "unstable", so it isolates whether ANY spatial appearance
                 transfer -- not just a possibly-finicky GAN -- can move the floor.

U (amount of UNLABELED real multi used) is swept over {250, 500, 1000}; the U tiles
are the first-U of the SAME nested permutation (rng 1000+seed over the oracle pool,
disjoint from eval) that the CORAL and few-shot arms use -> apples-to-apples "same
amount of extra real data" at each U. U=0 reproduces the content-blind floor in-code.

REFERENCE POINTS (same held-out real multi eval, resnet18, 3 seeds):
  content-blind floor  ~0.836  (in-code U=0 base; matches few-shot synth_plus_k K=0)
  FAILED CORAL arm     ~0.82   (landcover_unlabeled_align.csv)
  oracle               ~0.937  (few-shot real_only_k K=all)

HONESTY / early-kill: lambda_content is fixed by INIT loss-magnitude balance
(--calibrate), NOT by eval mAP, and held constant across all U and seeds. If neither
mechanism cleanly beats the content-blind floor, that is reported plainly -- combined
with the CORAL failure it is strong evidence the appearance floor requires LABELED
co-occurrence (an honest, publishable boundary for the (b) direction). No tuning to
force a win; if the GAN will not converge, pyramid_adain still answers the question.
"""
import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .datasets.dlrsd import prepare_landcover, harvest_mask_templates, DEFAULT_SUBSET
from .synthesis import landcover_ops as ops
from .models.resnet import build_resnet18
from .models.refiner import UNetRefiner, PatchGAN, content_lowfreq
from .metrics import bit_f1, compute_map, far, pos_neg_prob
from .run_operator_match_landcover import train_model, _predict
from .run_fewshot_ksweep_landcover import build_base_synth

DEFAULT_US = [250, 500, 1000]
# lambda_content fixed by init loss-magnitude balance (--calibrate), NOT eval mAP:
# measured init adv(LSGAN)=1.31 vs init content-L1=0.059 -> 22.33 for 1x balance at
# start. Rounded to 22 and held constant across all U/seeds/arms (pre-registered
# from the calibration BEFORE reading the U-sweep). See calibrate_lambda().
DEFAULT_LAM_CONTENT = 22.0

FIELDS = ["dataset", "arm", "U", "backbone", "seed", "n_labeled", "n_unlabeled",
          "transfer_method", "lam_content", "eval_map", "eval_bit_f1", "eval_far",
          "eval_pos_prob", "eval_neg_prob"]


# --------------------------------------------------------------------------- #
# gen_refine: unpaired residual-U-Net + PatchGAN (LSGAN) appearance refiner
# --------------------------------------------------------------------------- #
def train_refiner(srcX, tgtX, epochs, bs, lr, device, seed, lam_content,
                  content_factor=4, base_g=32, base_d=64, log_every=0):
    """Train a residual U-Net G (source=synthetic combos) against a PatchGAN D
    (target=unlabeled real multi tiles), LSGAN + low-freq content-preservation.
    srcX, tgtX: [N,3,H,W] float32 in [0,1] (NO ImageNet norm; G works in [0,1])."""
    torch.manual_seed(seed)
    G = UNetRefiner(base=base_g).to(device)
    D = PatchGAN(base=base_d).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    src_t = torch.from_numpy(srcX)
    tgt_t = torch.from_numpy(tgtX)
    src_ds = TensorDataset(src_t)
    loader = DataLoader(src_ds, batch_size=bs, shuffle=True, drop_last=True)
    g = torch.Generator().manual_seed(20000 + seed)
    nt = tgt_t.size(0)
    for ep in range(epochs):
        G.train(); D.train()
        for (xb,) in loader:
            xb = xb.to(device)
            ti = torch.randint(0, nt, (xb.size(0),), generator=g)
            tb = tgt_t[ti].to(device)
            # ---- D step (LSGAN: real->1, fake->0) ----
            with torch.no_grad():
                fake = G(xb)
            optD.zero_grad()
            d_real = D(tb)
            d_fake = D(fake)
            lossD = 0.5 * (mse(d_real, torch.ones_like(d_real)) +
                           mse(d_fake, torch.zeros_like(d_fake)))
            lossD.backward(); optD.step()
            # ---- G step (fool D + preserve coarse layout) ----
            optG.zero_grad()
            fake = G(xb)
            d_fake = D(fake)
            loss_adv = mse(d_fake, torch.ones_like(d_fake))
            loss_content = l1(content_lowfreq(fake, content_factor),
                              content_lowfreq(xb, content_factor))
            lossG = loss_adv + lam_content * loss_content
            lossG.backward(); optG.step()
        if log_every and (ep + 1) % log_every == 0:
            print(f"    [refiner ep{ep+1:02d}] lossD={float(lossD):.4f} "
                  f"adv={float(loss_adv):.4f} content={float(loss_content):.5f}",
                  flush=True)
    return G


@torch.no_grad()
def apply_refiner(G, X, bs, device):
    """Run G over all combos, returning refined [N,3,H,W] float32 in [0,1]."""
    G.eval()
    out = []
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=bs):
        out.append(G(xb.to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def calibrate_lambda(srcX, tgtX, bs, device, content_factor=4, target=1.0):
    """Report init adversarial-G vs content-L1 magnitude so lam_content is fixed by
    balance, NOT eval mAP. Returns (adv0, content0, suggested_lambda)."""
    torch.manual_seed(0)
    G = UNetRefiner().to(device)
    D = PatchGAN().to(device)
    mse = nn.MSELoss(); l1 = nn.L1Loss()
    xb = torch.from_numpy(srcX[:bs]).to(device)
    with torch.no_grad():
        fake = G(xb)
        d_fake = D(fake)
        adv0 = float(mse(d_fake, torch.ones_like(d_fake)))
        c0 = float(l1(content_lowfreq(fake, content_factor),
                      content_lowfreq(xb, content_factor)))
    lam = target * adv0 / c0 if c0 > 0 else float("nan")
    print(f"[calibrate] init adv(LSGAN)={adv0:.4f} content-L1={c0:.5f} -> "
          f"lam_content for {target}x balance = {lam:.2f}", flush=True)
    return adv0, c0, lam


# --------------------------------------------------------------------------- #
# pyramid_adain: deterministic multi-scale Laplacian texture-statistics transfer
# --------------------------------------------------------------------------- #
def _gauss_pyr(x, levels):
    """Gaussian pyramid via 2x avg-pool. x:[N,3,H,W]. Returns list len=levels+1."""
    pyr = [x]
    for _ in range(levels):
        pyr.append(F.avg_pool2d(pyr[-1], 2))
    return pyr


def _lap_pyr(x, levels):
    """Laplacian pyramid: bands[0..levels-1] + coarsest Gaussian at bands[levels]."""
    gp = _gauss_pyr(x, levels)
    bands = []
    for i in range(levels):
        up = F.interpolate(gp[i + 1], size=gp[i].shape[-2:], mode="bilinear",
                           align_corners=False)
        bands.append(gp[i] - up)
    bands.append(gp[levels])
    return bands


def _match_stats(band_s, band_t, eps=1e-5, keep_structure=True):
    """Match per-image per-channel mean+std of synthetic band to target band.
    keep_structure=True (finer bands): recolor src band's own spatial structure to
    tgt band stats. For the coarsest level this preserves the synthetic LAYOUT
    (region positions -> labels) while recoloring to the real tile's appearance."""
    ms = band_s.mean(dim=(2, 3), keepdim=True)
    ss = band_s.std(dim=(2, 3), keepdim=True)
    mt = band_t.mean(dim=(2, 3), keepdim=True)
    st = band_t.std(dim=(2, 3), keepdim=True)
    return (band_s - ms) / (ss + eps) * st + mt


def pyramid_transfer(srcX, tgtX_pool, seed, device, levels=3, bs=64):
    """For each synthetic combo, draw a random real multi tile and transfer its
    band-wise (multi-scale) mean+std onto the combo's Laplacian pyramid, then
    collapse. Layout positions (labels) preserved; appearance -> real. Batched."""
    rng = np.random.default_rng(30000 + seed)
    N = srcX.shape[0]
    Nt = tgtX_pool.shape[0]
    out = np.empty_like(srcX)
    tgt_t = torch.from_numpy(tgtX_pool).to(device)
    for i0 in range(0, N, bs):
        xb = torch.from_numpy(srcX[i0:i0 + bs]).to(device)
        ti = torch.from_numpy(rng.integers(0, Nt, size=xb.size(0))).to(device)
        tb = tgt_t[ti]
        sb = _lap_pyr(xb, levels)
        tb_p = _lap_pyr(tb, levels)
        rec = _match_stats(sb[levels], tb_p[levels])          # coarsest: recolor
        for l in range(levels - 1, -1, -1):
            rec = F.interpolate(rec, size=sb[l].shape[-2:], mode="bilinear",
                                align_corners=False)
            rec = rec + _match_stats(sb[l], tb_p[l])          # add matched band
        out[i0:i0 + bs] = torch.clamp(rec, 0.0, 1.0).cpu().numpy()
    return out


# --------------------------------------------------------------------------- #
# main sweep
# --------------------------------------------------------------------------- #
def _eval_classifier(trX, trY, evX, eval_Y, epochs, bs, lr, device, seed,
                     n_classes):
    model = train_model(trX, trY, epochs, bs, lr, device, seed, n_classes,
                        pretrained=True)
    P = _predict(model, evX, bs, device)
    pos, neg = pos_neg_prob(P, eval_Y)
    return {
        "eval_map": round(float(compute_map(P, eval_Y)), 4),
        "eval_bit_f1": round(float(bit_f1(P, eval_Y)), 4),
        "eval_far": round(float(far(P, eval_Y)), 4),
        "eval_pos_prob": round(float(pos), 4),
        "eval_neg_prob": round(float(neg), 4),
    }


def run(data, templates, seeds, Us, arms, per_class_single, n_multi, epochs, bs,
        lr, ref_epochs, ref_bs, ref_lr, lam_content, content_factor, pyr_levels,
        device, cell, grid, min_frac, feather_sigma, out_csv, dataset="landcover"):
    (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
     names, meta) = data
    n_classes = len(names)
    canvas = cell * grid
    evX = ops.real_to_chw(eval_imgs)
    oX_all = ops.real_to_chw(oracle_imgs)          # real multi-label TRAIN pool
    No = oX_all.shape[0]
    Us = [u for u in Us if u <= No]
    rows = []
    for seed in seeds:
        baseX, baseY, spX, spY = build_base_synth(
            pool_imgs, pool_labels, templates, per_class_single, n_multi, seed,
            cell, grid, canvas, n_classes, min_frac, feather_sigma)
        n_singles = spX.shape[0]
        srcX = baseX[:baseX.shape[0] - n_singles]  # synthetic combos only (source)
        srcY = baseY[:baseY.shape[0] - n_singles]
        perm = np.random.default_rng(1000 + seed).permutation(No)  # SAME as CORAL
        print(f"[seed {seed}] base_synth n={baseX.shape[0]} "
              f"(combos={srcX.shape[0]}+singles={n_singles}), oracle_pool={No}",
              flush=True)

        # U=0 content-blind floor (in-code, same recipe) -- one row, arm='base'
        m = _eval_classifier(baseX, baseY, evX, eval_Y, epochs, bs, lr, device,
                             seed, n_classes)
        rows.append({"dataset": dataset, "arm": "base", "U": 0,
                     "backbone": "resnet18", "seed": int(seed),
                     "n_labeled": int(baseX.shape[0]), "n_unlabeled": 0,
                     "transfer_method": "none", "lam_content": 0.0, **m})
        print(f"[base    ] U=   0 s{seed} mAP={m['eval_map']:.4f} "
              f"bitF1={m['eval_bit_f1']:.4f} FAR={m['eval_far']:.4f} "
              f"pos={m['eval_pos_prob']:.4f} neg={m['eval_neg_prob']:.4f} "
              f"(content-blind floor)", flush=True)

        for U in Us:
            tgtX = oX_all[perm[:U]]
            for arm in arms:
                if arm == "gen_refine":
                    G = train_refiner(srcX, tgtX, ref_epochs, ref_bs, ref_lr,
                                      device, seed, lam_content, content_factor)
                    ref_combos = apply_refiner(G, srcX, bs, device)
                    method = "unet_patchgan_lsgan"
                    lam = lam_content
                elif arm == "pyramid_adain":
                    ref_combos = pyramid_transfer(srcX, tgtX, seed, device,
                                                  levels=pyr_levels, bs=bs)
                    method = f"laplacian_pyr_l{pyr_levels}_meanstd"
                    lam = 0.0
                else:
                    raise ValueError(arm)
                trX = np.concatenate([ref_combos, spX])
                trY = np.concatenate([srcY, spY])
                m = _eval_classifier(trX, trY, evX, eval_Y, epochs, bs, lr, device,
                                     seed, n_classes)
                rows.append({"dataset": dataset, "arm": arm, "U": int(U),
                             "backbone": "resnet18", "seed": int(seed),
                             "n_labeled": int(trX.shape[0]), "n_unlabeled": int(U),
                             "transfer_method": method, "lam_content": float(lam),
                             **m})
                print(f"[{arm:13s}] U={U:4d} s{seed} n_lab={trX.shape[0]} "
                      f"n_unlab={U} mAP={m['eval_map']:.4f} "
                      f"bitF1={m['eval_bit_f1']:.4f} FAR={m['eval_far']:.4f} "
                      f"pos={m['eval_pos_prob']:.4f} neg={m['eval_neg_prob']:.4f}",
                      flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(rows)
    return rows, No


def _agg(rows, arm, key, U=None):
    vals = [r[key] for r in rows
            if r["arm"] == arm and (U is None or r["U"] == U)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def _load_map_csv(path, arm, key_col, key_val, col):
    """Aggregate mean/std of `col` for rows with arm and key_col==key_val."""
    if not os.path.isfile(path):
        return float("nan"), float("nan")
    vals = []
    with open(path, newline="") as f:
        for r in csvmod.DictReader(f):
            if r.get("arm") == arm and int(r.get(key_col, -1)) == key_val:
                vals.append(float(r[col]))
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def print_report(rows, Us, No, fewshot_csv, coral_csv, tol=0.02):
    Us = [u for u in Us if u <= No]
    # in-code content-blind floor (arm='base', U=0)
    floor_m, floor_s = _agg(rows, "base", "eval_map", U=0)
    # oracle from few-shot CSV (real_only_k @ K=all)
    or_m, or_s = _load_map_csv(fewshot_csv, "real_only_k", "K", No, "eval_map")
    if np.isnan(or_m):
        or_m, or_s = _load_map_csv(fewshot_csv, "real_only_k", "K", 1000, "eval_map")
    gap = or_m - floor_m

    print("\n=== GENERATIVE / SPATIAL APPEARANCE TRANSFER vs CONTENT-BLIND FLOOR "
          "vs FAILED CORAL vs ORACLE ===")
    print("    (real DLRSD/UC-Merced land cover, resnet18, mean+/-std over 3 seeds; "
          "labels stay SYNTHETIC, only UNLABELED real-multi images used)")
    print(f"    content-blind floor (in-code base, U=0) mAP={floor_m:.4f}+/-{floor_s:.4f}"
          f"   oracle (real multi ALL) mAP={or_m:.4f}+/-{or_s:.4f}"
          f"   appearance gap={gap:+.4f}")
    print("| arm           | U    | eval mAP           | gap closed | beats floor? "
          "| CORAL mAP (same U) | vs CORAL       |")
    print("|---------------|------|--------------------|------------|--------------"
          "|--------------------|----------------|")
    for arm in ("gen_refine", "pyramid_adain"):
        for U in Us:
            am, as_ = _agg(rows, arm, "eval_map", U=U)
            if np.isnan(am):
                continue
            closed = (am - floor_m) / gap if abs(gap) > 1e-9 else float("nan")
            beats = (am - as_) > (floor_m + floor_s)
            cm, cs = _load_map_csv(coral_csv, "unlabeled_align", "U", U, "eval_map")
            if np.isnan(cm):
                vs = "n/a"
            elif (am - as_) > (cm + cs):
                vs = "gen WINS"
            elif (cm - cs) > (am + as_):
                vs = "CORAL WINS"
            else:
                vs = "tie"
            cm_str = f"{cm:.4f} +/- {cs:.4f}" if not np.isnan(cm) else "      n/a       "
            print(f"| {arm:13s} | {U:4d} | {am:.4f} +/- {as_:.4f} | "
                  f"{closed*100:6.1f}%    | {str(beats):5s}        | {cm_str} | "
                  f"{vs:14s} |")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    best = {}
    for arm in ("gen_refine", "pyramid_adain"):
        bm, bs_, bU, any_beats = -1.0, 0.0, None, False
        for U in Us:
            am, as_ = _agg(rows, arm, "eval_map", U=U)
            if np.isnan(am):
                continue
            if (am - as_) > (floor_m + floor_s):
                any_beats = True
            if am > bm:
                bm, bs_, bU = am, as_, U
        best[arm] = (bm, bs_, bU, any_beats)
        print(f"  {arm}: best mAP={bm:.4f}+/-{bs_:.4f} at U={bU} vs floor "
              f"{floor_m:.4f}+/-{floor_s:.4f} -> clean_win_over_floor={any_beats}; "
              f"approaches oracle(within {tol})={bm >= or_m - tol}")

    gen_beats = best["gen_refine"][3]
    pyr_beats = best["pyramid_adain"][3]
    best_arm = max(best, key=lambda a: best[a][0])
    best_m = best[best_arm][0]
    if not gen_beats and not pyr_beats:
        verdict = ("NEGATIVE: NEITHER the generative U-Net+PatchGAN refiner NOR the "
                   "deterministic multi-scale pyramid transfer cleanly beats the "
                   "content-blind floor -> spatial/pixel-level appearance transfer "
                   "from UNLABELED real-multi does NOT close the gap, just as CORAL's "
                   "global-moment alignment did not. Combined, this is strong evidence "
                   "the appearance floor A*(S) requires LABELED co-occurrence (honest "
                   "bound on the (b) direction, consistent with the lower-bound theory).")
    elif best_m >= or_m - tol:
        verdict = (f"STRONG-POSITIVE: {best_arm} reaches mAP={best_m:.4f}, within {tol} "
                   f"of the oracle ({or_m:.4f}) -> spatial appearance transfer from "
                   f"UNLABELED real-multi closes the gap where CORAL failed.")
    else:
        frac = (best_m - floor_m) / gap if abs(gap) > 1e-9 else float("nan")
        verdict = (f"PARTIAL: {best_arm} beats the content-blind floor (best "
                   f"mAP={best_m:.4f}, closes {frac*100:.0f}% of the appearance gap) "
                   f"but does NOT reach the oracle -> spatial transfer helps beyond "
                   f"CORAL yet unlabeled appearance alone is insufficient to fully "
                   f"close the floor.")
    print(f"\nPLAIN VERDICT: {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Us", nargs="+", type=int, default=DEFAULT_US)
    ap.add_argument("--arms", nargs="+", default=["gen_refine", "pyramid_adain"])
    ap.add_argument("--subset", nargs="+", default=DEFAULT_SUBSET)
    ap.add_argument("--per-class-single", type=int, default=140)
    ap.add_argument("--n-multi", type=int, default=2000)
    ap.add_argument("--n-oracle", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--per-class-cap", type=int, default=140)
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--min-side", type=int, default=48)
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--feather-sigma", type=float, default=3.0)
    ap.add_argument("--min-frac", type=float, default=0.01)
    # classifier recipe (identical to CORAL / few-shot)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    # refiner recipe
    ap.add_argument("--ref-epochs", type=int, default=25)
    ap.add_argument("--ref-bs", type=int, default=16)
    ap.add_argument("--ref-lr", type=float, default=2e-4)
    ap.add_argument("--lam-content", type=float, default=DEFAULT_LAM_CONTENT)
    ap.add_argument("--content-factor", type=int, default=4)
    ap.add_argument("--pyr-levels", type=int, default=3)
    ap.add_argument("--calibrate", action="store_true",
                    help="only report init adv vs content magnitude then exit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base", default="E:/data/dlrsd_extracted/DLRSD")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--fewshot-csv",
                    default="outputs/multilabel_synth/landcover_fewshot_ksweep.csv")
    ap.add_argument("--coral-csv",
                    default="outputs/multilabel_synth/landcover_unlabeled_align.csv")
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/landcover_gen_transfer.csv")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu", flush=True)
        args.device = "cpu"

    canvas = args.cell * args.grid
    data = prepare_landcover(
        args.base, subset=args.subset, cell=args.cell, canvas=canvas,
        n_eval=args.n_eval, n_oracle=args.n_oracle, per_class_cap=args.per_class_cap,
        min_side=args.min_side, purity=args.purity, seed=0, cache=args.cache)
    print(f"[meta] {data[-1]}", flush=True)
    templates = harvest_mask_templates(
        args.base, subset=args.subset, canvas=canvas, n_eval=args.n_eval,
        min_frac=args.min_frac, seed=0)

    if args.calibrate:
        (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
         names, meta) = data
        n_classes = len(names)
        baseX, baseY, spX, spY = build_base_synth(
            pool_imgs, pool_labels, templates, args.per_class_single, args.n_multi,
            0, args.cell, args.grid, canvas, n_classes, args.min_frac,
            args.feather_sigma)
        srcX = baseX[:baseX.shape[0] - spX.shape[0]]
        tgtX = ops.real_to_chw(oracle_imgs)
        calibrate_lambda(srcX, tgtX, args.ref_bs, args.device, args.content_factor)
        return

    rows, No = run(data, templates, args.seeds, args.Us, args.arms,
                   args.per_class_single, args.n_multi, args.epochs, args.bs,
                   args.lr, args.ref_epochs, args.ref_bs, args.ref_lr,
                   args.lam_content, args.content_factor, args.pyr_levels,
                   args.device, args.cell, args.grid, args.min_frac,
                   args.feather_sigma, args.out_csv)
    print_report(rows, args.Us, No, args.fewshot_csv, args.coral_csv)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
