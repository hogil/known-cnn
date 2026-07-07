"""V1+V2: VOC natural-RGB confirmation (3 seeds, 20ep) + held-out-pair
compositional protocol. Oracle trains on natural multi images EXCLUDING any
image that contains a held-out class pair; copypaste synthesizes all pairs
from singles. Evaluate on full multi test and on held-out-pair test images.
"""
import os
import argparse
from collections import Counter

import numpy as np

from .datasets.voc import (_ids, _cats, CLS2IDX, build_single_pool,
                           build_single_pool_crops, build_multi)
from .synthesis.voc_arms import synth_copypaste, synth_arm
from .run_voc import train_model_voc, eval_voc


def pair_counts(root, split):
    c = Counter()
    for i in _ids(root, split):
        idxs = sorted(CLS2IDX[x] for x in _cats(root, i))
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                c[frozenset((idxs[a], idxs[b]))] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-holdout-pairs", type=int, default=6)
    ap.add_argument("--per-class-cap", type=int, default=100)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--n-holdout-test", type=int, default=250)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default="E:/data/torchvision")
    args = ap.parse_args()

    # held-out pairs: mid-frequency (train 20-80, test >= 15), deterministic
    tr_c = pair_counts(args.root, "trainval")
    te_c = pair_counts(args.root, "test")
    cands = sorted([p for p in te_c
                    if te_c[p] >= 15 and 20 <= tr_c.get(p, 0) <= 80])
    rng = np.random.default_rng(777)
    hold = set(tuple(sorted(p)) for p in
               [cands[i] for i in rng.permutation(len(cands))[:args.n_holdout_pairs]])
    hold_fs = {frozenset(p) for p in hold}
    print(f"held-out pairs ({len(hold)}): {sorted(hold)}", flush=True)

    natX, natY, _ = build_single_pool(args.root, "trainval", args.per_class_cap, args.size, seed=0)
    crpX, crpY, _ = build_single_pool_crops(args.root, "trainval", args.per_class_cap, args.size, seed=0)
    orX, orY = build_multi(args.root, "trainval", args.n_train, args.size, seed=5,
                           exclude_pairs=hold_fs)
    teX, teY = build_multi(args.root, "test", args.n_test, args.size, seed=1)
    hoX, hoY = build_multi(args.root, "test", args.n_holdout_test, args.size, seed=2,
                           require_pairs=hold_fs)
    print(f"singles {natX.shape[0]} crops {crpX.shape[0]} oracle {orX.shape[0]} "
          f"test {teX.shape[0]} holdout-test {hoX.shape[0]}", flush=True)

    rng2 = np.random.default_rng(0)
    for arm in ["oracle", "copypaste", "single_only"]:
        for seed in args.seeds:
            if arm == "oracle":
                trX, trY = orX, orY
            elif arm == "single_only":
                trX, trY = natX, natY
            else:
                sX, sY = synth_copypaste(natX, natY, crpX, crpY, args.n_train, seed)
                trX = np.concatenate([sX, natX])
                trY = np.concatenate([sY, natY])
            model = train_model_voc(trX, trY, args.epochs, args.bs, args.lr,
                                    args.device, seed)
            idx = rng2.choice(len(trX), size=min(300, len(trX)), replace=False)
            tr = eval_voc(model, trX[idx], (trY[idx] >= 0.5).astype(np.float32),
                          args.bs, args.device)
            ev = eval_voc(model, teX, teY, args.bs, args.device)
            ho = eval_voc(model, hoX, hoY, args.bs, args.device)
            print(f"{arm:12s} s{seed} | TRAIN bitF1={tr['bitF1']:.3f} pos={tr['pos']:.3f} "
                  f"neg={tr['neg']:.3f} | EVAL bitF1={ev['bitF1']:.3f} FAR={ev['FAR']:.3f} "
                  f"mAP={ev['mAP']:.3f} pos={ev['pos']:.3f} neg={ev['neg']:.3f} | "
                  f"HOLDOUT bitF1={ho['bitF1']:.3f} mAP={ho['mAP']:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
