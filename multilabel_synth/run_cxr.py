import argparse
from .datasets.cxr14 import load_split
from .run_condition import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--per-single-cap", type=int, default=400)
    ap.add_argument("--n-multi", type=int, default=3000)
    ap.add_argument("--n-normal", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--arms", nargs="+",
                    default=["oracle", "overlay", "cutmix", "mixup", "single_only"])
    args = ap.parse_args()
    split = load_split(size=args.size, per_single_cap=args.per_single_cap,
                       n_multi=args.n_multi, n_normal=args.n_normal, seed=0)
    sp = split["single"][0].shape
    print(f"CXR loaded: single {sp[0]} / multi {split['multi'][0].shape[0]} / "
          f"normal {split['normal'][0].shape[0]} | per-class {split['n_single_per_class']}",
          flush=True)
    run(split, args.arms, args.seeds, in_ch=1, K=14, n_train=args.n_multi,
        n_syn_normal=2000, neg_target=0.03, epochs=args.epochs, tag="CXR")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
