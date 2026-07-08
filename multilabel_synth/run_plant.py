import argparse
from .datasets.plant2021 import load_split, CLASSES
from .run_condition import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--per-single-cap", type=int, default=800)
    ap.add_argument("--n-multi", type=int, default=1200)
    ap.add_argument("--n-normal", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--arms", nargs="+",
                    default=["oracle", "overlay", "cutmix", "mixup", "single_only"])
    args = ap.parse_args()
    split = load_split(size=args.size, per_single_cap=args.per_single_cap,
                       n_multi=args.n_multi, n_normal=args.n_normal, seed=0)
    print(f"Plant loaded: single {split['single'][0].shape[0]} / "
          f"multi {split['multi'][0].shape[0]} / normal {split['normal'][0].shape[0]} | "
          f"per-class {split['n_single_per_class']}", flush=True)
    run(split, args.arms, args.seeds, in_ch=3, K=len(CLASSES), n_train=1200,
        n_syn_normal=1500, neg_target=0.03, epochs=args.epochs, tag="Plant")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
