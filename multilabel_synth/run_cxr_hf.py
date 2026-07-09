"""ChestX-ray14 via HF streaming (BahaaEldin0/NIH-Chest-Xray-14 parquet) —
grayscale = near-signal-ordered, the key test of the refined thesis. Streams
just enough to fill single/multi/normal quotas, caches to npz, runs the
condition-type arm comparison. No 45GB download.
"""
import os
import argparse
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
try:
    os.environ.setdefault("HF_TOKEN", open("D:/project/known-cnn/.hf_token").read().split("=")[1].strip())
except Exception:
    pass

FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
IDX = {f: i for i, f in enumerate(FINDINGS)}
NPZ = "E:/data/cxr14_hf/cxr_split.npz"


def build(size=128, per_single_cap=300, n_multi=2500, n_normal=3500, seed=0):
    if os.path.exists(NPZ):
        d = np.load(NPZ)
        print(f"loaded cache {NPZ}", flush=True)
        return {"single": (d["spX"], d["spY"]), "multi": (d["mX"], d["mY"]),
                "normal": (d["nX"], d["nY"])}
    from datasets import load_dataset
    from PIL import Image
    ds = load_dataset("BahaaEldin0/NIH-Chest-Xray-14", split="train", streaming=True)
    per = {i: [] for i in range(14)}
    mx, my, nx = [], [], []
    def arr(img):
        return (np.asarray(img.convert("L").resize((size, size), Image.BILINEAR),
                np.float32) / 255.0)[None]
    need_single = 14 * per_single_cap
    for e in ds:
        labs = e["label"]
        if not isinstance(labs, list):
            labs = [labs]
        if labs == ["No Finding"]:
            if len(nx) < n_normal:
                nx.append(arr(e["image"]))
        elif len(labs) == 1 and labs[0] in IDX:
            c = IDX[labs[0]]
            if len(per[c]) < per_single_cap:
                per[c].append(arr(e["image"]))
        elif len(labs) >= 2:
            if len(mx) < n_multi:
                y = np.zeros(14, np.float32)
                for l in labs:
                    if l in IDX:
                        y[IDX[l]] = 1.0
                if y.sum() >= 2:
                    mx.append(arr(e["image"])); my.append(y)
        nsingle = sum(len(v) for v in per.values())
        if nsingle >= need_single and len(mx) >= n_multi and len(nx) >= n_normal:
            break
        if (nsingle + len(mx) + len(nx)) % 500 == 0:
            print(f"  streamed single={nsingle} multi={len(mx)} normal={len(nx)}", flush=True)
    spX, spY = [], []
    for c, ims in per.items():
        for im in ims:
            spX.append(im); y = np.zeros(14, np.float32); y[c] = 1.0; spY.append(y)
    spX, spY = np.stack(spX), np.stack(spY)
    mX, mY = np.stack(mx), np.stack(my)
    nX = np.stack(nx); nY = np.zeros((len(nx), 14), np.float32)
    os.makedirs(os.path.dirname(NPZ), exist_ok=True)
    np.savez(NPZ, spX=spX, spY=spY, mX=mX, mY=mY, nX=nX, nY=nY)
    print(f"cached {NPZ}: single {len(spX)} multi {len(mX)} normal {len(nX)}", flush=True)
    return {"single": (spX, spY), "multi": (mX, mY), "normal": (nX, nY)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--arms", nargs="+",
                    default=["oracle", "overlay", "cutmix", "mixup", "single_only"])
    ap.add_argument("--backbone", choices=["small", "resnet18"], default="small")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    split = build()
    from .run_condition import run
    print(f"CXR-HF: single {split['single'][0].shape[0]} / multi "
          f"{split['multi'][0].shape[0]} / normal {split['normal'][0].shape[0]} "
          f"| backbone={args.backbone} device={args.device}", flush=True)
    run(split, args.arms, args.seeds, in_ch=1, K=14, n_train=2500,
        n_syn_normal=2000, neg_target=0.03, epochs=args.epochs, tag="CXR",
        backbone=args.backbone, device=args.device, lr=args.lr)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
