"""Model Soup (Wortsman 2022, arxiv:2203.05482) — uniform weight averaging of ckpts.

Usage:
  python -m chip_multilabel._model_soup \
    --ckpts ckpt1.pth ckpt2.pth ckpt3.pth \
    --out outputs/soup_v1/best_model.pth
"""
import argparse, json, sys
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ckpt_paths = [Path(p) for p in args.ckpts]
    for p in ckpt_paths:
        if not p.exists():
            sys.exit(f"missing ckpt: {p}")

    print(f"[soup] loading {len(ckpt_paths)} ckpts")
    states = []
    template = None
    for i, p in enumerate(ckpt_paths):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        if template is None:
            template = ck
        sd = ck.get("model_state_dict") or ck.get("state_dict") or ck
        if not isinstance(sd, dict):
            sys.exit(f"ckpt {p} has no recognizable state_dict")
        states.append(sd)
        print(f"[soup]  [{i}] {p} keys={len(sd)}")

    # Verify all states have same keys
    ref_keys = set(states[0].keys())
    for i, sd in enumerate(states[1:], start=1):
        if set(sd.keys()) != ref_keys:
            diff = ref_keys.symmetric_difference(set(sd.keys()))
            sys.exit(f"ckpt [{i}] key mismatch, diff={len(diff)}")

    # Uniform weight average — only float tensors averaged, int (e.g. num_batches_tracked) take from first
    print(f"[soup] averaging {len(ref_keys)} tensors uniformly")
    souped = {}
    n = len(states)
    for k in ref_keys:
        t0 = states[0][k]
        if not isinstance(t0, torch.Tensor):
            souped[k] = t0
            continue
        if t0.dtype.is_floating_point:
            acc = torch.zeros_like(t0, dtype=torch.float64)
            for sd in states:
                acc += sd[k].to(torch.float64)
            souped[k] = (acc / n).to(t0.dtype)
        else:
            # int / bool / etc — keep first
            souped[k] = t0.clone()

    # Save as same ckpt shape as template
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve template metadata, replace state_dict
    saved = dict(template)
    if "model_state_dict" in saved:
        saved["model_state_dict"] = souped
    elif "state_dict" in saved:
        saved["state_dict"] = souped
    else:
        saved = souped
    # Add soup metadata
    saved["_soup_meta"] = {
        "n_ckpts": n,
        "ckpts": [str(p) for p in ckpt_paths],
        "method": "uniform_weight_average",
        "ref": "Wortsman et al. 2022 ICML, arxiv:2203.05482",
    }
    torch.save(saved, out_path)
    print(f"[soup] saved -> {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
