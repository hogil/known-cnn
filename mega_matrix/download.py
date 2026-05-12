"""
Mega matrix - backbone weight downloader for closed-network servers.

Downloads multiple timm backbone weights from HuggingFace Hub, converts each to
PyTorch .pth (torch.save state_dict), and saves to mega_matrix/weights/.

Usage (on an internet-connected machine):
    python mega_matrix/download.py                      # download all in BACKBONES
    python mega_matrix/download.py --only convnextv2    # download names matching substring
    python mega_matrix/download.py --list               # print BACKBONES list and exit

Then scp mega_matrix/weights/ to the server. run.sh / run_ddp.sh auto-detects
mega_matrix/weights/<backbone>.pth and passes --backbone-timm-weights.

Manual override:
    python -m chip_multilabel._train_chip_variant \\
        --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 \\
        --backbone-timm-weights mega_matrix/weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \\
        ... (other args)

Add backbones by appending to BACKBONES. Format:
    (timm_model_name, hf_repo_id, [candidate_filenames_in_repo])

Filenames usually `model.safetensors` first, `pytorch_model.bin` fallback.
"""
import sys
import argparse
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# (timm_model_name, hf_repo_id, candidate filenames in HF repo to try in order)
BACKBONES = [
    # Primary - mega_matrix baseline winner
    ("convnextv2_base.fcmae_ft_in22k_in1k_384",
     "timm/convnextv2_base.fcmae_ft_in22k_in1k_384",
     ["model.safetensors", "pytorch_model.bin"]),

    # Large variant (200M params, val_f1 reference)
    ("convnextv2_large.fcmae_ft_in22k_in1k_384",
     "timm/convnextv2_large.fcmae_ft_in22k_in1k_384",
     ["model.safetensors", "pytorch_model.bin"]),

    # Swin V2 384 (windowed attention, prior iter95/iter98 reference)
    ("swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
     "timm/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
     ["model.safetensors", "pytorch_model.bin"]),

    # ViT base 384 (transformer baseline)
    ("vit_base_patch16_384.augreg_in21k_ft_in1k",
     "timm/vit_base_patch16_384.augreg_in21k_ft_in1k",
     ["model.safetensors", "pytorch_model.bin"]),

    # DeiT3 base 384 (distilled ViT, often strong on small data)
    ("deit3_base_patch16_384.fb_in22k_ft_in1k",
     "timm/deit3_base_patch16_384.fb_in22k_ft_in1k",
     ["model.safetensors", "pytorch_model.bin"]),

    # EfficientNetV2 (smaller, faster - speed-tier reference, 224 input)
    ("efficientnetv2_rw_m.agc_in1k",
     "timm/efficientnetv2_rw_m.agc_in1k",
     ["model.safetensors", "pytorch_model.bin"]),
]


def log(msg):
    print(f"[dl] {msg}", flush=True)


def load_state_dict_any(src_path: Path):
    """Load state_dict from .safetensors / .bin / .pt / .pth."""
    if src_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(src_path))
    import torch
    sd = torch.load(str(src_path), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def get_input_size(timm_name: str) -> int:
    """Derive input H/W from timm pretrained_cfg (no weight load)."""
    import timm
    m = timm.create_model(timm_name, pretrained=False)
    cfg = m.pretrained_cfg or {}
    isz = cfg.get("input_size", (3, 224, 224))
    return int(isz[-1])


def download_and_convert(timm_name: str, repo_id: str, candidates: list) -> Path:
    """Download from HF, convert to .pth via torch.save, return .pth path."""
    import torch
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    target_pth = WEIGHTS_DIR / f"{timm_name}.pth"
    if target_pth.exists() and target_pth.stat().st_size > 1_000_000:
        log(f"  {target_pth.name} exists ({target_pth.stat().st_size / 1e6:.1f} MB), skip")
        return target_pth

    last_err = None
    for fname in candidates:
        log(f"  fetch {repo_id}/{fname}")
        try:
            cached = hf_hub_download(repo_id=repo_id, filename=fname,
                                     cache_dir=str(WEIGHTS_DIR / "_hf_cache"))
        except (EntryNotFoundError, RepositoryNotFoundError) as e:
            last_err = e
            log(f"  not in repo, trying next candidate")
            continue
        except Exception as e:
            last_err = e
            log(f"  ERROR fetching {fname}: {e}")
            continue

        log(f"  convert -> {target_pth.name}")
        sd = load_state_dict_any(Path(cached))
        torch.save(sd, str(target_pth))
        log(f"  saved {target_pth.name} ({target_pth.stat().st_size / 1e6:.1f} MB)")
        return target_pth

    raise RuntimeError(f"All candidate files failed for {repo_id}: {last_err}")


def verify_load(timm_name: str, weight_path: Path) -> bool:
    """Sanity check: load via timm pretrained_cfg_overlay(file=...) + dummy forward."""
    import timm
    import torch
    try:
        input_size = get_input_size(timm_name)
        model = timm.create_model(timm_name, pretrained=True,
                                  pretrained_cfg_overlay=dict(file=str(weight_path)),
                                  num_classes=4)
        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, input_size, input_size)
            y = model(x)
        log(f"  verify OK: input={input_size}x{input_size}, output={tuple(y.shape)}")
        return True
    except Exception as e:
        log(f"  verify FAIL: {e}")
        return False


def cleanup_hf_cache():
    """Remove _hf_cache/ after conversion (the .pth has everything we need)."""
    import shutil
    cache_dir = WEIGHTS_DIR / "_hf_cache"
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            log(f"  cleaned {cache_dir.name}/")
        except Exception as e:
            log(f"  cache cleanup skipped: {e}")


def main():
    ap = argparse.ArgumentParser(description="Download timm backbones for offline use")
    ap.add_argument("--only", type=str, default=None,
                    help="Substring filter - only download backbones whose timm_name contains this")
    ap.add_argument("--list", action="store_true",
                    help="Print BACKBONES list and exit")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep _hf_cache/ after conversion (debug)")
    args = ap.parse_args()

    if args.list:
        log("Configured backbones:")
        for n, r, _ in BACKBONES:
            log(f"  {n:<60s}  ({r})")
        return

    targets = BACKBONES
    if args.only:
        targets = [b for b in BACKBONES if args.only in b[0]]
        log(f"filter '{args.only}' -> {len(targets)}/{len(BACKBONES)} backbones")
        if not targets:
            log("no matches, abort")
            sys.exit(1)

    log(f"target dir: {WEIGHTS_DIR}")
    log(f"backbones: {len(targets)}")
    failed = []
    for timm_name, repo_id, candidates in targets:
        log(f"=== {timm_name} ===")
        try:
            wpath = download_and_convert(timm_name, repo_id, candidates)
        except Exception as e:
            log(f"  DOWNLOAD FAILED: {e}")
            failed.append(timm_name)
            continue
        if not verify_load(timm_name, wpath):
            failed.append(timm_name)

    if not args.no_cleanup:
        cleanup_hf_cache()

    log("=== summary ===")
    for timm_name, _, _ in BACKBONES:
        target = WEIGHTS_DIR / f"{timm_name}.pth"
        size_mb = target.stat().st_size / 1e6 if target.exists() else 0
        status = "OK" if target.exists() and timm_name not in failed else ("FAIL" if timm_name in failed else "skip")
        log(f"  [{status:<4s}] {target.name} ({size_mb:.1f} MB)")
    if failed:
        log(f"FAILED: {failed}")
        sys.exit(1)
    log("DONE - copy mega_matrix/weights/ to the server.")


if __name__ == "__main__":
    main()
