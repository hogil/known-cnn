"""
Mega matrix — backbone weight downloader for closed-network servers.

Downloads timm backbone weights from HuggingFace Hub to mega_matrix/weights/
so the server can run training without internet.

Usage (on an internet-connected machine):
    python mega_matrix/download_weights.py
    # → produces mega_matrix/weights/convnextv2_base.fcmae_ft_in22k_in1k_384.safetensors

Then copy mega_matrix/weights/ to the server, and run.sh / run_ddp.sh
auto-passes --backbone-timm-weights to the trainer if the file exists.

Manual override:
    python -m chip_multilabel._train_chip_variant \\
        --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 \\
        --backbone-timm-weights mega_matrix/weights/convnextv2_base.fcmae_ft_in22k_in1k_384.safetensors \\
        ... (other args)

Add additional backbones by editing BACKBONES below.
"""
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# (timm_model_name, hf_repo_id, candidate filenames in priority order)
BACKBONES = [
    (
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "timm/convnextv2_base.fcmae_ft_in22k_in1k_384",
        ["model.safetensors", "pytorch_model.bin"],
    ),
    # Future backbones — uncomment as needed
    # ("vit_base_patch16_384.augreg_in21k_ft_in1k",
    #  "timm/vit_base_patch16_384.augreg_in21k_ft_in1k",
    #  ["model.safetensors", "pytorch_model.bin"]),
    # ("swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    #  "timm/swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    #  ["model.safetensors", "pytorch_model.bin"]),
]


def log(msg):
    print(f"[dl] {msg}", flush=True)


def derive_local_filename(timm_name: str, hf_filename: str) -> str:
    ext = Path(hf_filename).suffix  # .safetensors / .bin
    return f"{timm_name}{ext}"


def download_one(timm_name: str, repo_id: str, candidates: list) -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    last_err = None
    for fname in candidates:
        local_name = derive_local_filename(timm_name, fname)
        target = WEIGHTS_DIR / local_name
        if target.exists() and target.stat().st_size > 1_000_000:
            log(f"  {target.name} exists ({target.stat().st_size / 1e6:.1f} MB), skip")
            return target
        log(f"  fetch {repo_id}/{fname} → {target.name}")
        try:
            cached = hf_hub_download(repo_id=repo_id, filename=fname,
                                     cache_dir=str(WEIGHTS_DIR / "_hf_cache"))
            # Copy (not symlink — cache layout breaks if user moves folder later)
            import shutil
            shutil.copy2(cached, target)
            log(f"  saved {target.name} ({target.stat().st_size / 1e6:.1f} MB)")
            return target
        except (EntryNotFoundError, RepositoryNotFoundError) as e:
            last_err = e
            log(f"  not found in repo ({fname}), trying next candidate")
            continue
        except Exception as e:
            last_err = e
            log(f"  ERROR fetching {fname}: {e}")
            continue
    raise RuntimeError(f"All candidate files failed for {repo_id}: {last_err}")


def verify_load(timm_name: str, weight_path: Path) -> bool:
    """Sanity check: load via timm with pretrained_cfg_overlay(file=...) and run dummy forward."""
    import timm
    import torch
    try:
        model = timm.create_model(timm_name, pretrained=True,
                                  pretrained_cfg_overlay=dict(file=str(weight_path)),
                                  num_classes=4)
        model.eval()
        with torch.no_grad():
            # convnextv2_base 384 expects 3x384x384 — use a small batch
            x = torch.randn(1, 3, 384, 384)
            y = model(x)
        log(f"  verify OK: forward shape={tuple(y.shape)}")
        return True
    except Exception as e:
        log(f"  verify FAIL: {e}")
        return False


def main():
    log(f"target dir: {WEIGHTS_DIR}")
    log(f"backbones: {len(BACKBONES)}")
    failed = []
    for timm_name, repo_id, candidates in BACKBONES:
        log(f"=== {timm_name} ===")
        try:
            wpath = download_one(timm_name, repo_id, candidates)
        except Exception as e:
            log(f"  DOWNLOAD FAILED: {e}")
            failed.append(timm_name)
            continue
        if not verify_load(timm_name, wpath):
            failed.append(timm_name)
    log("=== summary ===")
    for timm_name, _, _ in BACKBONES:
        local = list(WEIGHTS_DIR.glob(f"{timm_name}.*"))
        local = [p for p in local if p.suffix in (".safetensors", ".bin", ".pt")]
        size_mb = local[0].stat().st_size / 1e6 if local else 0
        status = "OK" if local and timm_name not in failed else "MISSING"
        log(f"  [{status}] {timm_name}: {local[0].name if local else '-'} ({size_mb:.1f} MB)")
    if failed:
        log(f"FAILED: {failed}")
        sys.exit(1)
    log("DONE - copy mega_matrix/weights/ to the server.")


if __name__ == "__main__":
    main()
