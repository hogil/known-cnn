"""
Pretrained backbone weights checker/downloader (mega_matrix).

기본값은 폐쇄망 안전 모드다. 최종 산출물은 항상
mega_matrix/weights/{model_id}.pth 이다. HuggingFace/timm 다운로드는 인터넷
머신에서 명시적으로 --allow-download 를 준 경우에만 실행한다.

- 항상 MODELS 목록 전부 시도. 1개 실패해도 나머지는 계속 진행.
- 이미 있는 파일은 자동 skip (재다운 안 함).
- 학습/추론 코드는 항상 pretrained=False + weights/{model_name}.pth 로드
  (mega_matrix/run.sh 가 자동 --backbone-timm-weights passthrough).
- 가중치 파일은 절대 git 에 올리지 않는다 (.gitignore 처리됨).
- 폐쇄망 서버: 인터넷 머신에서 받아 mega_matrix/weights/ 폴더 통째로 FTP/scp.

Usage:
    python mega_matrix/download.py             # verify/skip only, no network
    python mega_matrix/download.py --allow-download
    python mega_matrix/download.py --allow-download --force
    python mega_matrix/download.py --only X    # 이름에 X 가 포함된 것만
    python mega_matrix/download.py --list      # MODELS 목록만 출력
"""
import argparse
import difflib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Default closed-network mode must be active before importing timm/huggingface helpers.
if "--allow-download" not in sys.argv:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
else:
    # A parent shell may have exported offline flags during server-side runs.
    # Explicit --allow-download means this process is running on an internet box.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)

from huggingface_hub import hf_hub_download
import timm
import torch

WEIGHTS_DIR = Path(__file__).parent / "weights"
HF_SOURCE_FILENAMES = ("model.safetensors", "pytorch_model.bin", "model.bin")

# HF / timm model id (파일명으로 그대로 사용)
MODELS = [
    # mega_matrix baseline winner (paper)
    "convnextv2_base.fcmae_ft_in22k_in1k_384",
    # 비교군 (img-size 384)
    "convnextv2_large.fcmae_ft_in22k_in1k_384",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
    "vit_base_patch16_384.augreg_in21k_ft_in1k",
    "deit3_base_patch16_384.fb_in22k_ft_in1k",
    # 비교군 (img-size 224, smaller/faster)
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "convnextv2_base.fcmae_ft_in22k_in1k",
    "tf_efficientnetv2_s.in21k_ft_in1k",
    "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k",
    "maxvit_tiny_tf_224.in1k",
    "vit_base_patch16_clip_224.laion2b_ft_in12k_in1k",
]


def verify_weight_file(path: Path) -> None:
    """Fail fast if an existing local weight file is clearly unusable."""
    if path.stat().st_size <= 0:
        raise ValueError("empty weight file")
    if path.suffix == ".pth":
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        if not hasattr(state, "keys") or len(state) == 0:
            raise ValueError("not a non-empty state_dict")


def load_weight_state(path: Path):
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file as _load_safetensors
        except ImportError as e:
            raise RuntimeError("safetensors is required to convert .safetensors weights") from e
        return _load_safetensors(str(path), device="cpu")
    return torch.load(str(path), map_location="cpu", weights_only=False)


def convert_to_pth(src: Path, model_name: str) -> Path:
    dst = WEIGHTS_DIR / f"{model_name}.pth"
    if src.resolve() == dst.resolve():
        verify_weight_file(dst)
        return dst
    state = load_weight_state(src)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and hasattr(state[key], "keys"):
                state = state[key]
                break
    if not hasattr(state, "keys") or len(state) == 0:
        raise ValueError(f"not a non-empty state_dict: {src}")
    torch.save(state, str(dst))
    verify_weight_file(dst)
    return dst


def find_existing_weight(model_name: str) -> Path | None:
    p = WEIGHTS_DIR / f"{model_name}.pth"
    return p if p.exists() else None


def validate_timm_models(model_names: list[str]) -> list[str]:
    """Return configured names that are not real timm pretrained model ids."""
    available = set(timm.list_models(pretrained=True))
    return [name for name in model_names if name not in available]


def print_invalid_models(invalid: list[str]) -> None:
    available = sorted(timm.list_models(pretrained=True))
    print("ERROR: invalid timm pretrained model id(s):", file=sys.stderr)
    for name in invalid:
        print(f"  - {name}", file=sys.stderr)
        suggestions = difflib.get_close_matches(name, available, n=5, cutoff=0.55)
        if suggestions:
            print("    close matches:", file=sys.stderr)
            for s in suggestions:
                print(f"      {s}", file=sys.stderr)


def download_hf_weight(model_name: str) -> Path:
    cfg = timm.models.get_pretrained_cfg(model_name)
    if not cfg.hf_hub_id:
        raise RuntimeError("timm pretrained cfg has no hf_hub_id")
    errors = []
    for filename in HF_SOURCE_FILENAMES:
        try:
            cached = Path(hf_hub_download(repo_id=cfg.hf_hub_id, filename=filename))
            return convert_to_pth(cached, model_name)
        except Exception as e:
            errors.append(f"{filename}: {type(e).__name__}: {e}")
    raise RuntimeError("; ".join(errors))


def download_url_weight(model_name: str) -> Path:
    cfg = timm.models.get_pretrained_cfg(model_name)
    if not cfg.url:
        raise RuntimeError("timm pretrained cfg has no direct url")
    suffix = Path(cfg.url.split("?", 1)[0]).suffix.lower()
    if suffix not in (".pth", ".pt", ".bin"):
        raise RuntimeError(f"direct url is {suffix or 'unknown'}; use HF source and convert to .pth")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"{model_name}{suffix}"
        with urllib.request.urlopen(cfg.url, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return convert_to_pth(tmp, model_name)


def download_one(model_name: str, force: bool = False, allow_download: bool = False) -> str:
    """단일 모델 다운로드. 반환: 'ok' / 'skip' / 'fail'."""
    existing = find_existing_weight(model_name)
    if existing is not None and not force:
        size_mb = existing.stat().st_size / 1e6
        try:
            out_path = convert_to_pth(existing, model_name)
        except Exception as e:
            print(f"  invalid {existing.name} ({type(e).__name__}: {e})", file=sys.stderr)
            return "fail"
        if existing.suffix == ".pth":
            print(f"  skip   {existing.name} ({size_mb:.0f} MB, local)")
        else:
            out_mb = out_path.stat().st_size / 1e6
            print(f"  convert {existing.name} -> {out_path.name} ({out_mb:.0f} MB)")
        return "skip"
    if existing is not None and force and not allow_download:
        print(f"  skip   {existing.name} (--force ignored without --allow-download)")
        return "skip"
    if not allow_download:
        print(f"  FAIL   {model_name}: missing local weights under {WEIGHTS_DIR}; "
              f"download disabled", file=sys.stderr)
        return "fail"
    if force:
        p = WEIGHTS_DIR / f"{model_name}.pth"
        if p.exists():
            p.unlink()

    print(f"  download {model_name} ...")
    errors = []
    for label, fn in (("hf", download_hf_weight), ("url", download_url_weight)):
        try:
            out_path = fn(model_name)
            size_mb = out_path.stat().st_size / 1e6
            print(f"  saved  {out_path.name} ({size_mb:.0f} MB, {label})")
            return "ok"
        except Exception as e:
            msg = f"{label}: {type(e).__name__}: {e}"
            errors.append(msg)
            print(f"  warn   {msg}", file=sys.stderr)
    print(f"  FAIL   {model_name}: " + " | ".join(errors), file=sys.stderr)
    return "fail"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--allow-download", action="store_true",
                    help="인터넷 머신에서만 사용. 없거나 --force인 weight를 timm/HF에서 다운로드")
    ap.add_argument("--force", action="store_true",
                    help="--allow-download와 함께 있을 때만 이미 있는 .pth를 덮어쓰기")
    ap.add_argument("--only", type=str, default=None,
                    help="이름에 substring 포함된 모델만")
    ap.add_argument("--list", action="store_true", help="MODELS 목록만 출력")
    args = ap.parse_args()

    if args.list:
        available = set(timm.list_models(pretrained=True))
        print(f"timm: {getattr(timm, '__version__', 'unknown')}")
        print(f"Configured MODELS ({len(MODELS)}):")
        for n in MODELS:
            mark = "OK" if n in available else "MISSING"
            print(f"  [{mark}] {n}")
        invalid = [n for n in MODELS if n not in available]
        if invalid:
            print_invalid_models(invalid)
            sys.exit(2)
        return

    targets = MODELS
    if args.only:
        targets = [m for m in MODELS if args.only in m]
        print(f"Filter '{args.only}' -> {len(targets)}/{len(MODELS)} models")
        if not targets:
            print("No matches.", file=sys.stderr)
            sys.exit(1)

    invalid = validate_timm_models(targets)
    if invalid:
        print_invalid_models(invalid)
        sys.exit(2)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"timm: {getattr(timm, '__version__', 'unknown')}")
    print(f"Targets: {len(targets)} models")
    print(f"Weights: {WEIGHTS_DIR}/{{model_name}}.pth")
    print(f"Network: {'ENABLED (--allow-download)' if args.allow_download else 'DISABLED'}")
    print()

    counts = {"ok": 0, "skip": 0, "fail": 0}
    failures = []
    for i, name in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {name}")
        result = download_one(name, force=args.force, allow_download=args.allow_download)
        counts[result] += 1
        if result == "fail":
            failures.append(name)
        print()

    print("=" * 60)
    print(f"Done: {counts['ok']} downloaded, {counts['skip']} skipped, {counts['fail']} failed")
    if failures:
        print("Failures:")
        for n in failures:
            print(f"  - {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
