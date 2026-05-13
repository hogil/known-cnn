"""
Pretrained backbone weights checker/downloader (mega_matrix).

기본값은 폐쇄망 안전 모드다. mega_matrix/weights/{model_id}.{pth,safetensors,bin}
파일이 이미 있으면 skip하고, 없으면 실패한다. HuggingFace/timm 다운로드는
인터넷 머신에서 명시적으로 --allow-download 를 준 경우에만 실행한다.

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
import os
import sys
from pathlib import Path

import timm
import torch

WEIGHTS_DIR = Path(__file__).parent / "weights"
WEIGHT_EXTS = (".pth", ".safetensors", ".bin")

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


def find_existing_weight(model_name: str) -> Path | None:
    for ext in WEIGHT_EXTS:
        p = WEIGHTS_DIR / f"{model_name}{ext}"
        if p.exists():
            return p
    return None


def download_one(model_name: str, force: bool = False, allow_download: bool = False) -> str:
    """단일 모델 다운로드. 반환: 'ok' / 'skip' / 'fail'."""
    existing = find_existing_weight(model_name)
    if existing is not None and not force:
        size_mb = existing.stat().st_size / 1e6
        try:
            verify_weight_file(existing)
        except Exception as e:
            print(f"  invalid {existing.name} ({type(e).__name__}: {e})", file=sys.stderr)
            return "fail"
        print(f"  skip   {existing.name} ({size_mb:.0f} MB, local)")
        return "skip"
    if existing is not None and force and not allow_download:
        print(f"  skip   {existing.name} (--force ignored without --allow-download)")
        return "skip"
    if not allow_download:
        print(f"  FAIL   {model_name}: missing local weights under {WEIGHTS_DIR}; "
              f"download disabled", file=sys.stderr)
        return "fail"

    out_path = WEIGHTS_DIR / f"{model_name}.pth"
    print(f"  download {model_name} ...")
    try:
        m = timm.create_model(model_name, pretrained=True)
        torch.save(m.state_dict(), str(out_path))
        verify_weight_file(out_path)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  saved  {out_path.name} ({size_mb:.0f} MB, verified)")
        return "ok"
    except Exception as e:
        print(f"  FAIL   {model_name}: {type(e).__name__}: {e}", file=sys.stderr)
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
        print(f"Configured MODELS ({len(MODELS)}):")
        for n in MODELS:
            print(f"  {n}")
        return

    targets = MODELS
    if args.only:
        targets = [m for m in MODELS if args.only in m]
        print(f"Filter '{args.only}' -> {len(targets)}/{len(MODELS)} models")
        if not targets:
            print("No matches.", file=sys.stderr)
            sys.exit(1)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    print(f"Targets: {len(targets)} models")
    print(f"Weights: {WEIGHTS_DIR}/{{model_name}}.{{pth,safetensors,bin}}")
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
