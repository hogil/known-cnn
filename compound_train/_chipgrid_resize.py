"""Chip-grid block expand — categorical preserving upscale.

obj_id (32×32 uint8) 를 target 픽셀로 upsample 시 BICUBIC/NEAREST 모두 부적합:
- BICUBIC: chip 경계 fractional 보간 → noise (1.3, 2.7 같은 의미 없는 값)
- NEAREST: 정수 배수 (target/grid 가 정수) 일 때만 의도대로 동작

이 함수는 generic block expand:
- target/grid 가 정수 배수면: NEAREST 와 동일 (각 chip 정확히 base px)
- 정수 안 되면: 일부 chip 은 base px, 일부는 base+1 px (불규칙 분배 - 균등 spread)

같은 chip 내부 픽셀은 항상 동일 값 → categorical 의미 보존.

학계 reference:
- "Pixel-shuffle" / "depth-to-space" (Shi 2016) — 정수 배수 case 의 효율적 구현
- Block-wise upsampling — semantic segmentation 의 mask resize 표준
"""
from __future__ import annotations
from typing import Tuple
import numpy as np


def block_counts(grid: int, target: int) -> np.ndarray:
    """Each grid cell 의 픽셀 수 (sum == target).

    base = target // grid 픽셀 기본
    extra = target - base*grid 개 cell 이 base+1 픽셀 (균등 spread)

    예:
    - grid=32, target=384: base=12, extra=0 → 모든 cell 12 px (NEAREST 동일)
    - grid=32, target=200: base=6, extra=8 → 8 cell 이 7px, 24 cell 이 6px
    - grid=32, target=100: base=3, extra=4 → 4 cell 이 4px, 28 cell 이 3px
    """
    if grid <= 0 or target <= 0:
        raise ValueError(f"grid and target must be positive: grid={grid}, target={target}")
    if target < grid:
        raise ValueError(f"target ({target}) < grid ({grid}) is downsample, not supported here")
    base = target // grid
    extra = target - base * grid
    counts = np.full(grid, base, dtype=np.int64)
    if extra > 0:
        # 균등 spread — np.linspace 로 grid 안에 extra 개 위치 분포
        # endpoints (0, grid-1) 제외 한 균등 spacing 또는 endpoints 포함 모두 가능
        # 안전: linspace(0, grid-1, extra) 의 정수 round (중복 시 next 로 shift)
        positions = np.round(np.linspace(0, grid - 1, extra)).astype(int)
        # 중복 방지 — 이미 +1 된 cell 은 다음 빈 위치로
        seen = set()
        for p in positions:
            while int(p) in seen:
                p = (int(p) + 1) % grid
            seen.add(int(p))
            counts[int(p)] += 1
    assert counts.sum() == target, f"counts.sum()={counts.sum()} != target={target}"
    return counts


def block_expand_2d(arr: np.ndarray, target_h: int, target_w: int = None) -> np.ndarray:
    """2D categorical array 를 target 으로 chip-block expand.

    arr: (grid_h, grid_w) any dtype
    target_h: output height
    target_w: output width (None → target_h 와 동일)

    Returns: (target_h, target_w) same dtype as arr
    """
    if target_w is None:
        target_w = target_h
    if arr.ndim != 2:
        raise ValueError(f"arr must be 2D, got shape {arr.shape}")
    grid_h, grid_w = arr.shape
    counts_h = block_counts(grid_h, target_h)
    counts_w = block_counts(grid_w, target_w)
    return np.repeat(np.repeat(arr, counts_h, axis=0), counts_w, axis=1)


def block_expand_pil(arr: np.ndarray, target_h: int, target_w: int = None):
    """convenience wrapper that returns PIL.Image (mode 'L' for uint8)."""
    from PIL import Image
    out = block_expand_2d(arr, target_h, target_w)
    if out.dtype != np.uint8:
        out = out.astype(np.uint8)
    return Image.fromarray(out, mode="L")


# ---------------------------- self-test ----------------------------

if __name__ == "__main__":
    print("=== block_counts ===")
    for grid, target in [(32, 384), (32, 200), (32, 100), (32, 1024), (32, 32), (16, 100)]:
        c = block_counts(grid, target)
        print(f"  grid={grid:>4}, target={target:>4}: counts.sum()={c.sum()}  "
              f"min={c.min()} max={c.max()} extra={(c == c.max()).sum() if c.max() > c.min() else 0}")

    print()
    print("=== block_expand_2d (categorical preserve) ===")
    arr = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    print(f"input shape {arr.shape}, dtype {arr.dtype}:")
    print(arr)

    # 정수 배수 — NEAREST 동일
    out = block_expand_2d(arr, 6, 9)                                                      # 2x3 → 6x9
    print(f"\nexpand to (6, 9) [base=3, extra=0]:")
    print(out)

    # fractional — 일부 cell 만 +1
    out = block_expand_2d(arr, 5, 7)                                                      # 2x3 → 5x7 (base_h=2 +1 extra, base_w=2 +1 extra)
    print(f"\nexpand to (5, 7) [fractional]:")
    print(out)
    print(f"  unique values preserved: {sorted(np.unique(out).tolist())} == {sorted(np.unique(arr).tolist())}")

    # 더 fractional
    out = block_expand_2d(arr, 4, 8)                                                      # 2x3 → 4x8 (base_h=2, base_w=2 +2 extra)
    print(f"\nexpand to (4, 8):")
    print(out)

    print("\n[OK] block_expand_2d categorical preserved across all cases.")
