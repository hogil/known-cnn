"""Side-by-side preview: existing n1000 OOD (left) vs new canvas-crop (right)."""
from pathlib import Path
from PIL import Image

ROOT = Path("E:/data/images/chip_multilabel_v15direct")
OUT = Path("D:/project/known-cnn/_ood_preview")
TILE = 100; GRID = 4  # 4x4 = 16 chip each side

for cls in ("CenterDonut",):
    pngs = sorted((ROOT / cls).glob("*.png"))
    old = pngs[:16]  # n1000 originals
    new = pngs[200:216] if len(pngs) >= 216 else []  # canvas-crop additions
    if not new:
        print(f"{cls}: no new canvas chips (need at least 216 total, have {len(pngs)})")
        continue
    canvas = Image.new("RGB", (TILE * GRID * 2 + 20, TILE * GRID), (200, 200, 200))
    for i, p in enumerate(old):
        img = Image.open(p).convert("RGB").resize((TILE, TILE), Image.NEAREST)
        r, c = divmod(i, GRID)
        canvas.paste(img, (c * TILE, r * TILE))
    for i, p in enumerate(new):
        img = Image.open(p).convert("RGB").resize((TILE, TILE), Image.NEAREST)
        r, c = divmod(i, GRID)
        canvas.paste(img, (TILE * GRID + 20 + c * TILE, r * TILE))
    out = OUT / f"{cls}_compare_old_vs_new.png"
    canvas.save(out)
    print(f"{cls}: saved compare grid -> {out} (left=n1000 first 16, right=canvas-crop new 16)")

print(f"\n[OUT] {OUT.resolve()}")
