"""Multi-pattern wafer 합성 generator (Stage 3 of multi-label ablation).

기존 _sample_gen.py 의 render() 가 (1 distribution × 1 object) single-label 합성이라면,
본 generator 는 (1+ distribution × 1+ object) multi-label 합성 — chip 단위 OR 합성.

핵심 idea:
- 각 distribution 의 heatmap mask 를 OR 합성
- combined mask 의 chip 마다 random object 할당 (객체 list 에서)
- 같은 render 로직 (canvas / grade modulation / bin / border / palette / JSON) reuse
- multi-label GT 보존:
  * 파일명: __d-<dist1>+<dist2>__o-<obj1>+<obj2>
  * JSON: multi_labels { distributions, objects }
  * JSON.chips[].true_distribution, true_object (chip 단위 GT)

Mix 비율 (default 70/20/10):
- 70%: 1 distribution + 1 object  (single-label 동등)
- 20%: 2 distributions + 1-2 objects
- 10%: 3 distributions + 1-3 objects

특수 wafer-class 제외 (Normal/Starburst/CommaCluster/Thick-Edge_invalid_main):
- 본 ablation 의 multi-label 평가는 7 distribution × 4 object 만 (28 wafer-class).
- invalid_main 은 wafer 전체가 invalid 되는 special — multi 합성 부적합.

출력:
- <project>/data/wm-811k/unknown_multi/<basename>.png
- <project>/data/positions/unknown_multi/<basename>.json
- <project>/data/wm-811k/unknown_multi/_manifest.csv (basename × distributions × objects × n_dist × n_obj)
"""
import argparse, csv, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from ._fq_metadata import add_synthetic_fq_to_json
    from ._sample_gen import (
        GRID, CHIP, SIZE, PALETTE, KEY_TO_INDEX, IDX_BG, IDX_TEXT, IDX_BORDER_INV,
        BIN_TO_BORDER_IDX, FONT_BIG, CUM_BASE, CUM_DEFECT_BG, CUM_EDGE,
        ALPHA_FNS, OBJECT_DISTS, _wafer_inside_mask, select_distribution_chips,
        select_random_invalid, assign_defect_bin, assign_invalid_bin, rand_prefix,
        LT_OPTIONS, TM_OPTIONS, _GPU, _DEVICE,
    )
except ImportError:
    from _fq_metadata import add_synthetic_fq_to_json
    from _sample_gen import (
        GRID, CHIP, SIZE, PALETTE, KEY_TO_INDEX, IDX_BG, IDX_TEXT, IDX_BORDER_INV,
        BIN_TO_BORDER_IDX, FONT_BIG, CUM_BASE, CUM_DEFECT_BG, CUM_EDGE,
        ALPHA_FNS, OBJECT_DISTS, _wafer_inside_mask, select_distribution_chips,
        select_random_invalid, assign_defect_bin, assign_invalid_bin, rand_prefix,
        LT_OPTIONS, TM_OPTIONS, _GPU, _DEVICE,
    )

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DISTRIBUTIONS = ["Center", "Donut", "Edge-Ring", "Edge-Bottom", "Edge-Top", "Full", "Thick-Edge"]
OBJECTS = ["bank_boundary", "fork", "scratch", "scratch_rot"]                            # round 26: particle_blast→fork, scratch_21deg→scratch_rot

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("WM811K_ROOT", os.path.join(PROJ_ROOT, "data", "wm-811k"))
POSITIONS_ROOT = os.environ.get("POSITIONS_ROOT", os.path.join(PROJ_ROOT, "data", "positions"))

PNG_OUT_DIR = os.environ.get("WAFER_MULTI_PNG_OUT_DIR", os.path.join(DATA_ROOT, "unknown_multi"))
JSON_OUT_DIR = os.environ.get("WAFER_MULTI_JSON_OUT_DIR", os.path.join(POSITIONS_ROOT, "unknown_multi"))
MANIFEST_PATH = os.path.join(PNG_OUT_DIR, "_manifest.csv")


def random_multi_label_combo(rng: np.random.Generator, mix: Tuple[float, float, float]
                              ) -> Tuple[List[str], List[str]]:
    """Sample (distributions, objects) per mix ratio.

    mix = (single, double, triple) — sum to 1.0.
    """
    p = float(rng.random())
    cum = [mix[0], mix[0] + mix[1]]
    if p < cum[0]:                                                                       # single
        d_n, o_n = 1, 1
    elif p < cum[1]:                                                                     # 2-mix
        d_n = 2
        o_n = int(rng.integers(1, 3))                                                    # 1 or 2
    else:                                                                                 # 3-mix
        d_n = 3
        o_n = int(rng.integers(1, 4))                                                    # 1, 2, or 3

    dist_idx = rng.choice(len(DISTRIBUTIONS), size=d_n, replace=False)
    obj_idx = rng.choice(len(OBJECTS), size=o_n, replace=False)
    distributions = sorted([DISTRIBUTIONS[int(i)] for i in dist_idx])                    # sorted for filename consistency
    objects = sorted([OBJECTS[int(i)] for i in obj_idx])
    return distributions, objects


def render_multi(distributions: List[str], objects: List[str], seed: int,
                 png_dir: str = None, json_dir: str = None):
    """Render one multi-pattern wafer.

    Returns (png_path, json_path, n_defect, n_invalid, kind, yld, syp).
    """
    png_dir = png_dir or PNG_OUT_DIR
    json_dir = json_dir or JSON_OUT_DIR
    rng = np.random.default_rng(seed)
    inside = _wafer_inside_mask()

    # 1) Choose kind (다중 distribution 의 'invalid_main' 은 multi 에서 제외이므로 random P/C)
    kind = "00P" if rng.random() < 0.5 else "00C"

    # 2) 각 distribution 별 chip mask + 어느 distribution 에서 왔는지 GT 기록
    dist_masks = {}
    chip_dist_pref = {}                                                                  # (gy,gx) → first distribution it came from
    for d in distributions:
        mask_d = select_distribution_chips(d, rng, inside)
        dist_masks[d] = mask_d
        ys, xs = np.where(mask_d)
        for gy, gx in zip(ys, xs):
            key = (int(gy), int(gx))
            if key not in chip_dist_pref:                                                # first-come distribution wins (deterministic OR)
                chip_dist_pref[key] = d

    # combined defect mask = OR
    defect_mask = np.zeros((GRID, GRID), dtype=bool)
    for m in dist_masks.values():
        defect_mask |= m

    # 3) chip 마다 random object 할당 (objects list 에서)
    chip_obj_map = {}
    n_obj = len(objects)
    for key in chip_dist_pref:
        chip_obj_map[key] = objects[int(rng.integers(0, n_obj))]

    # 4) Random invalid chips (배경 noise)
    invalid_inside_mask = select_random_invalid(rng, defect_mask, inside, n=15)
    invalid_mask = invalid_inside_mask & ~defect_mask

    # 5) Per-chip meta (bin + obj + GT)
    chip_meta = {}
    for key, obj in chip_obj_map.items():
        gy, gx = key
        chip_meta[key] = {
            "kind": "defect", "obj": obj,
            "bin": assign_defect_bin(kind, rng), "inside": True,
            "true_distribution": chip_dist_pref[key], "true_object": obj,
        }
    for gy in range(GRID):
        for gx in range(GRID):
            if invalid_mask[gy, gx]:
                chip_meta[(int(gy), int(gx))] = {
                    "kind": "invalid", "obj": None,
                    "bin": assign_invalid_bin(kind, rng), "inside": True,
                    "true_distribution": None, "true_object": None,
                }

    # 6) Canvas (baseline grades + outside-wafer = bg color)
    if _GPU:
        import torch
        g_t = torch.Generator(device=_DEVICE).manual_seed(seed)
        u_t = torch.rand((SIZE, SIZE), generator=g_t, device=_DEVICE)
        cum_base_t = torch.tensor(CUM_BASE, device=_DEVICE, dtype=torch.float32)
        canvas_t = torch.searchsorted(cum_base_t, u_t).to(torch.uint8)
        canvas = canvas_t.cpu().numpy()
        del u_t, canvas_t
    else:
        u = rng.random((SIZE, SIZE))
        canvas = np.searchsorted(CUM_BASE, u).astype(np.uint8); del u
    inside_pix = np.repeat(np.repeat(inside, CHIP, axis=0), CHIP, axis=1)
    canvas[~inside_pix] = IDX_BG

    # 7) Defect chips: alpha modulation per chip with assigned object
    for (gy, gx), meta in chip_meta.items():
        if meta["kind"] != "defect":
            continue
        obj = meta["obj"]
        alpha = ALPHA_FNS[obj](rng)
        cum_obj = np.cumsum(OBJECT_DISTS[obj])
        center_power = {"bank_boundary": 6, "fork": 4, "scratch": 5,
                        "scratch_rot": 8}.get(obj, 4)                                    # round 26 keys
        w_bg = np.clip((0.40 - alpha) / 0.40, 0.0, 1.0).astype(np.float32)
        t_raw = np.clip((alpha - 0.40) / 0.60, 0.0, 1.0).astype(np.float32)
        w_center = (t_raw ** center_power).astype(np.float32)
        w_edge = np.clip(1.0 - w_bg - w_center, 0.0, 1.0).astype(np.float32)
        cum_mixed = (w_bg[..., None] * CUM_DEFECT_BG[None, None, :] +
                     w_edge[..., None] * CUM_EDGE[None, None, :] +
                     w_center[..., None] * cum_obj[None, None, :])
        uu = rng.random((CHIP, CHIP))
        grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
        y0, x0 = gy * CHIP, gx * CHIP
        canvas[y0:y0 + CHIP, x0:x0 + CHIP] = grades

    # 8) Invalid chips
    for (gy, gx), meta in chip_meta.items():
        if meta["kind"].startswith("invalid"):
            y0, x0 = gy * CHIP, gx * CHIP
            canvas[y0:y0 + CHIP, x0:x0 + CHIP] = 31

    # 9) Borders
    IDX_BORDER_NORMAL = KEY_TO_INDEX["border"]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]:
                continue
            y0, x0 = gy * CHIP, gx * CHIP
            y1, x1 = y0 + CHIP, x0 + CHIP
            meta = chip_meta.get((gy, gx))
            if meta is None:
                b, c = 1, IDX_BORDER_NORMAL
            elif meta["kind"] == "defect":
                b, c = 2, BIN_TO_BORDER_IDX.get(meta["bin"], KEY_TO_INDEX["border_etc"])
            else:
                b, c = 2, IDX_BORDER_INV
            canvas[y0:y0 + b, x0:x1] = c
            canvas[y1 - b:y1, x0:x1] = c
            canvas[y0:y1, x0:x0 + b] = c
            canvas[y0:y1, x1 - b:x1] = c

    # 10) Build palette image + draw bin number text on INVALID chips only
    img = Image.frombytes("P", (SIZE, SIZE), canvas.tobytes())
    img.putpalette(PALETTE)
    draw = ImageDraw.Draw(img)
    for (gy, gx), meta in chip_meta.items():
        if meta["kind"] != "invalid":
            continue
        text = str(meta["bin"])
        y0, x0 = gy * CHIP, gx * CHIP
        cx_px, cy_px = x0 + CHIP / 2, y0 + CHIP / 2
        try:
            bbox = draw.textbbox((0, 0), text, font=FONT_BIG)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ty = cy_px - th / 2 - bbox[1] if isinstance(FONT_BIG, ImageFont.FreeTypeFont) else cy_px - th / 2
        except Exception:
            tw = th = 40; ty = cy_px - th / 2
        draw.text((cx_px - tw / 2, ty), text, fill=IDX_TEXT, font=FONT_BIG)

    # 11) Yield / Sys / TD / LT
    sys_bins = ({285, 286, 287, 288, 290, 291} if kind == "00P"
                else {300, 385, 386, 388, 389, 390})
    netd = int(inside.sum())
    gd_count = sum(1 for m in chip_meta.values()
                   if m["inside"] and m["bin"] is not None and m["bin"] < 200)
    inside_chip_meta_count = sum(1 for m in chip_meta.values() if m["inside"])
    normal_inside = netd - inside_chip_meta_count
    gd_count += normal_inside
    sys_count = sum(1 for m in chip_meta.values() if m["inside"] and m["bin"] in sys_bins)
    yld = 100.0 * gd_count / max(1, netd)
    syp = 100.0 * sys_count / max(1, netd)
    TD = LT_OPTIONS[int(rng.integers(0, len(LT_OPTIONS)))]
    LT = TM_OPTIONS[int(rng.integers(0, len(TM_OPTIONS)))]

    # 12) Save PNG + JSON (multi-label suffix)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    prefix = rand_prefix(rng)
    w_idx = int(rng.integers(1, 25))
    base_core = f"{prefix}_{kind}_{w_idx:02d}_20260501_010000_{yld:.1f}_{syp:.0f}_{TD}_{LT}"
    dist_tag = "+".join(distributions)                                                   # already sorted
    obj_tag = "+".join(objects)
    base = f"{base_core}__d-{dist_tag}__o-{obj_tag}"
    png_path = os.path.join(png_dir, base + ".png")
    json_path = os.path.join(json_dir, base + ".json")
    img.save(png_path, optimize=False, compress_level=1)

    # 13) Generate matching positions JSON (with multi_labels + chip-level GT)
    chips_list = []
    norm_rng = np.random.default_rng(seed + 99999)
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]:
                continue
            key = (int(gy), int(gx))
            meta = chip_meta.get(key)
            if meta is not None and meta["bin"] is not None:
                bin_val = meta["bin"]
                true_dist = meta.get("true_distribution")
                true_obj = meta.get("true_object")
            else:
                bin_val = int(norm_rng.integers(1, 200))
                true_dist = None
                true_obj = None
            x0, y0 = gx * CHIP, gy * CHIP
            x1, y1 = x0 + CHIP, y0 + CHIP
            x_cal = gx - (GRID // 2 - 1)
            y_cal = gy - (GRID // 2)
            chip_entry = {
                "x_abs": gx, "y_abs": gy, "b": str(bin_val),
                "x_cal": x_cal, "y_cal": y_cal,
                "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "quad": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]},
            }
            if true_dist is not None:
                chip_entry["true_distribution"] = true_dist
            if true_obj is not None:
                chip_entry["true_object"] = true_obj
            chips_list.append(chip_entry)

    xs_edges = [k * CHIP for k in range(GRID + 1)]
    ys_edges = [k * CHIP for k in range(GRID + 1)]
    cls_label = f"multi_d{len(distributions)}_o{len(objects)}"                           # for FQ metadata grouping
    json_obj = {
        "bucket_b_key": "", "root": prefix, "step": kind, "wafer": f"W{w_idx:02d}",
        "stime": "20260501_010000", "partid": "", "tester": TD, "device": LT, "pgm": "",
        "netd": int(inside.sum()), "gd": int(gd_count),
        "yield": f"{yld:.2f}", "sys": f"{syp:.2f}",
        "tm": LT, "lt": TD,
        "multi_labels": {"distributions": list(distributions), "objects": list(objects)},
        "coord": {
            "rot_code": 5,
            "x_min_abs": 0, "y_min_abs": 0, "x_max_abs": GRID - 1, "y_max_abs": GRID - 1,
            "tiles_w_rot": GRID, "tiles_h_rot": GRID,
            "grid_edges": {"xs": xs_edges, "ys": ys_edges},
            "canvas": {"width": SIZE, "height": SIZE},
            "scale": {"sx": 1.0, "sy": 1.0},
            "border": 1, "defect_border": 2,
            "center_rule": {"even_x_zero": "left", "even_y_zero": "down"},
        },
        "chips": chips_list,
    }
    add_synthetic_fq_to_json(json_obj, cls_label, base)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, separators=(',', ':'))

    return (png_path, json_path, int(defect_mask.sum()),
            int(invalid_inside_mask.sum()), kind, yld, syp,
            distributions, objects)


def _worker(args):
    seed, distributions, objects, png_dir, json_dir = args
    try:
        out = render_multi(distributions, objects, seed, png_dir=png_dir, json_dir=json_dir)
        return (seed, distributions, objects, True, None)
    except Exception as e:
        import traceback
        return (seed, distributions, objects, False, f"{e}\n{traceback.format_exc()}")


def build_tasks(n: int, mix: Tuple[float, float, float], seed_base: int,
                png_dir: str, json_dir: str):
    rng = np.random.default_rng(seed_base)
    tasks = []
    for i in range(n):
        distributions, objects = random_multi_label_combo(rng, mix)
        tasks.append((seed_base + i, distributions, objects, png_dir, json_dir))
    return tasks


def write_manifest(records, manifest_path):
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["basename", "distributions", "objects", "n_dist", "n_obj"])
        for r in records:
            w.writerow([r["basename"], "+".join(r["distributions"]),
                        "+".join(r["objects"]), len(r["distributions"]),
                        len(r["objects"])])


def parse_mix(s: str) -> Tuple[float, float, float]:
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError(f"--mix expects 3 values, got: {s}")
    vals = tuple(float(x) for x in parts)
    if abs(sum(vals) - 1.0) > 1e-6:
        raise ValueError(f"--mix must sum to 1.0, got: {sum(vals)}")
    return vals


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000, help="total number of wafers to generate")
    p.add_argument("--mix", type=str, default="0.7,0.2,0.1",
                   help="(single, 2-mix, 3-mix) ratio, sum=1.0")
    p.add_argument("--workers", type=int, default=4, help="parallel workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-png", type=str, default=None,
                   help="override PNG_OUT_DIR (default <project>/data/wm-811k/unknown_multi)")
    p.add_argument("--output-json", type=str, default=None,
                   help="override JSON_OUT_DIR (default <project>/data/positions/unknown_multi)")
    p.add_argument("--manifest", type=str, default=None,
                   help="override manifest path (default <output-png>/_manifest.csv)")
    args = p.parse_args()

    png_dir = args.output_png or PNG_OUT_DIR
    json_dir = args.output_json or JSON_OUT_DIR
    manifest_path = args.manifest or os.path.join(png_dir, "_manifest.csv")

    mix = parse_mix(args.mix)
    tasks = build_tasks(args.n, mix, args.seed, png_dir, json_dir)

    n_total = len(tasks)
    n_single = sum(1 for _, d, o, _, _ in tasks if len(d) == 1)
    n_double = sum(1 for _, d, o, _, _ in tasks if len(d) == 2)
    n_triple = sum(1 for _, d, o, _, _ in tasks if len(d) == 3)
    print(f"Total tasks: {n_total} (single={n_single}, 2-mix={n_double}, 3-mix={n_triple})")
    print(f"PNG → {png_dir}")
    print(f"JSON → {json_dir}")
    print(f"Workers: {args.workers}")

    t0 = time.time()
    done = ok = fail = 0
    last_log = t0
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        for seed, distributions, objects, success, err in exe.map(_worker, tasks, chunksize=1):
            done += 1
            if success:
                ok += 1
                # basename reconstruction (deterministic by seed+combo, but easier to query later)
                # we simply collect from the actual filesystem after processing
                records.append({
                    "seed": seed, "distributions": distributions, "objects": objects,
                    "basename": "",                                                       # filled below
                })
            else:
                fail += 1
                print(f"  FAIL seed={seed} d={distributions} o={objects}: {err[:200]}")
            now = time.time()
            if now - last_log > 30 or done == n_total:
                elapsed = now - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{done:5d}/{n_total}] ok={ok} fail={fail} | "
                      f"rate={rate:.2f}/s | elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m")
                last_log = now

    # Manifest from filesystem (basename = filename minus .png)
    print("Building manifest from filesystem...")
    fs_records = []
    for fn in sorted(os.listdir(png_dir)):
        if not fn.endswith(".png"):
            continue
        base = fn[:-4]
        # parse __d-...__o-...
        try:
            idx_d = base.rindex("__d-")
            idx_o = base.rindex("__o-")
        except ValueError:
            continue
        dist_str = base[idx_d + 4:idx_o]
        obj_str = base[idx_o + 4:]
        fs_records.append({
            "basename": base,
            "distributions": dist_str.split("+"),
            "objects": obj_str.split("+"),
        })
    write_manifest(fs_records, manifest_path)
    print(f"Manifest → {manifest_path}  ({len(fs_records)} rows)")
    print(f"Done in {(time.time()-t0)/60:.1f}m. ok={ok} fail={fail}")
