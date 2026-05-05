#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production compound predict — walk product/line/date tree, inline obj_id_map build + 3ch wafer predict + chip-level expansion.

Pipeline per wafer:
    1. parse sibling JSON: chip rects (x_abs, y_abs, b, rect) for b ≥ 200
    2. crop chips from wafer PNG → chip CNN inference → obj_id grid (uint8)
    3. compose 3-channel input (R = palette/31 BICUBIC, G = obj_id/n_chip_objects BICUBIC, B = zeros) → ImageNet norm
    4. compound model forward → wafer-level prediction (33-class softmax)
    5. expand to chip rows: each chip in JSON gets one row with
        - wafer_class (compound model prediction, repeated)
        - chip_object_class (from obj_id_map at that gx,gy)
        - chip_x, chip_y, chip_b
        - wafer metadata (basename split + JSON top scalars)

Output:
    result_compound/<product>/<line>/<date>/preds.parquet     (1 row / chip)
    logs_predict_compound/<TS>_<product>_<line>_<date>/        (operational tracking)
"""
# ===================== CONFIG =====================
DEFAULT_MODEL_GLOB         = "logs_compound/{line}/overall/best_model.pth"
DEFAULT_CHIP_MODEL_GLOB    = "logs_chip/{line}/overall/best_model.pth"
DEFAULT_RESULT_ROOT        = "result_compound"
DEFAULT_LOGS_ROOT          = "logs_predict_compound"
DEFAULT_BATCH_SIZE_CHIP    = 128       # chip CNN inference batch (200x200 crops)
DEFAULT_BATCH_SIZE_WAFER   = 8         # compound wafer inference batch (3ch BICUBIC)
DEFAULT_THRESHOLD          = None
DEFAULT_DEVICE             = None
KIND_LABEL                 = "compound"
MIN_DEFECT_BIN             = 200
PALETTE_IDX_NORM           = 31        # palette idx 31 = invalid_fill (sample_gen domain spec)
DEFAULT_BASENAME_SCHEMA    = ["prefix","kind","w_idx","date","time","yld","syp","tester","device"]
DEFAULT_JSON_FIELDS        = ["partid","pgm","wafer","stime","step","yield","sys","tm","lt","netd","gd"]
# ==================================================

import argparse, glob, json, os, sys, time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import timm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


# --------- common helpers ----------
def resolve_glob_latest(glob_pattern: str) -> Optional[str]:
    matches = sorted(glob.glob(glob_pattern))
    return matches[-1] if matches else None


def print_overall_meta(model_path: str, label: str = ""):
    meta_path = Path(model_path).parent / "_overall_meta.json"
    if not meta_path.exists():
        return
    try:
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        v = m.get("val_f1")
        v_str = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        prefix = f"{label} " if label else ""
        print(f"[*]   {prefix}sourced from run='{m.get('best_run')}'  val_f1={v_str}",
              file=sys.stderr)
    except Exception:
        pass


def load_model_with_ema(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes: List[str] = ckpt["classes"]
    img_size: int = int(ckpt.get("img_size", 384))
    backbone: str = ckpt.get("backbone", "convnextv2_base.fcmae_ft_in22k_in1k_384")
    model = timm.create_model(backbone, pretrained=False, num_classes=len(classes))
    sd = ckpt.get("model") or ckpt.get("state_dict") or ckpt
    if "ema_state" in ckpt:
        ema = ckpt["ema_state"]
        sd_compat = {k: ema[k] if k in ema else v for k, v in sd.items()}
        model.load_state_dict(sd_compat, strict=False)
    else:
        model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    return model, classes, img_size


def find_batches(image_root: Path, target: Optional[str] = None) -> List[tuple]:
    if target:
        parts = target.replace("\\", "/").strip("/").split("/")
        if len(parts) != 3:
            raise SystemExit(f"--batch must be <product>/<line>/<date>, got {target!r}")
        leaf = image_root / parts[0] / parts[1] / parts[2]
        if not leaf.is_dir():
            raise SystemExit(f"target batch not found: {leaf}")
        return [(parts[0], parts[1], parts[2], leaf)]
    out = []
    for product_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        for line_dir in sorted(p for p in product_dir.iterdir() if p.is_dir()):
            for date_dir in sorted(p for p in line_dir.iterdir() if p.is_dir()):
                out.append((product_dir.name, line_dir.name, date_dir.name, date_dir))
    return out


def split_basename(basename: str, schema: List[str]) -> Dict[str, str]:
    parts = basename.split("_")
    return {name: (parts[i] if i < len(parts) else "") for i, name in enumerate(schema)}


def read_json_meta(json_path: Path, fields: List[str]) -> Dict:
    if not json_path.exists():
        return {f: "" for f in fields}
    try:
        j = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return {f: "" for f in fields}
    out = {}
    for f in fields:
        v = j.get(f, "")
        if isinstance(v, (list, dict)):
            v = ""
        out[f] = v
    return out


def parse_wafer_json(json_path: Path) -> Tuple[List[dict], int, int]:
    """Returns (chip_entries, grid_w, grid_h). chip entries filtered by b >= MIN_DEFECT_BIN."""
    if not json_path.exists():
        return [], 0, 0
    try:
        j = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return [], 0, 0
    coord = j.get("coord") or {}
    grid_w = int(coord.get("tiles_w_rot") or 0)
    grid_h = int(coord.get("tiles_h_rot") or 0)
    chip_entries = []
    xs = []; ys = []
    for chip in (j.get("chips") or []):
        try:
            b = int(str(chip.get("b", "0")).strip())
        except Exception:
            continue
        if b < MIN_DEFECT_BIN:
            continue
        rect = chip.get("rect") or {}
        try:
            x0 = int(rect["x0"]); y0 = int(rect["y0"])
            x1 = int(rect["x1"]); y1 = int(rect["y1"])
        except Exception:
            continue
        gx = int(chip.get("x_abs", -1))
        gy = int(chip.get("y_abs", -1))
        if gx < 0 or gy < 0:
            continue
        xs.append(gx); ys.append(gy)
        chip_entries.append({"gx": gx, "gy": gy, "b": b, "rect": (x0, y0, x1, y1)})
    if grid_w <= 0 or grid_h <= 0:
        if xs and ys:
            grid_w = max(xs) + 1
            grid_h = max(ys) + 1
        else:
            grid_w = grid_h = 32
    return chip_entries, grid_w, grid_h


def build_obj_id_map(wafer_img: Image.Image, chips: List[dict], grid_w: int, grid_h: int,
                     chip_model: torch.nn.Module, chip_classes: List[str], chip_img_size: int,
                     class_idx_to_obj_id: np.ndarray, batch_size: int,
                     device: torch.device, use_amp: bool) -> Tuple[np.ndarray, Dict[int, int]]:
    """Mirror _build_obj_id_maps inner loop. Return (obj_id_grid, idx_to_chip_obj_class_idx).

    obj_id_grid: (grid_h, grid_w) uint8 — 0 if no chip, otherwise 1..N.
    chip_object_class_at_chip[i] = chip_classes index (0..N-1) for chips[i].
    """
    norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    tfm = transforms.Compose([
        transforms.Resize((chip_img_size, chip_img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), norm,
    ])
    obj_map = np.zeros((grid_h, grid_w), dtype=np.uint8)
    chip_obj_class_idx_per_chip: List[int] = [-1] * len(chips)
    for i in range(0, len(chips), batch_size):
        batch_chips = chips[i:i+batch_size]
        tensors = [tfm(wafer_img.crop(c["rect"]).convert("RGB")) for c in batch_chips]
        tens = torch.stack(tensors, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = chip_model(tens)
                logits = logits.float()
            else:
                logits = chip_model(tens)
            preds = logits.argmax(dim=-1).cpu().numpy()
        for k, c in enumerate(batch_chips):
            cls_idx = int(preds[k])
            obj_id = int(class_idx_to_obj_id[cls_idx])
            if 0 <= c["gy"] < grid_h and 0 <= c["gx"] < grid_w:
                obj_map[c["gy"], c["gx"]] = obj_id
            chip_obj_class_idx_per_chip[i + k] = cls_idx
    return obj_map, chip_obj_class_idx_per_chip


def compose_3ch(wafer_img: Image.Image, obj_map: np.ndarray, n_chip_objects: int, img_size: int) -> torch.Tensor:
    """Mirror cnn_predict_compound.CompoundWaferDataset._load_3ch but with in-memory obj_map."""
    img = wafer_img if wafer_img.mode == "P" else wafer_img.convert("P")
    idx = np.asarray(img, dtype=np.uint8)
    idx_pil = Image.fromarray(idx, mode="L")
    idx_resized = idx_pil.resize((img_size, img_size), Image.BICUBIC)
    r = torch.from_numpy(np.asarray(idx_resized, dtype=np.float32) / float(PALETTE_IDX_NORM)) \
              .clamp_(0.0, 1.0).unsqueeze(0)
    # block_expand: chip-block categorical-preserving upscale (BICUBIC noise 회피)
    from _chipgrid_resize import block_expand_2d
    obj_expanded = block_expand_2d(obj_map.astype(np.uint8), img_size)
    g = torch.from_numpy(obj_expanded.astype(np.float32) / float(n_chip_objects)) \
              .clamp_(0.0, 1.0).unsqueeze(0)
    b = torch.zeros_like(r)
    x = torch.cat([r, g, b], dim=0)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


# --------- multi-label helpers (Stage 7) ----------

DISTRIBUTIONS_FOR_MATCHING = ["Center", "Donut", "Edge-Ring", "Edge-Bottom",
                                "Edge-Top", "Full", "Thick-Edge"]


def load_per_class_thresholds(path: Optional[str], classes: List[str],
                                fallback: float) -> np.ndarray:
    """Load per-class thresholds JSON. Fallback to scalar for missing classes."""
    ths = np.full(len(classes), fallback, dtype=np.float32)
    if not path:
        return ths
    p = Path(path)
    if not p.exists():
        print(f"[!] thresholds-json not found: {path}, using fallback={fallback}",
              file=sys.stderr)
        return ths
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        for i, c in enumerate(classes):
            if c in d:
                ths[i] = float(d[c])
        return ths
    except Exception as e:
        print(f"[!] thresholds-json load failed: {e}", file=sys.stderr)
        return ths


def load_matching_surfaces(surfaces_root: str, method: str
                            ) -> Tuple[Dict[str, np.ndarray], float]:
    """Load Stage 1 surface aggregated by distribution. Returns (per_dist_surface, outlier_th)."""
    root = Path(surfaces_root)
    if not root.exists():
        return {}, 0.001
    per_dist = {d: [] for d in DISTRIBUTIONS_FOR_MATCHING}
    for f in sorted(root.glob(f"*__{method}__n=full*.npy")):
        wc = f.stem.split("__")[0]
        for d in DISTRIBUTIONS_FOR_MATCHING:
            if wc.startswith(f"{d}_") or wc == d:
                arr = np.load(f).astype(np.float32)
                arr = arr / max(arr.sum(), 1e-7)
                per_dist[d].append(arr)
                break
    surfaces = {}
    for d, arrs in per_dist.items():
        if arrs:
            agg = np.mean(arrs, axis=0)
            surfaces[d] = agg / max(agg.sum(), 1e-7)
    if not surfaces:
        return {}, 0.001
    all_vals = np.concatenate([s.flatten() for s in surfaces.values()])
    outlier_th = float(np.percentile(all_vals, 5.0))
    return surfaces, outlier_th


def match_chip(chip_x: int, chip_y: int, wafer_distributions: List[str],
                surfaces: Dict[str, np.ndarray], outlier_th: float
                ) -> Tuple[Optional[str], float, str]:
    scores = {d: float(surfaces[d][chip_y, chip_x])
              for d in wafer_distributions if d in surfaces}
    if not scores:
        return (None, 0.0, "no_surface")
    sorted_items = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_s = sorted_items[0]
    if best_s < outlier_th:
        return (None, best_s, "outlier")
    if len(sorted_items) >= 2 and best_s / max(sorted_items[1][1], 1e-7) < 1.5:
        return (best, best_s, "ambiguous")
    return (best, best_s, "ok")


# --------- Per-batch ----------
def process_batch(product: str, line: str, date: str, leaf: Path,
                  positions_root: Path, result_root: Path, logs_root: Path,
                  args, device: torch.device) -> Dict:
    ts = time.strftime("%y%m%d_%H%M%S")
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    log_dir = logs_root / f"{ts}_{product}_{line}_{date}"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = log_dir / "run.log"

    def log(msg: str):
        line_out = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line_out, flush=True)
        with run_log_path.open("a", encoding="utf-8") as f:
            f.write(line_out + "\n")

    summary: Dict = {
        "kind": KIND_LABEL,
        "product": product, "line": line, "date": date,
        "batch_path": str(leaf),
        "ts": ts, "started_at": started_at,
        "status": "pending",
    }
    result_dir = result_root / product / line / date
    result_parquet = result_dir / "preds.parquet"
    summary["result_parquet"] = str(result_parquet)

    if result_parquet.exists() and not args.overwrite:
        log(f"SKIP (preds.parquet exists): {result_parquet}")
        summary["status"] = "skipped_existing"
        return summary

    # resolve both models
    compound_glob = args.model_glob.format(line=line, product=product)
    chip_glob = args.chip_model_glob.format(line=line, product=product)
    log(f"resolve compound model: {compound_glob}")
    compound_path = resolve_glob_latest(compound_glob)
    log(f"resolve chip model: {chip_glob}")
    chip_path = resolve_glob_latest(chip_glob)
    if not compound_path or not chip_path:
        miss = []
        if not compound_path: miss.append(f"compound={compound_glob}")
        if not chip_path: miss.append(f"chip={chip_glob}")
        log(f"MODEL NOT FOUND: {' '.join(miss)}")
        summary["status"] = "model_missing"
        summary["model_glob_compound"] = compound_glob
        summary["model_glob_chip"] = chip_glob
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
    log(f"compound: {compound_path}")
    print_overall_meta(compound_path, label="compound")
    log(f"chip:     {chip_path}")
    print_overall_meta(chip_path, label="chip")
    summary["model_path_compound"] = compound_path
    summary["model_path_chip"] = chip_path

    pngs = sorted(p for p in leaf.glob("*.png") if p.is_file())
    summary["n_input"] = len(pngs)
    if not pngs:
        log("no PNGs"); summary["status"] = "empty"
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
    log(f"n_input wafer PNG: {len(pngs)}")

    # load both models
    compound_model, compound_classes, compound_img_size = load_model_with_ema(Path(compound_path), device)
    chip_model, chip_classes, chip_img_size = load_model_with_ema(Path(chip_path), device)
    n_chip_objects = len(chip_classes)
    class_idx_to_obj_id = np.arange(1, n_chip_objects + 1, dtype=np.uint8)
    obj_id_to_label = ["none"] + list(chip_classes)
    summary["n_classes_compound"] = len(compound_classes)
    summary["n_classes_chip"] = n_chip_objects
    log(f"compound classes={len(compound_classes)}  chip classes={n_chip_objects}  img_size: compound={compound_img_size} chip={chip_img_size}")

    pos_dir = positions_root / product / line / date
    use_amp = (device.type == "cuda")
    threshold = args.threshold

    # Stage 7 — multi-label setup
    ml_mode = bool(getattr(args, "multi_label_mode", False))
    if ml_mode:
        per_class_th = load_per_class_thresholds(args.thresholds_json,
                                                   compound_classes,
                                                   fallback=threshold or 0.5)
        surfaces, outlier_th = load_matching_surfaces(args.surfaces_root,
                                                        args.matching_method)
        log(f"[multi-label] thresholds: {len(compound_classes)} classes, "
            f"min={float(per_class_th.min()):.3f} max={float(per_class_th.max()):.3f} "
            f"surfaces: {len(surfaces)} dists, outlier_th={outlier_th:.6f}")
        wafer_rows: List[dict] = []
        chip_rows: List[dict] = []
        match_status_counter = Counter()

    rows: List[dict] = []
    n_skipped_no_json = 0
    n_chip_processed = 0
    n_wafer_processed = 0
    n_normal = 0
    t0 = time.time()

    # accumulate compound inputs per batch_size_wafer
    pending_3ch: List[torch.Tensor] = []
    pending_meta: List[dict] = []  # (wb, chip list, json_meta, basename split, chip_obj_class_idx_per_chip, png_path_str)

    def flush_compound():
        if not pending_3ch:
            return
        nonlocal n_wafer_processed, n_chip_processed, n_normal
        x = torch.stack(pending_3ch, dim=0).to(device, non_blocking=True)
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = compound_model(x)
                logits = logits.float()
            else:
                logits = compound_model(x)
            if ml_mode:
                sigmoid_probs = torch.sigmoid(logits).cpu().numpy()                       # (B, C)
                positives_arr = sigmoid_probs > per_class_th[None, :]                     # (B, C) bool
            probs = F.softmax(logits, dim=-1)
            confs, preds = probs.max(dim=-1)
            preds = preds.cpu().numpy(); confs = confs.cpu().numpy()
        for i, meta in enumerate(pending_meta):
            wafer_pi = int(preds[i]); wafer_mp = float(confs[i])
            wafer_is_normal = int(threshold is not None and wafer_mp < threshold)
            if wafer_is_normal: n_normal += 1
            wafer_class = compound_classes[wafer_pi]

            # Stage 7 — multi-label rows
            if ml_mode:
                positive_idxs = np.where(positives_arr[i])[0]
                if len(positive_idxs) == 0:
                    # at least top1
                    positive_idxs = np.array([wafer_pi])
                ml_classes = [compound_classes[c] for c in positive_idxs]
                ml_dists_list = []
                for cls in ml_classes:
                    for d in DISTRIBUTIONS_FOR_MATCHING:
                        if cls.startswith(f"{d}_") or cls == d:
                            if d not in ml_dists_list:
                                ml_dists_list.append(d)
                            break
                # wafer-level rows (1 per positive class)
                for c_idx in positive_idxs:
                    wafer_rows.append({
                        "wafer_basename": meta["wb"],
                        "batch_product": product, "batch_line": line, "batch_date": date,
                        "wafer_class": compound_classes[int(c_idx)],
                        "prob": float(sigmoid_probs[i, int(c_idx)]),
                        "threshold": float(per_class_th[int(c_idx)]),
                    })
                # chip-level rows with matching
                for chip_i, c in enumerate(meta["chips"]):
                    cls_idx = meta["chip_obj_class_idx"][chip_i]
                    chip_obj = chip_classes[cls_idx] if cls_idx >= 0 else "none"
                    matched_dist, score, status = (None, 0.0, "no_dists")
                    if ml_dists_list and surfaces:
                        matched_dist, score, status = match_chip(
                            int(c["gx"]), int(c["gy"]),
                            ml_dists_list, surfaces, outlier_th)
                    match_status_counter[status] += 1
                    chip_rows.append({
                        "wafer_basename": meta["wb"],
                        "chip_x": int(c["gx"]), "chip_y": int(c["gy"]), "chip_b": c["b"],
                        "chip_object_class": chip_obj,
                        "matched_distribution": matched_dist or "",
                        "match_score": float(score),
                        "match_status": status,
                    })
                    n_chip_processed += 1
                n_wafer_processed += 1
                continue                                                                  # skip single-label row build below

            for chip_i, c in enumerate(meta["chips"]):
                cls_idx = meta["chip_obj_class_idx"][chip_i]
                if cls_idx < 0:
                    chip_obj_class = "none"
                    chip_obj_id = 0
                else:
                    chip_obj_class = chip_classes[cls_idx]
                    chip_obj_id = int(class_idx_to_obj_id[cls_idx])
                row = {
                    "path": meta["png_path"],
                    "wafer_basename": meta["wb"],
                    "batch_product": product,
                    "batch_line": line,
                    "batch_date": date,
                }
                row.update(meta["wb_split"])
                row.update(meta["json_meta"])
                row["chip_x"] = c["gx"]
                row["chip_y"] = c["gy"]
                row["chip_b"] = c["b"]
                row["wafer_class"] = wafer_class
                row["wafer_class_idx"] = wafer_pi
                row["wafer_max_prob"] = wafer_mp
                row["wafer_is_normal"] = wafer_is_normal
                row["chip_object_class"] = chip_obj_class
                row["chip_object_class_id"] = chip_obj_id
                # multi-label은 row 여러 개로 표현 — prob_<class> 와이드 dummy 컬럼 X
                rows.append(row)
                n_chip_processed += 1
            n_wafer_processed += 1
        pending_3ch.clear()
        pending_meta.clear()

    for png_path in pngs:
        wb = png_path.stem
        json_path = pos_dir / f"{wb}.json"
        chips, grid_w, grid_h = parse_wafer_json(json_path)
        if not chips:
            n_skipped_no_json += 1
            continue
        try:
            wafer_img = Image.open(png_path)
        except Exception as e:
            log(f"open failed {png_path}: {e}")
            continue

        # Stage 1: chip CNN inference → obj_map + per-chip class idx
        obj_map, chip_obj_class_idx = build_obj_id_map(
            wafer_img, chips, grid_w, grid_h,
            chip_model, chip_classes, chip_img_size, class_idx_to_obj_id,
            args.batch_size_chip, device, use_amp,
        )
        # Stage 2: compose 3ch input
        x = compose_3ch(wafer_img, obj_map, n_chip_objects, compound_img_size)
        pending_3ch.append(x)
        pending_meta.append({
            "wb": wb,
            "png_path": str(png_path),
            "chips": chips,
            "chip_obj_class_idx": chip_obj_class_idx,
            "wb_split": split_basename(wb, DEFAULT_BASENAME_SCHEMA),
            "json_meta": read_json_meta(json_path, DEFAULT_JSON_FIELDS),
        })
        if len(pending_3ch) >= args.batch_size_wafer:
            flush_compound()

    flush_compound()

    elapsed = time.time() - t0
    summary["n_skipped_no_json"] = n_skipped_no_json
    summary["n_wafer_processed"] = n_wafer_processed
    summary["n_chip_processed"] = n_chip_processed
    summary["n_normal"] = n_normal
    summary["elapsed_sec"] = round(elapsed, 1)

    if ml_mode:
        if not chip_rows and not wafer_rows:
            log("NO multi-label rows produced")
            summary["status"] = "empty_after_filter"
            with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            return summary
        result_dir.mkdir(parents=True, exist_ok=True)
        wafer_parquet = result_dir / "preds_wafer.parquet"
        chip_parquet = result_dir / "preds_chip.parquet"
        pd.DataFrame(wafer_rows).to_parquet(wafer_parquet, index=False)
        pd.DataFrame(chip_rows).to_parquet(chip_parquet, index=False)
        log(f"wrote {wafer_parquet}  rows={len(wafer_rows)}")
        log(f"wrote {chip_parquet}   rows={len(chip_rows)}")
        summary["result_parquet_wafer"] = str(wafer_parquet)
        summary["result_parquet_chip"] = str(chip_parquet)
        summary["matching_status_summary"] = dict(match_status_counter)
        summary["multi_label_thresholds_path"] = args.thresholds_json
        summary["matching_method"] = args.matching_method
        if args.csv:
            pd.DataFrame(wafer_rows).to_csv(result_dir / "preds_wafer.csv", index=False)
            pd.DataFrame(chip_rows).to_csv(result_dir / "preds_chip.csv", index=False)
        summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        summary["status"] = "ok"
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return summary

    if not rows:
        log("NO chip rows produced")
        summary["status"] = "empty_after_filter"
        with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        return summary

    df = pd.DataFrame(rows)
    result_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(result_parquet, index=False)
    log(f"wrote {result_parquet}  rows={len(rows)}  wafers={n_wafer_processed}  elapsed={elapsed:.1f}s")
    if args.csv:
        csv_path = result_dir / "preds.csv"
        df.to_csv(csv_path, index=False)
        log(f"wrote {csv_path}")
        summary["result_csv"] = str(csv_path)

    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["status"] = "ok"
    with (log_dir / "_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--positions-root", required=True)
    ap.add_argument("--result-root", default=DEFAULT_RESULT_ROOT)
    ap.add_argument("--logs-root", default=DEFAULT_LOGS_ROOT)
    ap.add_argument("--model-glob", default=DEFAULT_MODEL_GLOB,
                    help="compound model glob; default substitutes {line}")
    ap.add_argument("--chip-model-glob", default=DEFAULT_CHIP_MODEL_GLOB,
                    help="chip model glob (for inline obj_id build)")
    ap.add_argument("--batch", default=None,
                    help="single-batch override: <product>/<line>/<date>")
    ap.add_argument("--limit-batches", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--device", default=DEFAULT_DEVICE, choices=[None, "cuda", "cpu"])
    ap.add_argument("--batch-size-chip", type=int, default=DEFAULT_BATCH_SIZE_CHIP)
    ap.add_argument("--batch-size-wafer", type=int, default=DEFAULT_BATCH_SIZE_WAFER)
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    # ----- multi-label option (Stage 7) -----
    ap.add_argument("--multi-label-mode", action="store_true",
                    help="enable multi-label inference (sigmoid + threshold + chip-wafer matching). "
                         "produces preds_wafer.parquet + preds_chip.parquet instead of single preds.parquet")
    ap.add_argument("--thresholds-json", default=None,
                    help="per-class threshold JSON (e.g. {'Donut_scratch': 0.42, ...}). "
                         "if missing, uses --threshold (single value) for all classes.")
    ap.add_argument("--surfaces-root", default="dist_learn/_dist_heatmaps_per_class",
                    help="Stage 1 surface root (for chip-wafer matching)")
    ap.add_argument("--matching-method", default="hybrid",
                    choices=["heatmap", "heatmap_smooth", "gmm", "kde", "hybrid"],
                    help="Stage 1 surface variant to use for chip matching")
    ap.add_argument("--outlier-percentile", type=float, default=5.0,
                    help="surface percentile for outlier threshold (Stage 6 C5+ default)")
    args = ap.parse_args()

    image_root = Path(args.image_root)
    positions_root = Path(args.positions_root)
    result_root = Path(args.result_root)
    logs_root = Path(args.logs_root)

    if not image_root.is_dir():
        raise SystemExit(f"image-root not found: {image_root}")

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    print(f"[*] cnn_predict kind={KIND_LABEL}  device={device}", file=sys.stderr)

    batches = find_batches(image_root, args.batch)
    if args.limit_batches:
        batches = batches[:args.limit_batches]
    print(f"[*] batches: {len(batches)}", file=sys.stderr)

    for product, line, date, leaf in batches:
        try:
            s = process_batch(product, line, date, leaf,
                              positions_root, result_root, logs_root,
                              args, device)
        except Exception as e:
            s = {"product": product, "line": line, "date": date,
                 "status": "error", "error": str(e)}
            print(f"[!] batch {product}/{line}/{date} failed: {e}", file=sys.stderr)
        print(f"  [{product}/{line}/{date}] status={s.get('status')} "
              f"wafers={s.get('n_wafer_processed','?')} chips={s.get('n_chip_processed','?')} "
              f"-> {s.get('result_parquet','-')}", file=sys.stderr)

    print(f"[*] DONE — {len(batches)} batches", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
