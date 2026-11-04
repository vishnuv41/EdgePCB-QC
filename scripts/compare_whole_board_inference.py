"""
Whole-Board Inference Comparison Experiment (Phase 4)
Bharat AI-SoC Challenge 2026-27 | PS5

Compares whole-board inference configurations across:
1. Overlap ratios: 25% (stride 480), 40% (stride 384), 50% (stride 320)
2. Edge-aware padding: ON (BORDER_REFLECT_101) vs OFF (backward clamp)
3. Duplicate detection merging: Class-Specific NMS vs Class-Agnostic NMS
4. Fusion IoU thresholds: 0.30, 0.45, 0.60
5. Evaluates on the exact same untouched whole-board dataset:
   25 whole colour boards from PCB-Defect V2 (boards 68, 80, 81, 82, 349 ground truth defects).
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner, compute_iou

CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
RAW_PCB_DEFECT_DIR = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train"
OUT_JSON = WORKSPACE_ROOT / "benchmarks" / "results" / "whole_board_inference_comparison.json"

CANONICAL_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole_missing_pad"
}

RAW_TO_CANONICAL = {0: 5, 1: 2, 2: 0, 3: 1, 4: 3, 5: 4}


def load_whole_board_dataset():
    """Loads whole-board images and parsed ground-truth annotations."""
    img_dir = RAW_PCB_DEFECT_DIR / "images"
    lbl_dir = RAW_PCB_DEFECT_DIR / "labels"
    ext_boards = ["68", "80", "81", "82"]

    board_data = []
    for img_p in sorted(img_dir.glob("*.jpg")):
        prefix = img_p.name.split("-")[0]
        if prefix not in ext_boards:
            continue

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                raw_c = int(parts[0])
                can_cid = RAW_TO_CANONICAL.get(raw_c, raw_c)

                if len(parts) >= 9:
                    coords = list(map(float, parts[1:9]))
                    xs = [coords[0] * w, coords[2] * w, coords[4] * w, coords[6] * w]
                    ys = [coords[1] * h, coords[3] * h, coords[5] * h, coords[7] * h]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                elif len(parts) >= 5:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                else:
                    continue

                area = (x2 - x1) * (y2 - y1)
                is_small = (area < 1024)

                gt_boxes.append({
                    "cls": can_cid,
                    "name": CANONICAL_CLASSES.get(can_cid, f"class_{can_cid}"),
                    "bbox": [x1, y1, x2, y2],
                    "is_small": is_small
                })

        board_data.append({
            "path": img_p,
            "image": img,
            "width": w,
            "height": h,
            "gt_boxes": gt_boxes
        })

    return board_data


def evaluate_runner_config(runner, board_data):
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    small_gt = 0
    small_tp = 0
    latencies = []
    tile_counts = []

    for b in board_data:
        img = b["image"]
        gt_boxes = b["gt_boxes"]
        total_gt += len(gt_boxes)
        small_gt += sum(1 for g in gt_boxes if g["is_small"])

        tiles, _, _ = runner.slice_image(img)
        tile_counts.append(len(tiles))

        t0 = time.perf_counter()
        fused = runner.inspect_tiled(img)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        # Match at IoU >= 0.50
        matched_gt = set()
        for d in fused:
            pred_cls = [k for k, v in CANONICAL_CLASSES.items() if v == d.get("class")]
            pred_cls_id = pred_cls[0] if pred_cls else -1
            pred_box = d.get("bbox", [])

            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g_idx in matched_gt or g["cls"] != pred_cls_id:
                    continue
                iou = compute_iou(pred_box, g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= 0.50 and best_g_idx >= 0:
                total_tp += 1
                matched_gt.add(best_g_idx)
                if gt_boxes[best_g_idx]["is_small"]:
                    small_tp += 1
            else:
                total_fp += 1

        total_fn += (len(gt_boxes) - len(matched_gt))

    recall = total_tp / total_gt if total_gt > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    small_recall = small_tp / small_gt if small_gt > 0 else 0.0

    return {
        "ground_truth": total_gt,
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "small_defect_recall": round(small_recall, 4),
        "small_defect_gt": small_gt,
        "small_defect_tp": small_tp,
        "mean_tiles_per_board": round(float(np.mean(tile_counts)), 1),
        "mean_latency_ms": round(float(np.mean(latencies)), 2)
    }


def run_whole_board_inference_study():
    print("=" * 75)
    print("EdgePCB-QC: WHOLE-BOARD INFERENCE OPTIMIZATION STUDY (PHASE 4)")
    print("=" * 75)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    board_data = load_whole_board_dataset()
    print(f"[*] Loaded {len(board_data)} whole boards with {sum(len(b['gt_boxes']) for b in board_data)} total ground-truth defects.")

    inspector = CPUInspector(model_path=CHECKPOINT_PATH, conf_threshold=0.25, device=device)

    # Matrix of configurations:
    # 1. Overlap comparison: 25% vs 40% vs 50%
    # 2. Edge-padding: ON vs OFF
    # 3. NMS: Per-class vs Class-agnostic
    # 4. Fusion IoU: 0.30 vs 0.45 vs 0.60
    configurations = [
        {"name": "Config_1_Baseline_25pct_NoPad_PerClass", "overlap": 0.25, "edge_padding": False, "class_agnostic": False, "iou": 0.45},
        {"name": "Config_2_EdgePad_25pct_PerClass", "overlap": 0.25, "edge_padding": True, "class_agnostic": False, "iou": 0.45},
        {"name": "Config_3_EdgePad_40pct_PerClass", "overlap": 0.40, "edge_padding": True, "class_agnostic": False, "iou": 0.45},
        {"name": "Config_4_EdgePad_50pct_PerClass", "overlap": 0.50, "edge_padding": True, "class_agnostic": False, "iou": 0.45},
        {"name": "Config_5_EdgePad_25pct_ClassAgnostic", "overlap": 0.25, "edge_padding": True, "class_agnostic": True, "iou": 0.45},
        {"name": "Config_6_EdgePad_40pct_ClassAgnostic", "overlap": 0.40, "edge_padding": True, "class_agnostic": True, "iou": 0.45},
        {"name": "Config_7_EdgePad_40pct_IoU_030", "overlap": 0.40, "edge_padding": True, "class_agnostic": False, "iou": 0.30},
        {"name": "Config_8_EdgePad_40pct_IoU_060", "overlap": 0.40, "edge_padding": True, "class_agnostic": False, "iou": 0.60},
    ]

    results = {}
    print("\nRunning configuration matrix...")
    for cfg in configurations:
        cname = cfg["name"]
        print(f"\n--- Testing: {cname} ---")
        print(f"    Overlap={cfg['overlap']*100:.0f}%, EdgePad={cfg['edge_padding']}, ClassAgnostic={cfg['class_agnostic']}, IoU={cfg['iou']}")

        runner = TiledInferenceRunner(
            inspector=inspector,
            tile_size=640,
            overlap=cfg["overlap"],
            iou_threshold=cfg["iou"],
            edge_padding=cfg["edge_padding"],
            class_agnostic_nms=cfg["class_agnostic"]
        )

        res = evaluate_runner_config(runner, board_data)
        res["config"] = cfg
        results[cname] = res

        print(f"    -> Recall: {res['recall']*100:.2f}% ({res['true_positives']}/{res['ground_truth']}), "
              f"Precision: {res['precision']*100:.2f}%, F1: {res['f1']:.4f}, "
              f"Small Defect Recall: {res['small_defect_recall']*100:.2f}%, "
              f"Avg Tiles: {res['mean_tiles_per_board']}, Latency: {res['mean_latency_ms']:.1f}ms")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Whole-board inference comparison report saved to: {OUT_JSON}")
    return results


if __name__ == "__main__":
    run_whole_board_inference_study()
