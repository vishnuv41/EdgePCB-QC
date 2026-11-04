"""
Board-Level Evaluation & Clean-Board False Alarm Rate Audit
Bharat AI-SoC Challenge 2026-27 | PS5

Executes:
1. Clean-Board Evaluation on untouched test golden reference templates (195 defect-free boards).
   - Measures False Positive Detections, False-INSPECT (Overkill), and False-REJECT (False Scrap).
   - Compares Baseline vs Unified Multi-Domain models at confidence 0.10, 0.25, 0.40.
2. Whole-Board High-Resolution Evaluation on PCB-Defect V2 external colour boards (68, 80, 81, 82).
   - Evaluates TiledInferenceRunner across tile sizes: 640, 960, and 1280 (with 25% overlap).
   - Compares defect recall, precision, and latency on GPU (RTX 5050) and CPU.
3. Exports results to benchmarks/results/board_level_evaluation.json.
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
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from decision_engine.decision import DecisionEngine
from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner, global_nms, compute_iou

BASELINE_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
MULTIDOMAIN_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
OUT_JSON = WORKSPACE_ROOT / "benchmarks" / "results" / "board_level_evaluation.json"

CANONICAL_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole_missing_pad"
}


def evaluate_clean_boards(model_path: Path, clean_img_paths: list, device: str = "0"):
    """
    Evaluates detector on verified clean (defect-free) boards to measure
    false alarm (false-INSPECT) and false scrap (false-REJECT) rates.
    """
    print(f"\n[*] Evaluating Clean Boards with {model_path.name} on {len(clean_img_paths)} images...")
    engine = DecisionEngine()
    model = YOLO(str(model_path))

    thresholds = [0.10, 0.25, 0.40]
    results_by_thresh = {}

    for conf in thresholds:
        total_boards = len(clean_img_paths)
        decisions_count = Counter()
        false_dets_by_class = Counter()
        total_fp_boxes = 0

        for img_p in clean_img_paths:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            # Run inference
            res = model.predict(source=img, conf=conf, imgsz=640, device=device, verbose=False)[0]
            boxes = res.boxes

            defects = []
            for b in boxes:
                c = int(b.cls[0].item())
                cf = float(b.conf[0].item())
                xyxy = b.xyxy[0].tolist()
                cname = CANONICAL_CLASSES.get(c, "unknown")
                false_dets_by_class[cname] += 1
                total_fp_boxes += 1
                defects.append({
                    "class": cname,
                    "confidence": cf,
                    "bbox": xyxy
                })

            # Evaluate decision engine
            payload = {
                "board_id": img_p.stem,
                "defects": defects,
                "coverage": 1.0,
                "domain_status": "within_domain",
                "image_quality_status": "acceptable"
            }
            rec = engine.evaluate(payload, preset="consumer")
            decisions_count[rec["decision"]] += 1

        pass_count = decisions_count["PASS"]
        inspect_count = decisions_count["INSPECT"]
        reject_count = decisions_count["REJECT"]

        results_by_thresh[str(conf)] = {
            "total_clean_boards": total_boards,
            "true_clean_pass_count": pass_count,
            "false_inspect_count": inspect_count,
            "false_reject_count": reject_count,
            "true_pass_rate": round(pass_count / total_boards, 4),
            "false_inspect_rate": round(inspect_count / total_boards, 4),
            "false_reject_rate": round(reject_count / total_boards, 4),
            "total_false_positive_boxes": total_fp_boxes,
            "mean_fp_per_board": round(total_fp_boxes / total_boards, 3),
            "false_positive_classes": dict(false_dets_by_class)
        }
        print(f"  Conf {conf:.2f}: PASS={pass_count} ({pass_count/total_boards*100:.1f}%), "
              f"INSPECT={inspect_count} ({inspect_count/total_boards*100:.1f}%), "
              f"REJECT={reject_count} ({reject_count/total_boards*100:.1f}%), "
              f"Total FP boxes={total_fp_boxes}")

    return results_by_thresh


def evaluate_whole_board_tiling(model_path: Path, whole_board_imgs: list, whole_board_lbl_dir: Path, device: str = "0"):
    """
    Evaluates high-resolution whole boards using TiledInferenceRunner
    across tile sizes 640, 960, and 1280.
    """
    print(f"\n[*] Evaluating Whole-Board Tiling ({len(whole_board_imgs)} boards)...")
    tile_sizes = [640, 960, 1280]
    tile_results = {}

    # Raw PCB-Defect V2 class to canonical class mapping
    # 0: missing_pad -> 5: pin_hole_missing_pad
    # 1: mouse_bite -> 2: mouse_bite
    # 2: open_circuit -> 0: open_circuit
    # 3: short -> 1: short_circuit
    # 4: spur -> 3: spur
    # 5: spurious_copper -> 4: spurious_copper
    raw_to_canonical = {0: 5, 1: 2, 2: 0, 3: 1, 4: 3, 5: 4}

    for ts in tile_sizes:
        print(f"\n--- Testing Tile Size: {ts} x {ts} (Overlap: 25%) ---")
        inspector = CPUInspector(model_path=model_path, conf_threshold=0.25, device=device)
        runner = TiledInferenceRunner(inspector, tile_size=ts, overlap=0.25, iou_threshold=0.45)

        total_gt = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        latencies = []
        tile_counts = []

        for img_p in whole_board_imgs:
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            h, w = img.shape[:2]
            tiles = runner.slice_image(img)
            tile_counts.append(len(tiles))

            # Load ground truth annotations
            lbl_p = whole_board_lbl_dir / f"{img_p.stem}.txt"
            gt_boxes = []
            if lbl_p.exists():
                with open(lbl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        raw_c = int(parts[0])
                        can_cid = raw_to_canonical.get(raw_c, raw_c)
                        if len(parts) >= 9:
                            # 4-corner polygon format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                            coords = list(map(float, parts[1:9]))
                            xs = [coords[0] * w, coords[2] * w, coords[4] * w, coords[6] * w]
                            ys = [coords[1] * h, coords[3] * h, coords[5] * h, coords[7] * h]
                            gt_boxes.append({"cls": can_cid, "bbox": [min(xs), min(ys), max(xs), max(ys)]})
                        elif len(parts) >= 5:
                            xc, yc, bw, bh = map(float, parts[1:5])
                            x1 = (xc - bw / 2.0) * w
                            y1 = (yc - bh / 2.0) * h
                            x2 = (xc + bw / 2.0) * w
                            y2 = (yc + bh / 2.0) * h
                            gt_boxes.append({"cls": can_cid, "bbox": [x1, y1, x2, y2]})

            total_gt += len(gt_boxes)

            t0 = time.perf_counter()
            fused_dets = runner.inspect_tiled(img)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

            # Match detections with ground truth at IoU >= 0.50
            matched_gt = set()
            for d in fused_dets:
                pred_cls = [k for k, v in CANONICAL_CLASSES.items() if v == d.get("class")]
                pred_cls_id = pred_cls[0] if pred_cls else -1
                pred_box = d.get("bbox", [])

                best_iou = 0.0
                best_gt_idx = -1
                for g_idx, g in enumerate(gt_boxes):
                    if g_idx in matched_gt or g["cls"] != pred_cls_id:
                        continue
                    iou = compute_iou(pred_box, g["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= 0.50 and best_gt_idx >= 0:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1

            total_fn += (len(gt_boxes) - len(matched_gt))

        recall = total_tp / total_gt if total_gt > 0 else 0.0
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        mean_latency = float(np.mean(latencies))
        mean_tiles = float(np.mean(tile_counts))

        print(f"  Tile Size {ts}: Recall={recall*100:.2f}% ({total_tp}/{total_gt}), "
              f"Precision={precision*100:.2f}%, F1={f1:.4f}, "
              f"Avg Tiles={mean_tiles:.1f}, Latency={mean_latency:.1f}ms")

        tile_results[str(ts)] = {
            "tile_size": ts,
            "overlap": 0.25,
            "mean_tiles_per_board": round(mean_tiles, 1),
            "ground_truth": total_gt,
            "true_positives": total_tp,
            "false_positives": total_fp,
            "false_negatives": total_fn,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "mean_latency_ms": round(mean_latency, 2)
        }

    return tile_results


def run_board_level_audit():
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Compute Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    # 1. Clean Reference Boards (from DeepPCB untouched test groups 90100 & 92000)
    deeppcb_raw = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData"
    clean_templates = [
        f for f in deeppcb_raw.glob("*/*/*_temp.jpg")
        if "group90100" in str(f) or "group92000" in str(f)
    ]
    print(f"[*] Found {len(clean_templates)} clean reference boards in untouched test groups.")

    # 2. Whole-board colour images (PCB-Defect V2 external boards 68, 80, 81, 82)
    pcbdefect_img_dir = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train" / "images"
    pcbdefect_lbl_dir = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train" / "labels"
    ext_boards = [68, 80, 81, 82]
    whole_board_imgs = [
        p for p in pcbdefect_img_dir.glob("*.jpg")
        if any(p.name.startswith(f"{b}-") for b in ext_boards)
    ]
    print(f"[*] Found {len(whole_board_imgs)} whole-board colour images for external boards {ext_boards}.")

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "clean_board_evaluation": {
            "dataset": "DeepPCB Untouched Test (group90100, group92000 clean templates)",
            "total_clean_boards": len(clean_templates),
            "baseline_model": evaluate_clean_boards(BASELINE_PATH, clean_templates, device=device),
            "multidomain_model": evaluate_clean_boards(MULTIDOMAIN_PATH, clean_templates, device=device)
        },
        "whole_board_tiling_evaluation": {
            "dataset": "PCB-Defect V2 External Colour Boards (68, 80, 81, 82)",
            "board_count": len(whole_board_imgs),
            "tiling_comparison": evaluate_whole_board_tiling(MULTIDOMAIN_PATH, whole_board_imgs, pcbdefect_lbl_dir, device=device)
        }
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Board-level evaluation report successfully saved to: {OUT_JSON}")
    return report


if __name__ == "__main__":
    run_board_level_audit()
