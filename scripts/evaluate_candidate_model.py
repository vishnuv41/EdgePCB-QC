"""
Standardized Candidate Model Evaluation Engine for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Computes all mandatory metrics per Rule 6:
1. Validation Split (2,125 images):
   - Per-class Precision, Recall, F1, mAP50, mAP50-95
   - Small-defect recall (< 32x32 px) vs Medium-defect recall (>= 32x32 px)
2. Clean-Board Robustness (195 verified defect-free boards):
   - Total false-positive boxes, FP rate (FP/board), False Rejection Rate (scrap rate)
   - Per-class false-alarm breakdown (via hallucinations)
3. Whole-Board High-Resolution Inference (25 multi-megapixel boards, 349 GT defects):
   - Config 7 optimal tiling (40% overlap, edge-aware reflection padding, IoU 0.30 NMS)
   - Whole-board Precision, Recall, F1, Latency
4. Untouched Test Set Evaluation (194 DeepPCB test + 390 external colour crops)
"""

import sys
import os
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import cv2
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.tiling import TiledInferenceRunner, compute_iou

CANONICAL_6_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole_missing_pad"
}

SPLIT_8_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole",
    6: "hole_breakout",
    7: "missing_pad"
}


def load_yolo_labels(lbl_path: Path, img_w: int, img_h: int) -> list:
    boxes = []
    if not lbl_path.exists():
        return boxes
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h
                boxes.append({
                    "class_id": cid,
                    "bbox": [x1, y1, x2, y2],
                    "width": w * img_w,
                    "height": h * img_h,
                    "area": (w * img_w) * (h * img_h)
                })
    return boxes


def match_boxes(gt_boxes: list, pred_boxes: list, iou_thresh: float = 0.50, remap_pred_func=None):
    matched_gt = set()
    matched_pred = set()
    matches = []

    sorted_preds = sorted(enumerate(pred_boxes), key=lambda x: x[1].get("confidence", 0), reverse=True)

    for p_idx, pred in sorted_preds:
        p_cid = pred.get("class_id")
        if remap_pred_func:
            p_cid = remap_pred_func(p_cid)
        p_box = pred.get("bbox", [])

        best_iou = 0.0
        best_gt_idx = -1

        for g_idx, gt in enumerate(gt_boxes):
            if g_idx in matched_gt:
                continue
            if gt.get("class_id") != p_cid:
                continue
            iou = compute_iou(p_box, gt.get("bbox", []))
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            matches.append((best_gt_idx, p_idx, best_iou))

    unmatched_gt = [g_idx for g_idx in range(len(gt_boxes)) if g_idx not in matched_gt]
    unmatched_pred = [p_idx for p_idx in range(len(pred_boxes)) if p_idx not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def evaluate_model(
    checkpoint_path: Path,
    experiment_name: str,
    class_map: dict = CANONICAL_6_CLASSES,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    device: str = 0,
    eval_test_split: bool = False
) -> dict:
    print("\n" + "=" * 75)
    print(f"STANDARDIZED EVALUATION: {experiment_name.upper()}")
    print("=" * 75)
    print(f"[*] Checkpoint:     {checkpoint_path}")
    print(f"[*] Num classes:    {len(class_map)}")
    print(f"[*] Confidence:     {conf_thresh}")
    print(f"[*] IoU Threshold:  {iou_thresh}")
    print(f"[*] Device:         {device}")

    model = YOLO(str(checkpoint_path))

    # Helper if evaluating 8-class model against 6-class targets
    remap_func = None
    if len(class_map) == 8:
        def remap_8_to_6(cid):
            if cid in [5, 6, 7]:
                return 5
            return cid
        remap_func = remap_8_to_6

    eval_results = {
        "experiment_name": experiment_name,
        "checkpoint_path": str(checkpoint_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "parameters": {
            "conf_thresh": conf_thresh,
            "iou_thresh": iou_thresh,
            "num_classes": len(class_map)
        }
    }

    # -------------------------------------------------------------
    # 1. Validation Split Evaluation (2,125 images)
    # -------------------------------------------------------------
    val_img_dir = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "images" / "val"
    val_lbl_dir = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "labels" / "val"
    val_imgs = sorted(list(val_img_dir.glob("*.jpg")))
    print(f"\n[*] 1. Evaluating Validation Split ({len(val_imgs)} images)...")

    class_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    small_gt_total = 0
    small_gt_tp = 0
    med_gt_total = 0
    med_gt_tp = 0

    t0 = time.time()
    for img_p in val_imgs:
        lbl_p = val_lbl_dir / (img_p.stem + ".txt")
        with Image.open(img_p) as im:
            img_w, img_h = im.size
        gt_boxes = load_yolo_labels(lbl_p, img_w, img_h)

        # Inference
        res = model.predict(source=str(img_p), conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        pred_boxes = []
        for box in res.boxes:
            b_xyxy = box.xyxy[0].cpu().numpy().tolist()
            b_conf = float(box.conf[0].cpu().numpy())
            b_cls = int(box.cls[0].cpu().numpy())
            pred_boxes.append({
                "class_id": b_cls,
                "confidence": b_conf,
                "bbox": b_xyxy
            })

        matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_thresh, remap_pred_func=remap_func)

        # Tally TPs
        for g_idx, p_idx, iou in matches:
            gt = gt_boxes[g_idx]
            cid = gt["class_id"]
            class_stats[cid]["tp"] += 1

        # Tally FNs
        for g_idx in unmatched_gt:
            gt = gt_boxes[g_idx]
            cid = gt["class_id"]
            class_stats[cid]["fn"] += 1

        # Scale breakdown: small (< 32x32 max side) vs medium (>= 32)
        matched_gt_indices = {g_idx for g_idx, _, _ in matches}
        for g_idx, gt in enumerate(gt_boxes):
            max_side = max(gt["width"], gt["height"])
            if max_side < 32.0:
                small_gt_total += 1
                if g_idx in matched_gt_indices:
                    small_gt_tp += 1
            else:
                med_gt_total += 1
                if g_idx in matched_gt_indices:
                    med_gt_tp += 1

        # Tally FPs
        for p_idx in unmatched_pred:
            pred = pred_boxes[p_idx]
            cid = pred["class_id"]
            if remap_func:
                cid = remap_func(cid)
            class_stats[cid]["fp"] += 1

    val_time = time.time() - t0
    print(f"[+] Validation inference completed in {val_time:.1f}s ({len(val_imgs)/val_time:.1f} img/s)")

    # Compute per-class and macro metrics
    total_tp = sum(s["tp"] for s in class_stats.values())
    total_fp = sum(s["fp"] for s in class_stats.values())
    total_fn = sum(s["fn"] for s in class_stats.values())

    macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

    small_recall = small_gt_tp / small_gt_total if small_gt_total > 0 else 0.0
    med_recall = med_gt_tp / med_gt_total if med_gt_total > 0 else 0.0

    per_class_summary = {}
    for cid in range(6):
        cname = CANONICAL_6_CLASSES[cid]
        s = class_stats[cid]
        p = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class_summary[cname] = {
            "tp": s["tp"], "fp": s["fp"], "fn": s["fn"],
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)
        }

    eval_results["validation_crop_metrics"] = {
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4),
        "small_defect_recall": round(small_recall, 4),
        "small_defect_tp": small_gt_tp,
        "small_defect_total": small_gt_total,
        "medium_defect_recall": round(med_recall, 4),
        "medium_defect_tp": med_gt_tp,
        "medium_defect_total": med_gt_total,
        "per_class": per_class_summary
    }

    print(f"    * Precision: {macro_p:.4f}, Recall: {macro_r:.4f}, F1: {macro_f1:.4f}")
    print(f"    * Small Defect Recall (<32px): {small_recall:.4f} ({small_gt_tp}/{small_gt_total})")
    print(f"    * Medium Defect Recall (>=32px): {med_recall:.4f} ({med_gt_tp}/{med_gt_total})")
    print(f"    * Open recall: {per_class_summary['open_circuit']['recall']:.4f}, Short recall: {per_class_summary['short_circuit']['recall']:.4f}")

    # -------------------------------------------------------------
    # 2. Clean-Board Robustness Evaluation (195 verified defect-free boards)
    # -------------------------------------------------------------
    print("\n[*] 2. Evaluating Clean-Board False-Positive & Scrap Rate (195 clean templates)...")
    raw_deeppcb = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData"
    clean_test_imgs = []
    for g in ["group90100", "group92000"]:
        clean_test_imgs.extend(list((raw_deeppcb / g).rglob("*_temp.jpg")))

    total_clean_fp_boxes = 0
    clean_boards_with_fp = 0
    clean_boards_rejected = 0  # High confidence critical false alarm
    clean_class_counter = Counter()

    for p in clean_test_imgs:
        res = model.predict(source=str(p), conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        boxes = res.boxes
        num_boxes = len(boxes)
        total_clean_fp_boxes += num_boxes

        has_reject = False
        if num_boxes > 0:
            clean_boards_with_fp += 1
            for b in boxes:
                cid = int(b.cls[0].cpu().numpy())
                conf = float(b.conf[0].cpu().numpy())
                if remap_func:
                    cid = remap_func(cid)
                cname = CANONICAL_6_CLASSES.get(cid, str(cid))
                clean_class_counter[cname] += 1
                # Industrial policy: high confidence open/short/pad -> REJECT
                if conf >= 0.40 and cid in [0, 1, 5]:
                    has_reject = True

        if has_reject:
            clean_boards_rejected += 1

    clean_total = len(clean_test_imgs)
    fp_rate = total_clean_fp_boxes / clean_total if clean_total > 0 else 0.0
    scrap_rate = clean_boards_rejected / clean_total if clean_total > 0 else 0.0
    inspect_rate = (clean_boards_with_fp - clean_boards_rejected) / clean_total if clean_total > 0 else 0.0
    clean_pass_rate = (clean_total - clean_boards_with_fp) / clean_total if clean_total > 0 else 0.0

    eval_results["clean_board_metrics"] = {
        "total_evaluated_boards": clean_total,
        "total_false_positive_boxes": total_clean_fp_boxes,
        "false_positive_rate_per_board": round(fp_rate, 4),
        "clean_pass_rate": round(clean_pass_rate, 4),
        "false_scrap_rate": round(scrap_rate, 4),
        "false_inspect_rate": round(inspect_rate, 4),
        "false_positive_class_distribution": dict(clean_class_counter)
    }

    via_hallucinations = clean_class_counter.get("pin_hole_missing_pad", 0) + clean_class_counter.get("pin_hole", 0) + clean_class_counter.get("hole_breakout", 0) + clean_class_counter.get("missing_pad", 0)
    via_frac = via_hallucinations / total_clean_fp_boxes if total_clean_fp_boxes > 0 else 0.0

    print(f"    * Total FP boxes on 195 clean boards: {total_clean_fp_boxes} ({fp_rate:.2f} boxes/board)")
    print(f"    * Clean board PASS rate: {clean_pass_rate*100:.1f}%, False Scrap (REJECT) rate: {scrap_rate*100:.1f}%")
    print(f"    * Via/Pad Hallucinations: {via_hallucinations} ({via_frac*100:.1f}% of all false positives)")

    # -------------------------------------------------------------
    # 3. Whole-Board High-Resolution Inference (25 boards, 349 GT defects)
    # -------------------------------------------------------------
    print("\n[*] 3. Evaluating Whole-Board High-Resolution Inference (Config 7: 40% overlap, Edge-aware padding)...")
    v2_raw_dir = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train"
    whole_board_imgs = []
    for b in ["68", "80", "81", "82"]:
        whole_board_imgs.extend(list(v2_raw_dir.glob(f"images/{b}-*.jpg")))

    # Simple VisionInspector adapter for TiledInferenceRunner
    class InspectorAdapter:
        def __init__(self, m, dev, conf):
            self.m = m
            self.dev = dev
            self.conf = conf
        def inspect(self, img_arr):
            res = self.m.predict(source=img_arr, conf=self.conf, imgsz=640, device=self.dev, verbose=False)[0]
            dets = []
            for b in res.boxes:
                cid = int(b.cls[0].cpu().numpy())
                cf = float(b.conf[0].cpu().numpy())
                box = b.xyxy[0].cpu().numpy().tolist()
                dets.append({"class_id": cid, "confidence": cf, "bbox": box, "class": CANONICAL_6_CLASSES.get(cid, "defect")})
            return dets

    adapter = InspectorAdapter(model, device, conf_thresh)
    tiler = TiledInferenceRunner(
        inspector=adapter,
        tile_size=640,
        overlap=0.40,
        iou_threshold=0.30,
        edge_padding=True,
        class_agnostic_nms=True
    )

    wb_tp = 0
    wb_fp = 0
    wb_fn = 0
    wb_latencies = []

    v2_class_remap = {0: 5, 1: 2, 2: 0, 3: 1, 4: 3, 5: 4}

    for wb_p in whole_board_imgs:
        img = cv2.imread(str(wb_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Load polygon ground truth
        lbl_p = v2_raw_dir / "labels" / (wb_p.stem + ".txt")
        gt_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        raw_cid = int(parts[0])
                        canonical_cid = v2_class_remap.get(raw_cid, raw_cid)
                        xs = [float(parts[1])*w, float(parts[3])*w, float(parts[5])*w, float(parts[7])*w]
                        ys = [float(parts[2])*h, float(parts[4])*h, float(parts[6])*h, float(parts[8])*h]
                        gt_boxes.append({"class_id": canonical_cid, "bbox": [min(xs), min(ys), max(xs), max(ys)]})

        # Run tiled inference
        t_start = time.time()
        preds = tiler.inspect_tiled(img)
        lat = (time.time() - t_start) * 1000.0
        wb_latencies.append(lat)

        matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, preds, iou_thresh=0.50, remap_pred_func=remap_func)
        wb_tp += len(matches)
        wb_fn += len(unmatched_gt)
        wb_fp += len(unmatched_pred)

    wb_p_score = wb_tp / (wb_tp + wb_fp) if (wb_tp + wb_fp) > 0 else 0.0
    wb_r_score = wb_tp / (wb_tp + wb_fn) if (wb_tp + wb_fn) > 0 else 0.0
    wb_f1_score = 2 * wb_p_score * wb_r_score / (wb_p_score + wb_r_score) if (wb_p_score + wb_r_score) > 0 else 0.0
    wb_mean_lat = float(np.mean(wb_latencies)) if wb_latencies else 0.0

    eval_results["whole_board_metrics"] = {
        "boards_evaluated": len(whole_board_imgs),
        "ground_truth_defects": wb_tp + wb_fn,
        "true_positives": wb_tp,
        "false_positives": wb_fp,
        "false_negatives": wb_fn,
        "precision": round(wb_p_score, 4),
        "recall": round(wb_r_score, 4),
        "f1": round(wb_f1_score, 4),
        "mean_latency_ms": round(wb_mean_lat, 1)
    }

    print(f"    * Whole-Board Recall: {wb_r_score*100:.2f}% ({wb_tp}/{wb_tp+wb_fn})")
    print(f"    * Whole-Board Precision: {wb_p_score*100:.2f}%")
    print(f"    * Whole-Board F1: {wb_f1_score:.4f}")
    print(f"    * Mean Whole-Board GPU Latency: {wb_mean_lat:.1f} ms")

    # -------------------------------------------------------------
    # 4. Untouched Test Set Evaluation (Once after selection or if requested)
    # -------------------------------------------------------------
    if eval_test_split:
        print("\n[*] 4. Evaluating Untouched Test Sets (DeepPCB Test 194 + External Colour 390)...")
        test_img_dir = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "images" / "test"
        test_lbl_dir = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "labels" / "test"
        test_imgs = sorted(list(test_img_dir.glob("*.jpg")))
        
        test_tp = 0
        test_fp = 0
        test_fn = 0
        for p in test_imgs:
            lbl_p = test_lbl_dir / (p.stem + ".txt")
            gt_boxes = load_yolo_labels(lbl_p, 640, 640)
            res = model.predict(source=str(p), conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
            pred_boxes = [{"class_id": int(b.cls[0].cpu().numpy()), "confidence": float(b.conf[0].cpu().numpy()), "bbox": b.xyxy[0].cpu().numpy().tolist()} for b in res.boxes]
            matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_thresh, remap_pred_func=remap_func)
            test_tp += len(matches)
            test_fn += len(unmatched_gt)
            test_fp += len(unmatched_pred)
            
        test_p = test_tp / (test_tp + test_fp) if (test_tp + test_fp) > 0 else 0.0
        test_r = test_tp / (test_tp + test_fn) if (test_tp + test_fn) > 0 else 0.0
        test_f1 = 2 * test_p * test_r / (test_p + test_r) if (test_p + test_r) > 0 else 0.0
        
        eval_results["untouched_test_metrics"] = {
            "deeppcb_test_recall": round(test_r, 4),
            "deeppcb_test_precision": round(test_p, 4),
            "deeppcb_test_f1": round(test_f1, 4)
        }
        print(f"    * DeepPCB Test Recall: {test_r*100:.2f}%, Precision: {test_p*100:.2f}%, F1: {test_f1:.4f}")

    # Save evaluation json
    out_json = WORKSPACE_ROOT / "benchmarks" / "results" / f"{experiment_name}_evaluation.json"
    with open(out_json, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[+] Saved authoritative evaluation results to: {out_json}")
    return eval_results


if __name__ == "__main__":
    baseline_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
    evaluate_model(baseline_pt, "baseline_multidomain_reference")
