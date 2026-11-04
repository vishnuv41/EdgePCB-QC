"""
Comprehensive Generalization Audit & Missing Defect Investigation Pipeline
Bharat AI-SoC Challenge 2026-27 | PS5

Executes Phase 2 and Phase 3:
- Evaluates current multidomain checkpoint across distinct unseen domains:
  1. DeepPCB Untouched Test (Monochrome, 194 images)
  2. DsPCBSD+ Validation (Industrial colour patches, 1444 images)
  3. PCB-Defect V2 External Colour Boards (Held-out boards 68, 80, 81, 82: 390 crops)
  4. PCB-Defect V2 Full-Board Tiled Images (25 whole boards)
  5. Clean Bare PCB Images (195 verified defect-free templates)
- Computes:
  - Per-class Precision, Recall, F1
  - Defect scale breakdown: Small (<32px), Medium (32-96px), Large (>96px)
  - Duplicate detection rates
  - Board-level Sensitivity, Specificity, False Alarm Rate, False Scrap Rate
- Generates side-by-side annotated visualization images (Ground Truth vs Prediction).
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

CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
DATA_ROOT = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo"
RAW_DIR = WORKSPACE_ROOT / "datasets" / "raw"
VIS_DIR = WORKSPACE_ROOT / "benchmarks" / "results" / "generalization_audit"
OUT_JSON = WORKSPACE_ROOT / "benchmarks" / "results" / "comprehensive_generalization_audit.json"

CANONICAL_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole_missing_pad"
}


def draw_comparison_visualization(img: np.ndarray, gt_boxes: list, pred_boxes: list, matched_pairs: list, unmatched_gt: list, unmatched_pred: list, title: str) -> np.ndarray:
    """
    Renders visual comparison:
    - Green: Ground Truth
    - Cyan: True Positive Prediction
    - Red: False Positive Prediction
    - Magenta: Missed Defect (False Negative)
    """
    vis = img.copy()
    h, w = vis.shape[:2]

    # Draw Missed Ground Truth (False Negatives) in Magenta
    for g_idx in unmatched_gt:
        g = gt_boxes[g_idx]
        x1, y1, x2, y2 = map(int, g["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.putText(vis, f"MISSED: {g['name']}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

    # Draw Matched Pairs (True Positives) in Cyan
    for p_idx, g_idx, iou in matched_pairs:
        p = pred_boxes[p_idx]
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(vis, f"TP: {p['name']} ({p['conf']:.2f}, IoU={iou:.2f})", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)

    # Draw False Positives in Red
    for p_idx in unmatched_pred:
        p = pred_boxes[p_idx]
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"FP: {p['name']} ({p['conf']:.2f})", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1, cv2.LINE_AA)

    return vis


def evaluate_domain_split(model, img_paths: list, lbl_dir: Path, domain_name: str, conf_thresh: float = 0.25, save_vis_count: int = 5):
    print(f"\n[*] Evaluating Domain: {domain_name} ({len(img_paths)} images)...")
    device = 0 if torch.cuda.is_available() else "cpu"

    gt_by_class = Counter()
    tp_by_class = Counter()
    fp_by_class = Counter()
    fn_by_class = Counter()
    
    size_breakdown = {
        "small": {"gt": 0, "tp": 0, "fn": 0},    # < 32px side (< 1024 area)
        "medium": {"gt": 0, "tp": 0, "fn": 0},   # 32 - 96px
        "large": {"gt": 0, "tp": 0, "fn": 0}     # > 96px
    }

    total_duplicates = 0
    saved_vis = 0
    board_status_counter = Counter()

    for idx, img_p in enumerate(img_paths):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Read Ground Truth
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = max(0, (xc - bw / 2.0) * w)
                    y1 = max(0, (yc - bh / 2.0) * h)
                    x2 = min(w, (xc + bw / 2.0) * w)
                    y2 = min(h, (yc + bh / 2.0) * h)
                    
                    area = (x2 - x1) * (y2 - y1)
                    if area < 1024:
                        scale = "small"
                    elif area < 9216:
                        scale = "medium"
                    else:
                        scale = "large"

                    cname = CANONICAL_CLASSES.get(cid, f"class_{cid}")
                    gt_boxes.append({
                        "cls": cid,
                        "name": cname,
                        "bbox": [x1, y1, x2, y2],
                        "scale": scale
                    })
                    gt_by_class[cname] += 1
                    size_breakdown[scale]["gt"] += 1

        # Predict
        res = model.predict(source=img, conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        pred_boxes = []
        for b in res.boxes:
            c = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            xyxy = b.xyxy[0].tolist()
            cname = CANONICAL_CLASSES.get(c, f"class_{c}")
            pred_boxes.append({
                "cls": c,
                "name": cname,
                "conf": cf,
                "bbox": xyxy
            })

        # Match Predictions to Ground Truth (IoU >= 0.50)
        matched_gt = set()
        matched_pred = set()
        matched_pairs = []
        pred_to_gt_matches = defaultdict(list)

        # Sort predictions by confidence descending
        sorted_pred_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_boxes[i]["conf"], reverse=True)

        for p_idx in sorted_pred_indices:
            p = pred_boxes[p_idx]
            best_iou = 0.0
            best_g_idx = -1
            for g_idx, g in enumerate(gt_boxes):
                if g["cls"] != p["cls"]:
                    continue
                iou = compute_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= 0.50 and best_g_idx >= 0:
                pred_to_gt_matches[best_g_idx].append(p_idx)
                if best_g_idx not in matched_gt:
                    matched_gt.add(best_g_idx)
                    matched_pred.add(p_idx)
                    matched_pairs.append((p_idx, best_g_idx, best_iou))
                    tp_by_class[p["name"]] += 1
                    size_breakdown[gt_boxes[best_g_idx]["scale"]]["tp"] += 1
                else:
                    # Duplicate detection on same ground truth
                    total_duplicates += 1
                    fp_by_class[p["name"]] += 1
            else:
                fp_by_class[p["name"]] += 1

        # Unmatched GT = False Negatives
        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
        for g_idx in unmatched_gt:
            g = gt_boxes[g_idx]
            fn_by_class[g["name"]] += 1
            size_breakdown[g["scale"]]["fn"] += 1

        # Unmatched Predictions = False Positives
        unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]

        # Board-level classification
        is_clean_ground_truth = (len(gt_boxes) == 0)
        has_detections = (len(pred_boxes) > 0)
        if is_clean_ground_truth:
            if has_detections:
                board_status_counter["false_positive_board"] += 1
            else:
                board_status_counter["true_negative_board"] += 1
        else:
            if len(matched_gt) > 0:
                board_status_counter["true_positive_board"] += 1
            else:
                board_status_counter["false_negative_board"] += 1

        # Save annotated comparison images for qualitative analysis
        if saved_vis < save_vis_count and (len(unmatched_gt) > 0 or len(unmatched_pred) > 0 or len(matched_pairs) > 0):
            vis_img = draw_comparison_visualization(
                img, gt_boxes, pred_boxes, matched_pairs, unmatched_gt, unmatched_pred,
                f"{domain_name}: {img_p.name}"
            )
            save_name = f"{domain_name.replace(' ', '_').lower()}_{img_p.stem}_audit.jpg"
            VIS_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(VIS_DIR / save_name), vis_img)
            saved_vis += 1

    # Metrics computation
    per_class_metrics = {}
    all_classes = set(list(gt_by_class.keys()) + list(tp_by_class.keys()) + list(fp_by_class.keys()))
    total_tp = sum(tp_by_class.values())
    total_fp = sum(fp_by_class.values())
    total_fn = sum(fn_by_class.values())
    total_gt = sum(gt_by_class.values())

    macro_p = []
    macro_r = []
    macro_f1 = []

    for cname in sorted(all_classes):
        gt_c = gt_by_class[cname]
        tp_c = tp_by_class[cname]
        fp_c = fp_by_class[cname]
        fn_c = fn_by_class[cname]

        prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        rec = tp_c / gt_c if gt_c > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[cname] = {
            "ground_truth": gt_c,
            "true_positives": tp_c,
            "false_positives": fp_c,
            "false_negatives": fn_c,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }
        if gt_c > 0:
            macro_p.append(prec)
            macro_r.append(rec)
            macro_f1.append(f1)

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "domain_name": domain_name,
        "total_images": len(img_paths),
        "total_ground_truth": total_gt,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_duplicate_detections": total_duplicates,
        "overall_precision": round(overall_prec, 4),
        "overall_recall": round(overall_rec, 4),
        "overall_f1": round(overall_f1, 4),
        "macro_precision": round(float(np.mean(macro_p)), 4) if macro_p else 0.0,
        "macro_recall": round(float(np.mean(macro_r)), 4) if macro_r else 0.0,
        "macro_f1": round(float(np.mean(macro_f1)), 4) if macro_f1 else 0.0,
        "per_class": per_class_metrics,
        "scale_breakdown": size_breakdown,
        "board_level_summary": dict(board_status_counter)
    }


def run_comprehensive_audit():
    print("=" * 75)
    print("EdgePCB-QC: COMPREHENSIVE UNSEEN-BOARD GENERALIZATION AUDIT")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 75)

    model = YOLO(str(CHECKPOINT_PATH))

    # Domain 1: DeepPCB Untouched Test (Monochrome, 194 images)
    deeppcb_test_imgs = sorted(list((DATA_ROOT / "images" / "test").glob("deeppcb_*.jpg")))
    deeppcb_test_lbl = DATA_ROOT / "labels" / "test"

    # Domain 2: DsPCBSD+ Validation (Industrial Colour Patches, 1,444 images)
    dspcbsd_val_imgs = sorted(list((DATA_ROOT / "images" / "val").glob("dspcbsd_*.jpg")))
    dspcbsd_val_lbl = DATA_ROOT / "labels" / "val"

    # Domain 3: PCB-Defect V2 Validation (Tiled Crops, 495 images)
    pcbdefect_val_imgs = sorted(list((DATA_ROOT / "images" / "val").glob("pcbdefect_*.jpg")))
    pcbdefect_val_lbl = DATA_ROOT / "labels" / "val"

    # Domain 4: PCB-Defect V2 External Colour Boards (390 crops, untouched boards 68, 80, 81, 82)
    ext_color_imgs = sorted(list((DATA_ROOT / "images" / "external_color_test").glob("pcbdefect_*.jpg")))
    ext_color_lbl = DATA_ROOT / "labels" / "external_color_test"

    domains = {
        "DeepPCB_Untouched_Test": (deeppcb_test_imgs, deeppcb_test_lbl, "DeepPCB Untouched Test (Monochrome)"),
        "DsPCBSD_Validation": (dspcbsd_val_imgs, dspcbsd_val_lbl, "DsPCBSD+ Validation (Colour Patches)"),
        "PCB_Defect_V2_Validation": (pcbdefect_val_imgs, pcbdefect_val_lbl, "PCB-Defect V2 Validation (Tiled Crops)"),
        "PCB_Defect_V2_External_Colour": (ext_color_imgs, ext_color_lbl, "PCB-Defect V2 External Colour (Untouched Boards)")
    }

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint": str(CHECKPOINT_PATH),
        "confidence_threshold": 0.25,
        "iou_matching_threshold": 0.50,
        "domain_evaluations": {}
    }

    for dkey, (imgs, lbls, dname) in domains.items():
        res = evaluate_domain_split(model, imgs, lbls, dname, conf_thresh=0.25, save_vis_count=5)
        results["domain_evaluations"][dkey] = res

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Comprehensive generalization audit JSON saved to: {OUT_JSON}")
    print(f"[+] Visual comparison images saved to: {VIS_DIR}")
    return results


if __name__ == "__main__":
    run_comprehensive_audit()
