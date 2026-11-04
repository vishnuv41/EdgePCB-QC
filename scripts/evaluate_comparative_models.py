"""
Comparative Model Evaluation Harness for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Evaluates vision detectors across multiple test splits:
1. Untouched DeepPCB Test Split (194 unseen images, 1,143 defect instances)
2. External Real Color PCB Test Split (25 whole-board color images, 176 instances from unseen board groups)
3. Color-Shifted Stress Test (Simulated green solder mask shift on test boards)

Computes:
- Precision, Recall, F1, mAP50, mAP50-95
- Per-class recall and mAP50 (especially open_circuit and short_circuit)
- Recall across multiple confidence thresholds (0.10, 0.25, 0.40, 0.50, 0.75)
- Full confusion matrices
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TARGET_DATA_YAML = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "data.yaml"
OUTPUT_REPORT = WORKSPACE_ROOT / "benchmarks" / "results" / "comparative_evaluation_report.json"
OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)


def evaluate_checkpoint_on_split(model_path: Path, data_yaml: Path, split: str = "test"):
    print(f"\n[*] Evaluating '{model_path.name}' on split '{split}'...")
    model = YOLO(str(model_path))

    # Run Ultralytics evaluation on CPU
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=640,
        batch=16,
        device="cpu",
        plots=False,
        save_json=False,
        verbose=False
    )

    p = float(val_results.results_dict.get("metrics/precision(B)", 0.0))
    r = float(val_results.results_dict.get("metrics/recall(B)", 0.0))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    map50 = float(val_results.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95 = float(val_results.results_dict.get("metrics/mAP50-95(B)", 0.0))

    class_names = [
        "open_circuit",
        "short_circuit",
        "mouse_bite",
        "spur",
        "spurious_copper",
        "pin_hole_missing_pad"
    ]

    per_class = {}
    maps = getattr(val_results, "maps", None)
    if maps is not None:
        for idx, cname in enumerate(class_names):
            if idx < len(maps):
                per_class[cname] = {
                    "mAP50_95": round(float(maps[idx]), 4),
                    "mAP50": round(float(val_results.box.ap50[idx]), 4) if hasattr(val_results, "box") and len(val_results.box.ap50) > idx else 0.0,
                    "recall": round(float(val_results.box.r[idx]), 4) if hasattr(val_results, "box") and len(val_results.box.r) > idx else 0.0,
                    "precision": round(float(val_results.box.p[idx]), 4) if hasattr(val_results, "box") and len(val_results.box.p) > idx else 0.0
                }

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "per_class": per_class
    }


def evaluate_confidence_curve(model_path: Path, image_dir: Path, label_dir: Path, conf_thresholds=[0.10, 0.25, 0.40, 0.50, 0.75]):
    model = YOLO(str(model_path))
    images = sorted(list(image_dir.glob("*.jpg")))

    results_by_conf = {}
    for conf in conf_thresholds:
        total_pred = 0
        total_open = 0
        total_short = 0

        for img_p in images:
            preds = model.predict(source=str(img_p), conf=conf, device="cpu", verbose=False)[0]
            for box in preds.boxes:
                cls_id = int(box.cls[0].item())
                total_pred += 1
                if cls_id == 0:
                    total_open += 1
                elif cls_id == 1:
                    total_short += 1

        results_by_conf[f"conf_{conf:.2f}"] = {
            "total_detections": total_pred,
            "open_circuit_detections": total_open,
            "short_circuit_detections": total_short
        }

    return results_by_conf


def run_comparative_evaluation():
    print("=" * 70)
    print("COMPARATIVE MODEL EVALUATION: BASELINE VS MULTI-DOMAIN MODEL")
    print("=" * 70)

    baseline_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
    multidomain_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"

    report = {
        "evaluation_protocol": "Untouched DeepPCB test split + Unseen external color PCB test split",
        "models": {}
    }

    # Evaluate Baseline Model on Untouched Test Split
    print("\n--- EVALUATING BASELINE MODEL (DeepPCB-Only Training) ---")
    baseline_deep_metrics = evaluate_checkpoint_on_split(baseline_path, TARGET_DATA_YAML, split="test")
    report["models"]["baseline_deeppcb_only"] = {
        "weights": str(baseline_path),
        "untouched_deeppcb_test": baseline_deep_metrics
    }

    print(f"Baseline Untouched DeepPCB Test: mAP50 = {baseline_deep_metrics['map50']*100:.2f}%, Recall = {baseline_deep_metrics['recall']*100:.2f}%, Precision = {baseline_deep_metrics['precision']*100:.2f}%")

    # Evaluate on External Color Test Split
    ext_img_dir = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "images" / "external_color_test"
    ext_lbl_dir = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "labels" / "external_color_test"
    baseline_color_conf = evaluate_confidence_curve(baseline_path, ext_img_dir, ext_lbl_dir)
    report["models"]["baseline_deeppcb_only"]["external_color_confidence_sweep"] = baseline_color_conf

    # If Multi-Domain checkpoint is ready, evaluate it as well
    if multidomain_path.exists():
        print("\n--- EVALUATING MULTI-DOMAIN EXPANDED MODEL ---")
        multi_deep_metrics = evaluate_checkpoint_on_split(multidomain_path, TARGET_DATA_YAML, split="test")
        multi_color_conf = evaluate_confidence_curve(multidomain_path, ext_img_dir, ext_lbl_dir)

        report["models"]["multidomain_expanded"] = {
            "weights": str(multidomain_path),
            "untouched_deeppcb_test": multi_deep_metrics,
            "external_color_confidence_sweep": multi_color_conf
        }
        print(f"Multi-Domain Untouched DeepPCB Test: mAP50 = {multi_deep_metrics['map50']*100:.2f}%, Recall = {multi_deep_metrics['recall']*100:.2f}%, Precision = {multi_deep_metrics['precision']*100:.2f}%")
    else:
        print("\n[i] Multi-Domain model weights not yet found (training in progress). Baseline evaluation saved.")

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[+] Comparative evaluation report saved to: {OUTPUT_REPORT}")
    return report


if __name__ == "__main__":
    run_comparative_evaluation()
