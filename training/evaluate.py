"""
Model Evaluation on Unseen PCB Boards
Bharat AI-SoC Challenge 2026-27 | PS5

Evaluates trained edge detector against test set:
- Computes overall Precision, Recall, mAP50, mAP50-95
- Computes per-class metrics
- Specifically reports open-circuit and short-circuit recall separately
- Exports evaluation report to benchmarks/results/
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = WORKSPACE_ROOT / "datasets" / "processed" / "deeppcb_yolo" / "data.yaml"
RESULTS_DIR = WORKSPACE_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    weights_path: Path,
    data_yaml: Path = DEFAULT_DATA_YAML,
    split: str = "test",
    imgsz: int = 640,
    device: str = "cpu"
):
    print("=" * 70)
    print("EVALUATING MODEL ON UNSEEN PCB BOARDS")
    print("=" * 70)
    print(f"[*] Weights: {weights_path}")
    print(f"[*] Data:    {data_yaml}")
    print(f"[*] Split:   {split} (Unseen board groups)")
    print(f"[*] Device:  {device}")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        verbose=True,
        plots=True
    )

    class_names = model.names
    overall_p = float(metrics.box.mp)
    overall_r = float(metrics.box.mr)
    overall_map50 = float(metrics.box.map50)
    overall_map = float(metrics.box.map)

    # Per-class metrics
    per_class = {}
    for i, cname in class_names.items():
        p = float(metrics.box.p[i]) if len(metrics.box.p) > i else 0.0
        r = float(metrics.box.r[i]) if len(metrics.box.r) > i else 0.0
        map50 = float(metrics.box.ap50[i]) if len(metrics.box.ap50) > i else 0.0
        map_all = float(metrics.box.ap[i]) if len(metrics.box.ap) > i else 0.0
        per_class[cname] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(map50, 4),
            "map50_95": round(map_all, 4)
        }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights_path),
        "split_evaluated": split,
        "input_resolution": f"{imgsz}x{imgsz}",
        "overall": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "map50": round(overall_map50, 4),
            "map50_95": round(overall_map, 4)
        },
        "critical_classes": {
            "open_circuit": per_class.get("open_circuit", {}),
            "short_circuit": per_class.get("short_circuit", {})
        },
        "per_class": per_class
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY (UNSEEN BOARDS)")
    print("=" * 70)
    print(f"Overall Precision: {overall_p:.4f} | Recall: {overall_r:.4f} | mAP50: {overall_map50:.4f} | mAP50-95: {overall_map:.4f}")
    print("-" * 70)
    print(f"{'Class Name':25s} | {'Precision':10s} | {'Recall':10s} | {'mAP50':10s} | {'mAP50-95':10s}")
    print("-" * 70)
    for cname, vals in per_class.items():
        is_crit = "**CRITICAL** " if cname in ["open_circuit", "short_circuit"] else "             "
        print(f"{cname:25s} | {vals['precision']:10.4f} | {vals['recall']:10.4f} | {vals['map50']:10.4f} | {vals['map50_95']:10.4f}")

    print("\n" + "=" * 70)
    print("SPECIFIC ELECTRICAL SAFETY DEFECT AUDIT:")
    print("=" * 70)
    oc = per_class.get("open_circuit", {})
    sc = per_class.get("short_circuit", {})
    print(f"[*] Open Circuit Recall:  {oc.get('recall', 0.0):.4f} (Precision: {oc.get('precision', 0.0):.4f}, mAP50: {oc.get('map50', 0.0):.4f})")
    print(f"[*] Short Circuit Recall: {sc.get('recall', 0.0):.4f} (Precision: {sc.get('precision', 0.0):.4f}, mAP50: {sc.get('map50', 0.0):.4f})")

    out_file = RESULTS_DIR / "eval_unseen_boards.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Detailed evaluation metrics saved to: {out_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_model(
        weights_path=Path(args.weights),
        data_yaml=Path(args.data),
        split=args.split,
        imgsz=args.imgsz,
        device=args.device
    )
