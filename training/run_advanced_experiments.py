"""
Advanced Training & Controlled Experiments Pipeline for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Orchestrates 5 separate, controlled experiments (50 epochs, seed=42, RTX 5050 GPU):
- Experiment A: Baseline fine-tuning (continued training on unified dataset)
- Experiment B: Class taxonomy experiment (8-class split taxonomy)
- Experiment C: Clean-board hard-negative learning (with empty-label clean templates)
- Experiment D: Domain-balanced sampling (1:1:1 domain parity)
- Experiment E: Small-defect targeted augmentation (micro-defect copy-paste & scale tuning)

Protects Drive C: by executing runs on E:\EdgePCB-QC_cache\runs\ and copying final weights
and results to models/checkpoints/<exp_name>/ and benchmarks/results/.
"""

import sys
import os
import shutil
import time
import json
from pathlib import Path
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.evaluate_candidate_model import evaluate_model, CANONICAL_6_CLASSES, SPLIT_8_CLASSES

CACHE_RUNS_DIR = Path("E:/EdgePCB-QC_cache/runs")
CHECKPOINTS_DIR = WORKSPACE_ROOT / "models" / "checkpoints"
ONNX_DIR = WORKSPACE_ROOT / "models" / "onnx"
BENCHMARKS_DIR = WORKSPACE_ROOT / "benchmarks" / "results"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)


def check_gpu():
    if torch.cuda.is_available():
        return 0
    return "cpu"


def run_experiment(
    exp_id: str,
    base_model_path: str,
    data_yaml_path: Path,
    num_classes: int = 6,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 10,
    seed: int = 42,
    custom_aug: dict = None
):
    print("\n" + "=" * 80)
    print(f"EXECUTING EXPERIMENT: {exp_id.upper()}")
    print("=" * 80)
    print(f"[*] Base Model:       {base_model_path}")
    print(f"[*] Dataset YAML:     {data_yaml_path}")
    print(f"[*] Epochs:           {epochs}")
    print(f"[*] Batch size:       {batch}")
    print(f"[*] Image size:       {imgsz}")
    print(f"[*] Patience:         {patience}")
    print(f"[*] Seed:             {seed}")
    print(f"[*] Cache runs dir:   {CACHE_RUNS_DIR}")

    device = check_gpu()
    print(f"[*] Compute device:   {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    # Hyperparameter defaults preserving PCB geometry
    train_args = {
        "data": str(data_yaml_path),
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "seed": seed,
        "project": str(CACHE_RUNS_DIR),
        "name": exp_id,
        "exist_ok": True,
        "save": True,
        "plots": False,
        "verbose": True,
        "workers": 0,
        "patience": patience,
        "hsv_h": 0.015,
        "hsv_s": 0.50,
        "hsv_v": 0.35,
        "degrees": 0.0,
        "shear": 0.0,
        "translate": 0.10,
        "scale": 0.40,
        "fliplr": 0.50,
        "flipud": 0.50,
        "mosaic": 1.0,
    }

    if custom_aug:
        train_args.update(custom_aug)

    # Initialize model
    model = YOLO(base_model_path)

    start_train_time = time.time()
    results = model.train(**train_args)
    train_duration = time.time() - start_train_time

    # Locate best weights in cache
    exp_cache_dir = CACHE_RUNS_DIR / exp_id
    best_weights_cached = exp_cache_dir / "weights" / "best.pt"
    if not best_weights_cached.exists():
        best_weights_cached = exp_cache_dir / "weights" / "last.pt"

    # Copy checkpoints & training logs to repository
    target_exp_dir = CHECKPOINTS_DIR / exp_id
    target_weights_dir = target_exp_dir / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)

    dest_best_pt = target_weights_dir / "best.pt"
    shutil.copy2(best_weights_cached, dest_best_pt)

    csv_cached = exp_cache_dir / "results.csv"
    if csv_cached.exists():
        shutil.copy2(csv_cached, target_exp_dir / "results.csv")

    args_cached = exp_cache_dir / "args.yaml"
    if args_cached.exists():
        shutil.copy2(args_cached, target_exp_dir / "args.yaml")

    print(f"\n[+] Training Complete! Total time: {train_duration/60:.1f} minutes.")
    print(f"[+] Saved best checkpoint to: {dest_best_pt}")

    # Export ONNX
    onnx_dest = ONNX_DIR / f"{exp_id}.onnx"
    try:
        print(f"[*] Exporting ONNX model to {onnx_dest}...")
        model_best = YOLO(str(dest_best_pt))
        exported_path = model_best.export(format="onnx", imgsz=imgsz, opset=17)
        if Path(exported_path).exists():
            shutil.copy2(exported_path, onnx_dest)
            print(f"[+] ONNX export succeeded: {onnx_dest}")
    except Exception as e:
        print(f"[!] ONNX export error: {e}")

    # Run Standardized Evaluation
    class_map = SPLIT_8_CLASSES if num_classes == 8 else CANONICAL_6_CLASSES
    eval_metrics = evaluate_model(
        checkpoint_path=dest_best_pt,
        experiment_name=exp_id,
        class_map=class_map,
        conf_thresh=0.25,
        iou_thresh=0.50,
        device=device,
        eval_test_split=False
    )
    eval_metrics["train_duration_seconds"] = round(train_duration, 1)

    # Save summary
    summary_path = BENCHMARKS_DIR / f"{exp_id}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)

    return eval_metrics


EXPERIMENTS_CONFIG = {
    "exp_a": {
        "exp_id": "exp_a_baseline_finetune",
        "description": "Baseline Fine-Tuning (Continued training on 6-class unified dataset)",
        "base_model": str(WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"),
        "data_yaml": WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "data.yaml",
        "num_classes": 6,
        "epochs": 50,
        "batch": 32,
        "patience": 8,
        "custom_aug": None
    },
    "exp_b": {
        "exp_id": "exp_b_taxonomy_split",
        "description": "Class Taxonomy Split (8-class split taxonomy: pin_hole vs hole_breakout vs missing_pad)",
        "base_model": "yolov8n.pt",
        "data_yaml": WORKSPACE_ROOT / "datasets" / "processed" / "taxonomy_8class_yolo" / "data.yaml",
        "num_classes": 8,
        "epochs": 50,
        "batch": 32,
        "patience": 8,
        "custom_aug": None
    },
    "exp_c": {
        "exp_id": "exp_c_hard_negatives",
        "description": "Clean-Board Hard-Negative Learning (Ingesting 1,306 clean template images with empty labels)",
        "base_model": str(WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"),
        "data_yaml": WORKSPACE_ROOT / "datasets" / "processed" / "hard_negatives_yolo" / "data.yaml",
        "num_classes": 6,
        "epochs": 50,
        "batch": 32,
        "patience": 8,
        "custom_aug": None
    },
    "exp_d": {
        "exp_id": "exp_d_domain_balanced",
        "description": "Domain-Balanced Sampling (1:1:1 domain parity: 1500 images per modality)",
        "base_model": str(WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"),
        "data_yaml": WORKSPACE_ROOT / "datasets" / "processed" / "domain_balanced_yolo" / "data.yaml",
        "num_classes": 6,
        "epochs": 50,
        "batch": 32,
        "patience": 8,
        "custom_aug": None
    },
    "exp_e": {
        "exp_id": "exp_e_small_defect_aug",
        "description": "Small-Defect Targeted Augmentation (Micro-defect copy-paste & scale tuning)",
        "base_model": str(WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"),
        "data_yaml": WORKSPACE_ROOT / "datasets" / "processed" / "small_defect_yolo" / "data.yaml",
        "num_classes": 6,
        "epochs": 50,
        "batch": 32,
        "patience": 8,
        "custom_aug": {
            "scale": 0.20,
            "mosaic": 1.0,
            "hsv_h": 0.015,
            "hsv_s": 0.40,
            "hsv_v": 0.30
        }
    }
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run EdgePCB-QC Controlled Training Experiments")
    parser.add_argument("--exp", type=str, required=True, choices=["exp_a", "exp_b", "exp_c", "exp_d", "exp_e", "all"], help="Experiment identifier to run")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    args = parser.parse_args()

    to_run = list(EXPERIMENTS_CONFIG.keys()) if args.exp == "all" else [args.exp]

    print("=" * 80)
    print(f"EdgePCB-QC CONTROLLED EXPERIMENTS RUNNER: {len(to_run)} EXPERIMENT(S)")
    print("=" * 80)

    for key in to_run:
        cfg = EXPERIMENTS_CONFIG[key]
        epochs = args.epochs if args.epochs is not None else cfg["epochs"]
        patience = args.patience if args.patience is not None else cfg["patience"]

        print(f"\n>>> Starting {key.upper()}: {cfg['description']} <<<")
        run_experiment(
            exp_id=cfg["exp_id"],
            base_model_path=cfg["base_model"],
            data_yaml_path=cfg["data_yaml"],
            num_classes=cfg["num_classes"],
            epochs=epochs,
            batch=cfg["batch"],
            patience=patience,
            seed=42,
            custom_aug=cfg["custom_aug"]
        )

    print("\n" + "=" * 80)
    print("ALL REQUESTED EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
