"""
Fair Model Training Pipeline for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Trains both models (DeepPCB Baseline and Multi-Domain) under strictly identical,
controlled settings to ensure a scientifically valid comparison:
- Same base architecture (yolov8n.pt)
- Same input resolution (imgsz=640)
- Same epoch budget (10 epochs)
- Same batch size (16)
- Same optimizer, learning rate, and geometric augmentations
- Execution on local NVIDIA GeForce RTX 5050 (sm_120) via PyTorch 2.14.1+cu130
- Automatic export of ONNX checkpoints
- Comprehensive logging of training metrics and comparison summary
"""

import os
import sys
import json
import shutil
import time
from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = WORKSPACE_ROOT / "models" / "checkpoints"
ONNX_DIR = WORKSPACE_ROOT / "models" / "onnx"
BENCHMARKS_DIR = WORKSPACE_ROOT / "benchmarks" / "results"

BASELINE_DATA_YAML = WORKSPACE_ROOT / "datasets" / "processed" / "deeppcb_yolo" / "data.yaml"
MULTIDOMAIN_DATA_YAML = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "data.yaml"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)
BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> str:
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device="cuda:0")
            _ = t + 1
            return 0
        except Exception as e:
            print(f"[!] CUDA kernel error: {e}. Falling back to CPU.")
    return "cpu"


def train_single_model(
    name: str,
    data_yaml: Path,
    epochs: int = 10,
    batch: int = 16,
    imgsz: int = 640,
    seed: int = 42,
    device: str = 0
):
    print("\n" + "=" * 75)
    print(f"STARTING FAIR TRAINING: {name.upper()}")
    print("=" * 75)
    print(f"[*] Dataset:       {data_yaml}")
    print(f"[*] Base Model:    yolov8n.pt")
    print(f"[*] Epochs:        {epochs}")
    print(f"[*] Batch Size:    {batch}")
    print(f"[*] Resolution:    {imgsz}x{imgsz}")
    print(f"[*] Seed:          {seed}")
    print(f"[*] Device:        {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")
    print("=" * 75)

    start_time = time.time()
    model = YOLO("yolov8n.pt")

    # Controlled augmentations preserving PCB geometry (no shear or non-orthogonal rotation)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        seed=seed,
        project=str(CHECKPOINTS_DIR.resolve()),
        name=name,
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        degrees=0.0,
        shear=0.0,
        translate=0.10,
        scale=0.50,
        fliplr=0.50,
        flipud=0.50,
        mosaic=1.0,
        patience=10
    )

    elapsed_s = time.time() - start_time
    exp_dir = CHECKPOINTS_DIR / name
    best_pt = exp_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = exp_dir / "weights" / "last.pt"

    print(f"\n[+] {name} training finished in {elapsed_s:.1f}s.")
    print(f"[+] Best weights: {best_pt} ({best_pt.stat().st_size / (1024*1024):.2f} MB)")

    # Export ONNX
    onnx_dest = ONNX_DIR / f"{name}.onnx"
    try:
        print(f"[*] Exporting to ONNX: {onnx_dest}...")
        exported = model.export(format="onnx", imgsz=imgsz, opset=17)
        if Path(exported).exists():
            shutil.copy2(exported, onnx_dest)
            print(f"[+] ONNX export success: {onnx_dest} ({onnx_dest.stat().st_size / (1024*1024):.2f} MB)")
    except Exception as e:
        print(f"[!] ONNX export error: {e}")

    # Parse results.csv
    results_csv = exp_dir / "results.csv"
    metrics_summary = {}
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = [c.strip() for c in df.columns]
        best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
        metrics_summary = {
            "total_epochs": len(df),
            "best_epoch": int(best_row["epoch"]),
            "best_mAP50": float(best_row["metrics/mAP50(B)"]),
            "best_mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
            "precision_at_best": float(best_row["metrics/precision(B)"]),
            "recall_at_best": float(best_row["metrics/recall(B)"]),
            "final_train_box_loss": float(df.iloc[-1]["train/box_loss"]),
            "final_train_cls_loss": float(df.iloc[-1]["train/cls_loss"]),
            "final_val_box_loss": float(df.iloc[-1]["val/box_loss"]),
            "final_val_cls_loss": float(df.iloc[-1]["val/cls_loss"])
        }

    return {
        "name": name,
        "elapsed_seconds": elapsed_s,
        "best_weights": str(best_pt),
        "onnx_export": str(onnx_dest),
        "metrics": metrics_summary
    }


def main():
    device = get_device()
    epochs = 10
    batch = 16
    imgsz = 640

    print("=" * 75)
    print("EdgePCB-QC: CONTROLLED FAIR COMPARISON TRAINING RUN")
    print(f"Target Epochs: {epochs} | Batch: {batch} | Imgsz: {imgsz}")
    print("=" * 75)

    # 1. Baseline Model (DeepPCB-only)
    baseline_res = train_single_model(
        name="pcb_smoke_experiment",
        data_yaml=BASELINE_DATA_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=42,
        device=device
    )

    # 2. Multi-Domain Model (Unified Dataset)
    multidomain_res = train_single_model(
        name="pcb_multidomain_experiment",
        data_yaml=MULTIDOMAIN_DATA_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        seed=42,
        device=device
    )

    # Comparison record
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device != "cpu" else "CPU",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        },
        "hyperparameters": {
            "base_architecture": "yolov8n.pt",
            "epochs": epochs,
            "batch_size": batch,
            "imgsz": imgsz,
            "seed": 42,
            "optimizer": "AdamW (auto)",
            "patience": 10,
            "augmentations": {
                "hsv_h": 0.015,
                "hsv_s": 0.70,
                "hsv_v": 0.40,
                "degrees": 0.0,
                "shear": 0.0,
                "translate": 0.10,
                "scale": 0.50,
                "fliplr": 0.50,
                "flipud": 0.50,
                "mosaic": 1.0
            }
        },
        "baseline_model": baseline_res,
        "multidomain_model": multidomain_res
    }

    out_json = BENCHMARKS_DIR / "fair_training_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 75)
    print("FAIR TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Summary saved to: {out_json}")
    print("=" * 75)


if __name__ == "__main__":
    main()
