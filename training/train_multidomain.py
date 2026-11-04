"""
Multi-Domain YOLO Training Pipeline for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Trains on Unified Multi-Domain Dataset (DeepPCB + DsPCBSD+ + PCB-Defect V2):
- Incorporates realistic color HSV jitter, scaling, and flip augmentations.
- Avoids shear/rotation that would distort PCB trace dimensions.
- Evaluates on validation split after each epoch.
- Exports best checkpoint to models/checkpoints/pcb_multidomain_experiment/weights/best.pt.
- Exports edge ONNX model to models/onnx/pcb_multidomain_experiment.onnx.
"""

import os
import sys
import yaml
from pathlib import Path
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "data.yaml"
CHECKPOINTS_DIR = WORKSPACE_ROOT / "models" / "checkpoints"
ONNX_DIR = WORKSPACE_ROOT / "models" / "onnx"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)


def check_device():
    # PyTorch cu126 lacks sm_120 kernels on RTX 5050; use CPU
    return "cpu"


def run_training(epochs: int = 5, batch: int = 16, imgsz: int = 640):
    print("=" * 70)
    print("EdgePCB-QC MULTI-DOMAIN VISION MODEL TRAINING EXPERIMENT")
    print("=" * 70)

    device = check_device()
    experiment_name = "pcb_multidomain_experiment"

    print(f"[*] Configuration:")
    print(f"    - Data config:       {DATA_YAML}")
    print(f"    - Base model:        yolov8n.pt")
    print(f"    - Epochs:            {epochs}")
    print(f"    - Batch size:        {batch}")
    print(f"    - Image size:        {imgsz}")
    print(f"    - Device:            {device}")
    print(f"    - Augmentations:     HSV-H=0.02, HSV-S=0.6, HSV-V=0.4, FlipLR=0.5, FlipUD=0.5, Mosaic=1.0")

    model = YOLO("yolov8n.pt")

    # Hyperparameter dictionary with realistic PCB augmentations
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(CHECKPOINTS_DIR),
        name=experiment_name,
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        # Controlled augmentations preserving PCB geometry:
        hsv_h=0.02,       # Realistic solder mask & copper hue variations
        hsv_s=0.60,       # Saturation jitter bridging monochrome & color
        hsv_v=0.40,       # Exposure / lighting intensity jitter
        degrees=0.0,      # Zero rotation to preserve right-angle trace routing
        shear=0.0,        # Zero shear to maintain conductor spacing & trace width
        translate=0.10,   # Mild translation
        scale=0.50,       # Scale variations
        fliplr=0.50,      # Horizontal reflection
        flipud=0.50,      # Vertical reflection
        mosaic=1.0,       # Multi-scale mosaic
        patience=5        # Early stopping
    )

    exp_dir = CHECKPOINTS_DIR / experiment_name
    best_pt = exp_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = exp_dir / "weights" / "last.pt"

    print(f"\n[+] Multi-Domain Training Complete!")
    print(f"[+] Best weights saved to: {best_pt}")

    # Export ONNX
    onnx_dest = ONNX_DIR / f"{experiment_name}.onnx"
    try:
        print(f"[*] Exporting multi-domain model to ONNX format at {onnx_dest}...")
        exported_path = model.export(format="onnx", imgsz=imgsz, opset=17)
        if Path(exported_path).exists():
            import shutil
            shutil.copy2(exported_path, onnx_dest)
            print(f"[+] ONNX export successful: {onnx_dest}")
    except Exception as e:
        print(f"[!] ONNX export warning: {e}")

    return best_pt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    run_training(epochs=args.epochs, batch=args.batch)
