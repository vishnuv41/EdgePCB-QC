"""
YOLO Training Pipeline for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Trains lightweight edge detector on leak-free PCB defect splits:
- Handles auto-detection of device (CPU fallback if GPU sm_120 kernel is missing)
- Logs training metrics to benchmarks/results/
- Exports best checkpoint to models/checkpoints/ and ONNX to models/onnx/
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_YAML = WORKSPACE_ROOT / "datasets" / "processed" / "deeppcb_yolo" / "data.yaml"
CHECKPOINTS_DIR = WORKSPACE_ROOT / "models" / "checkpoints"
ONNX_DIR = WORKSPACE_ROOT / "models" / "onnx"

CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
ONNX_DIR.mkdir(parents=True, exist_ok=True)


def check_cuda_capability() -> str:
    """Tests if PyTorch can physically execute a CUDA kernel on the installed GPU."""
    if not torch.cuda.is_available():
        return "cpu"
    try:
        t = torch.zeros(1, device="cuda")
        _ = t + 1
        return "0"
    except Exception as e:
        print(f"[i] CUDA detected ({torch.cuda.get_device_name(0)}) but kernel execution unavailable on sm_120: {e}")
        print("[i] Falling back to CPU for training execution.")
        return "cpu"


def train_model(
    data_yaml: Path = DEFAULT_DATA_YAML,
    model_name: str = "yolov8n.pt",
    epochs: int = 5,
    imgsz: int = 640,
    batch: int = 8,
    device: str = None,
    experiment_name: str = "pcb_smoke_run"
):
    print("=" * 65)
    print("EdgePCB-QC VISION MODEL TRAINING PIPELINE")
    print("=" * 65)

    if device is None:
        device = check_cuda_capability()

    print(f"[*] Configuration:")
    print(f"    - Data config:       {data_yaml}")
    print(f"    - Base model:        {model_name}")
    print(f"    - Epochs:            {epochs}")
    print(f"    - Image size:        {imgsz}")
    print(f"    - Batch size:        {batch}")
    print(f"    - Execution device:  {device}")
    print(f"    - Output project:    {CHECKPOINTS_DIR}")

    model = YOLO(model_name)

    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(CHECKPOINTS_DIR),
        name=experiment_name,
        save=True,
        plots=True,
        verbose=True,
        workers=0  # Safe for Windows multiprocessing
    )

    best_pt = CHECKPOINTS_DIR / experiment_name / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = CHECKPOINTS_DIR / experiment_name / "weights" / "last.pt"

    print(f"\n[+] Training complete! Best weights saved to: {best_pt}")

    # Export ONNX for edge deployment
    onnx_dest = ONNX_DIR / f"{experiment_name}.onnx"
    try:
        print(f"[*] Exporting model to ONNX format at {onnx_dest}...")
        exported_path = model.export(format="onnx", imgsz=imgsz, opset=17)
        if Path(exported_path).exists():
            import shutil
            shutil.copy2(exported_path, onnx_dest)
            print(f"[+] ONNX export successful: {onnx_dest}")
    except Exception as e:
        print(f"[!] ONNX export warning: {e}")

    return best_pt, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--name", type=str, default="pcb_smoke_run", help="Experiment name")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu or 0)")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        experiment_name=args.name,
        device=args.device
    )
