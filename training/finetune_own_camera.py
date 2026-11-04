"""
EdgePCB-QC: Domain 5 (Own-Camera) Fine-Tuning Pipeline
Bharat AI-SoC Challenge 2026-27 | PS5

Fine-tunes the unified vision detector on custom edge camera images
captured via Raspberry Pi Camera Module 3 or bench optical setup.
Handles empty dataset gracefully by reporting NOT RUN without failing.
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

DEFAULT_BASE_CHECKPOINT = (
    WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
)
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "models" / "checkpoints" / "own_camera_finetune"
OWN_CAMERA_DIR = WORKSPACE_ROOT / "datasets" / "own_camera"


def get_image_count(folder: Path) -> int:
    """Counts valid image files in a folder."""
    if not folder.exists():
        return 0
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for f in folder.iterdir() if f.is_file() and f.suffix.lower() in valid_exts)


def create_own_camera_yaml(data_dir: Path) -> Path:
    """Generates YOLO dataset YAML for Domain 5."""
    train_dir = data_dir / "images" / "train"
    val_dir = data_dir / "images" / "val"
    
    yaml_dict = {
        "path": str(data_dir.resolve()),
        "train": "images/train",
        "val": "images/val" if get_image_count(val_dir) > 0 else "images/train",
        "nc": 6,
        "names": {
            0: "open_circuit",
            1: "short_circuit",
            2: "mouse_bite",
            3: "spur",
            4: "spurious_copper",
            5: "pin_hole_missing_pad"
        }
    }
    
    yaml_path = data_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO detector on Domain 5 (own_camera)")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu", help="Device (0 for GPU or cpu)")
    parser.add_argument("--base-checkpoint", type=str, default=str(DEFAULT_BASE_CHECKPOINT), help="Base checkpoint path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("EdgePCB-QC: Domain 5 (Own Camera) Fine-Tuning Pipeline")
    print("=" * 70)

    train_img_dir = OWN_CAMERA_DIR / "images" / "train"
    val_img_dir = OWN_CAMERA_DIR / "images" / "val"
    train_count = get_image_count(train_img_dir)
    val_count = get_image_count(val_img_dir)

    print(f"[*] Dataset Directory : {OWN_CAMERA_DIR}")
    print(f"[*] Training Images   : {train_count}")
    print(f"[*] Validation Images : {val_count}")

    if train_count == 0:
        print("\n" + "!" * 70)
        print("[!] NOTICE: Domain 5 (own_camera) contains 0 training images.")
        print("[!] Status: NOT RUN (Awaiting physical Raspberry Pi Camera Module 3 captures).")
        print("[!] Refer to datasets/own_camera/README.md for capture and annotation protocol.")
        print("!" * 70 + "\n")
        return 0

    base_ckpt = Path(args.base_checkpoint)
    if not base_ckpt.exists():
        print(f"[!] Base checkpoint not found at: {base_ckpt}")
        print("[!] Falling back to pretrained yolo11n.pt")
        base_ckpt = "yolo11n.pt"

    from ultralytics import YOLO

    yaml_path = create_own_camera_yaml(OWN_CAMERA_DIR)
    print(f"[*] Dataset YAML      : {yaml_path}")
    print(f"[*] Base Checkpoint   : {base_ckpt}")
    print(f"[*] Epochs            : {args.epochs}")
    print(f"[*] Batch Size        : {args.batch}")
    print(f"[*] Target Device     : {args.device}")

    model = YOLO(str(base_ckpt))
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(Path(args.output_dir).parent),
        name=Path(args.output_dir).name,
        lr0=0.001,
        lrf=0.01,
        exist_ok=True,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("Domain 5 Fine-Tuning Completed Successfully!")
    print(f"Best Weights: {Path(args.output_dir) / 'weights' / 'best.pt'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
