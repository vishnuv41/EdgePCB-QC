import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

WORKSPACE_ROOT = Path(".").resolve()
raw_img_path = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_YOLO" / "images" / "val" / "S_11833255.jpg"
raw_lbl_path = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_YOLO" / "labels" / "val" / "S_11833255.txt"
proc_lbl_path = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo" / "labels" / "val" / "dspcbsd_S_11833255.txt"

print(f"Raw image exists: {raw_img_path.exists()}")
print(f"Raw label exists: {raw_lbl_path.exists()}")
print(f"Proc label exists: {proc_lbl_path.exists()}")

img = cv2.imread(str(raw_img_path))
h, w = img.shape[:2]
print(f"Image dimensions: {w} x {h}")

# Read raw labels
raw_boxes = []
with open(raw_lbl_path, "r") as f:
    for line in f:
        p = line.strip().split()
        if p:
            raw_boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
print(f"Raw boxes count: {len(raw_boxes)}")
for b in raw_boxes:
    print(f"  Raw class {b[0]} (SP/spur): xc={b[1]:.4f}, yc={b[2]:.4f}, w={b[3]:.4f}, h={b[4]:.4f}")

# Read unified processed labels
proc_boxes = []
if proc_lbl_path.exists():
    with open(proc_lbl_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if p:
                proc_boxes.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
print(f"Processed boxes count: {len(proc_boxes)}")
for b in proc_boxes:
    print(f"  Canonical class {b[0]} (spur): xc={b[1]:.4f}, yc={b[2]:.4f}, w={b[3]:.4f}, h={b[4]:.4f}")

# Check checkpoints
baseline_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
multidomain_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"

print(f"\nBaseline checkpoint exists: {baseline_pt.exists()}")
print(f"Multi-domain checkpoint exists: {multidomain_pt.exists()}")

baseline_model = YOLO(str(baseline_pt))
multidomain_model = YOLO(str(multidomain_pt))

# Run inference across confidence thresholds [0.10, 0.25, 0.40] on CPU
conf_thresholds = [0.10, 0.25, 0.40]

print("\n--- BASELINE MODEL INFERENCE (CPU) ---")
for conf in conf_thresholds:
    res = baseline_model.predict(source=img, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    print(f"Conf {conf:.2f}: {len(res.boxes)} detections")
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        conf_val = float(box.conf[0].item())
        xywhn = box.xywhn[0].tolist()
        print(f"   Detected class={cls_id} conf={conf_val:.3f} box={xywhn}")

print("\n--- MULTI-DOMAIN MODEL INFERENCE (CPU) ---")
for conf in conf_thresholds:
    res = multidomain_model.predict(source=img, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
    print(f"Conf {conf:.2f}: {len(res.boxes)} detections")
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = multidomain_model.names[cls_id]
        conf_val = float(box.conf[0].item())
        xywhn = box.xywhn[0].tolist()
        print(f"   Detected class={cls_id} ({cls_name}) conf={conf_val:.3f} box={xywhn}")

# Also test native imgsz=256 or padding
print("\n--- MULTI-DOMAIN MODEL WITH imgsz=256 (CPU) ---")
for conf in conf_thresholds:
    res = multidomain_model.predict(source=img, conf=conf, imgsz=256, device="cpu", verbose=False)[0]
    print(f"Conf {conf:.2f} (imgsz=256): {len(res.boxes)} detections")
    for box in res.boxes:
        cls_id = int(box.cls[0].item())
        cls_name = multidomain_model.names[cls_id]
        conf_val = float(box.conf[0].item())
        print(f"   Detected class={cls_id} ({cls_name}) conf={conf_val:.3f}")

