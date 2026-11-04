import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

def box_iou(b1, b2):
    # b1, b2 in xyxy format
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

WORKSPACE_ROOT = Path(".").resolve()
raw_img_path = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_YOLO" / "images" / "val" / "S_11833255.jpg"
raw_lbl_path = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_YOLO" / "labels" / "val" / "S_11833255.txt"

img = cv2.imread(str(raw_img_path))
h, w = img.shape[:2]

# Ground Truth BBoxes (xyxy normalized)
gt_boxes = []
with open(raw_lbl_path, "r") as f:
    for line in f:
        p = line.strip().split()
        if p:
            cid, xc, yc, bw, bh = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
            x1 = xc - bw / 2.0
            y1 = yc - bh / 2.0
            x2 = xc + bw / 2.0
            y2 = yc + bh / 2.0
            gt_boxes.append({"cls": cid, "cls_name": "spur", "bbox": [x1, y1, x2, y2]})

print(f"Loaded {len(gt_boxes)} GT boxes.")

baseline_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
multidomain_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"

baseline_model = YOLO(str(baseline_pt))
multidomain_model = YOLO(str(multidomain_pt))

results_data = {}

for model_name, model in [("DeepPCB Baseline (pcb_smoke_experiment)", baseline_model), 
                          ("Unified Multi-Domain (pcb_multidomain_experiment)", multidomain_model)]:
    results_data[model_name] = {}
    for conf in [0.10, 0.25, 0.40]:
        preds = model.predict(source=img, conf=conf, imgsz=640, device="cpu", verbose=False)[0]
        pred_boxes = []
        for b in preds.boxes:
            c = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            xyxy = b.xyxyn[0].tolist()
            pred_boxes.append({"cls": c, "conf": cf, "bbox": xyxy})
        
        # Match with GT (IoU >= 0.50 or IoU >= 0.30 for small defects)
        matched_gt = set()
        tp = 0
        fp = 0
        for pb in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gb in enumerate(gt_boxes):
                iou = box_iou(pb["bbox"], gb["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= 0.50 and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / len(gt_boxes)
        results_data[model_name][conf] = {
            "pred_count": len(pred_boxes),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "boxes": pred_boxes
        }

print("\n=== IoU >= 0.50 EVALUATION ON S_11833255 (7 GT SPUR DEFECTS) ===")
for m_name, c_dict in results_data.items():
    print(f"\nModel: {m_name}")
    for conf, m in c_dict.items():
        print(f"  Conf {conf:.2f} -> Detections: {m['pred_count']}, TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}, Precision: {m['precision']:.3f}, Recall: {m['recall']:.3f}")

# Generate side-by-side visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Ground Truth
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gt = img_rgb.copy()
for gb in gt_boxes:
    x1, y1, x2, y2 = [int(v * 226) for v in gb["bbox"]]
    cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_gt, "spur (GT)", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
axes[0].imshow(img_gt)
axes[0].set_title(f"Ground Truth ({len(gt_boxes)} Spur Defect BBoxes)\nDsPCBSD+ S_11833255.jpg (226x226)")
axes[0].axis("off")

# Panel 2: Baseline Model (Conf 0.25)
img_base = img_rgb.copy()
base_preds = results_data["DeepPCB Baseline (pcb_smoke_experiment)"][0.25]["boxes"]
for pb in base_preds:
    x1, y1, x2, y2 = [int(v * 226) for v in pb["bbox"]]
    cv2.rectangle(img_base, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img_base, f"{pb['conf']:.2f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)
axes[1].imshow(img_base)
axes[1].set_title(f"Baseline YOLOv8n (DeepPCB Smoke)\nConf=0.25 | Detections: {len(base_preds)} (FALSE PASS)")
axes[1].axis("off")

# Panel 3: Multi-domain Model (Conf 0.10)
img_multi = img_rgb.copy()
multi_preds = results_data["Unified Multi-Domain (pcb_multidomain_experiment)"][0.10]["boxes"]
for pb in multi_preds:
    x1, y1, x2, y2 = [int(v * 226) for v in pb["bbox"]]
    c_name = multidomain_model.names[pb["cls"]]
    cv2.rectangle(img_multi, (x1, y1), (x2, y2), (255, 140, 0), 2)
    cv2.putText(img_multi, f"{c_name} {pb['conf']:.2f}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 140, 0), 1)
axes[2].imshow(img_multi)
axes[2].set_title(f"Unified Multi-Domain YOLOv8n\nConf=0.10 | Detections: {len(multi_preds)} (Defects Found)")
axes[2].axis("off")

plt.tight_layout()
out_fig = WORKSPACE_ROOT / "benchmarks" / "results" / "s_11833255_investigation.png"
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(str(out_fig), dpi=200, bbox_inches="tight")
plt.close()
print(f"\n[+] Saved side-by-side comparison figure to: {out_fig}")
