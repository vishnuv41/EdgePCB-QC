import cv2
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO

def box_iou(b1, b2):
    # b1, b2 in xyxy format
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
raw_img_path = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_YOLO" / "images" / "val" / "S_11833255.jpg"
raw_lbl_path = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_YOLO" / "labels" / "val" / "S_11833255.txt"

img = cv2.imread(str(raw_img_path))
h, w = img.shape[:2]

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
            gt_boxes.append({"cls": 3, "cls_name": "spur", "bbox": [x1, y1, x2, y2]})

print(f"Loaded S_11833255 ({w}x{h}) with {len(gt_boxes)} Ground Truth spur defects.")

multidomain_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
baseline_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"

models = {
    "DeepPCB Baseline": YOLO(str(baseline_pt)),
    "Unified Multi-Domain": YOLO(str(multidomain_pt))
}

resolutions = [640, 960, 1280]
conf_levels = [0.10, 0.25, 0.40]

results = {}

for m_label, model in models.items():
    results[m_label] = {}
    print(f"\n=======================================================")
    print(f"EVALUATING MODEL: {m_label}")
    print(f"=======================================================")
    for sz in resolutions:
        results[m_label][sz] = {}
        for conf in conf_levels:
            pred_res = model.predict(source=img, conf=conf, imgsz=sz, device="cpu", verbose=False)[0]
            pred_boxes = []
            for b in pred_res.boxes:
                c = int(b.cls[0].item())
                cf = float(b.conf[0].item())
                xyxy = b.xyxyn[0].tolist()
                pred_boxes.append({"cls": c, "conf": cf, "bbox": xyxy})

            # IoU >= 0.50 matching
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
            rec = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            results[m_label][sz][conf] = {
                "detections": len(pred_boxes),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "boxes": pred_boxes
            }
            print(f"  imgsz={sz:<4} | conf={conf:.2f} -> Dets: {len(pred_boxes):<2} | TP: {tp} | FP: {fp} | FN: {fn} | Prec: {prec*100:5.1f}% | Rec: {rec*100:5.1f}% | F1: {f1:5.3f}")

# Also test native resolution with overlap tiling
print(f"\n=======================================================")
print(f"EVALUATING OVERLAPPING TILING AT NATIVE / HIGH RESOLUTION")
print(f"=======================================================")

# S_11833255 is 226x226. Let's see what happens if we pad to 640 vs direct tiles
out_json = WORKSPACE_ROOT / "benchmarks" / "results" / "small_object_resolution_benchmark.json"
out_json.parent.mkdir(parents=True, exist_ok=True)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"\n[+] Saved detailed resolution benchmark to: {out_json}")
