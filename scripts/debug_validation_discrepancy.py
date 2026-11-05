"""
Verification of Coordinate Scaling Fix
"""

from pathlib import Path
import sys
import cv2
from PIL import Image

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from ultralytics import YOLO
from scripts.evaluate_candidate_model import load_yolo_labels, match_boxes
from collections import defaultdict

m = YOLO('models/checkpoints/exp_c_hard_negatives/weights/best.pt')
val_img_dir = Path('datasets/processed/unified_yolo/images/val')
val_lbl_dir = Path('datasets/processed/unified_yolo/labels/val')
val_imgs = sorted(list(val_img_dir.glob('*.jpg')))

groups = defaultdict(lambda: {'gt': 0, 'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})

for p in val_imgs:
    fname = p.name
    if fname.startswith("deeppcb_"):
        ds = "DeepPCB"
    elif fname.startswith("pcbdefect_"):
        ds = "PCB-Defect V2"
    elif fname.startswith("dspcbsd_"):
        ds = "DsPCBSD+"
    else:
        ds = "Unknown"
        
    # Get actual image dimensions
    with Image.open(p) as im:
        img_w, img_h = im.size
        
    gt = load_yolo_labels(val_lbl_dir / (p.stem + '.txt'), img_w, img_h)
    res = m.predict(source=str(p), conf=0.25, imgsz=640, device=0, verbose=False)[0]
    preds = [{'class_id': int(b.cls[0].cpu().numpy()), 'confidence': float(b.conf[0].cpu().numpy()), 'bbox': b.xyxy[0].cpu().numpy().tolist()} for b in res.boxes]
    matches, unmatched_gt, unmatched_pred = match_boxes(gt, preds, 0.50)
    
    g = groups[ds]
    g['count'] += 1
    g['gt'] += len(gt)
    g['tp'] += len(matches)
    g['fn'] += len(unmatched_gt)
    g['fp'] += len(unmatched_pred)

print(f"Total validation images: {len(val_imgs)}")
total_tp = 0
total_fp = 0
total_fn = 0
total_gt = 0

for ds, g in groups.items():
    p = g['tp'] / (g['tp'] + g['fp']) if (g['tp'] + g['fp']) > 0 else 0
    r = g['tp'] / (g['tp'] + g['fn']) if (g['tp'] + g['fn']) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    cnt = g['count']
    print(f"=== {ds} ({cnt} imgs) ===")
    print(f"GT: {g['gt']}, TP: {g['tp']}, FP: {g['fp']}, FN: {g['fn']}")
    print(f"Precision: {p*100:.2f}%, Recall: {r*100:.2f}%, F1: {f1:.4f}")
    total_tp += g['tp']
    total_fp += g['fp']
    total_fn += g['fn']
    total_gt += g['gt']

macro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
macro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0
print("\n=== CORRECTED OVERALL VALIDATION SUMMARY ===")
print(f"Total GT: {total_gt}, Total TP: {total_tp}, Total FP: {total_fp}, Total FN: {total_fn}")
print(f"Overall Precision: {macro_p*100:.2f}%, Overall Recall: {macro_r*100:.2f}%, Overall F1: {macro_f1:.4f}")
