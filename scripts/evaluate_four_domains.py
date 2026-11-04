"""
Domain-Specific Evaluation for EdgePCB-QC Across 4 Distinct Industrial Splits
Bharat AI-SoC Challenge 2026-27 | PS5

Evaluates both DeepPCB Baseline and Unified Multi-Domain models across:
1. DeepPCB untouched test (194 images, 1,143 bboxes) - Monochrome high-contrast
2. DsPCBSD+ validation (1,444 images) - Industrial colour patches from manufacturing line
3. PCB-Defect V2 validation (495 native crops) - High-resolution tiled colour boards
4. PCB-Defect V2 external colour test (390 native crops) - Unseen colour boards (68, 80, 81, 82)

Calculates:
- Per-class and macro Precision, Recall, F1, and IoU >= 0.50 matching
- Confusion matrices for each domain split
- Saves structured results to benchmarks/results/domain_specific_evaluation.json
- Generates side-by-side comparative charts
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo"
BENCHMARKS_DIR = WORKSPACE_ROOT / "benchmarks" / "results"

CANONICAL_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole_missing_pad"
}


def get_device():
    return 0 if torch.cuda.is_available() else "cpu"


def box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def evaluate_split(model, img_paths, lbl_dir, split_name, conf_thresh=0.25, imgsz=640, device=0):
    print(f"\n=======================================================")
    print(f"EVALUATING DOMAIN: {split_name} ({len(img_paths)} images)")
    print(f"=======================================================")

    gt_counts = Counter()
    tp_counts = Counter()
    fp_counts = Counter()
    confusion = defaultdict(lambda: Counter())

    total_images = len(img_paths)
    t0 = time.time()

    for idx, img_p in enumerate(img_paths, 1):
        if idx % 500 == 0 or idx == total_images:
            elapsed = time.time() - t0
            print(f"  [{idx}/{total_images}] images evaluated ({elapsed:.1f}s)...")

        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        gt_boxes = []
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = max(0.0, xc - bw / 2.0)
                        y1 = max(0.0, yc - bh / 2.0)
                        x2 = min(1.0, xc + bw / 2.0)
                        y2 = min(1.0, yc + bh / 2.0)
                        gt_boxes.append({"cls": cid, "bbox": [x1, y1, x2, y2]})
                        gt_counts[cid] += 1

        # Run inference
        res = model.predict(source=str(img_p), conf=conf_thresh, imgsz=imgsz, device=device, verbose=False)[0]
        pred_boxes = []
        for b in res.boxes:
            c = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            xyxy = b.xyxyn[0].tolist()
            pred_boxes.append({"cls": c, "conf": cf, "bbox": xyxy})

        # Match predictions with ground truth
        matched_gt = set()
        for pb in pred_boxes:
            pred_cls = pb["cls"]
            best_iou = 0.0
            best_gt_idx = -1
            for g_idx, gb in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = box_iou(pb["bbox"], gb["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= 0.50 and best_gt_idx >= 0:
                gt_cls = gt_boxes[best_gt_idx]["cls"]
                matched_gt.add(best_gt_idx)
                confusion[gt_cls][pred_cls] += 1
                if pred_cls == gt_cls:
                    tp_counts[pred_cls] += 1
                else:
                    fp_counts[pred_cls] += 1
            else:
                fp_counts[pred_cls] += 1
                confusion["background"][pred_cls] += 1

        # Count false negatives (unmatched ground truth)
        for g_idx, gb in enumerate(gt_boxes):
            if g_idx not in matched_gt:
                confusion[gb["cls"]]["background"] += 1

    # Compile metrics
    per_class = {}
    tot_tp, tot_fp, tot_fn = 0, 0, 0

    for cid, cname in CANONICAL_CLASSES.items():
        n_gt = gt_counts[cid]
        tp = tp_counts[cid]
        fp = fp_counts[cid]
        fn = n_gt - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / n_gt if n_gt > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[cname] = {
            "ground_truth": n_gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4)
        }
        tot_tp += tp
        tot_fp += fp
        tot_fn += fn

    macro_p = np.mean([m["precision"] for m in per_class.values()])
    macro_r = np.mean([m["recall"] for m in per_class.values()])
    macro_f1 = (2 * macro_p * macro_r) / (macro_p + macro_r) if (macro_p + macro_r) > 0 else 0.0

    overall = {
        "images": total_images,
        "total_ground_truth": sum(gt_counts.values()),
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "total_fn": tot_fn,
        "precision": round(tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else 0.0, 4),
        "recall": round(tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else 0.0, 4),
        "f1": round(2 * tot_tp / (2 * tot_tp + tot_fp + tot_fn) if (2 * tot_tp + tot_fp + tot_fn) > 0 else 0.0, 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4)
    }

    # Format confusion matrix
    formatted_conf = {}
    for k, row in confusion.items():
        k_str = CANONICAL_CLASSES.get(k, str(k))
        formatted_conf[k_str] = {CANONICAL_CLASSES.get(ck, str(ck)): count for ck, count in row.items()}

    print(f"\n--- {split_name} Results ---")
    print(f"  Total GT Defect Boxes: {overall['total_ground_truth']}")
    print(f"  Precision:             {overall['precision'] * 100:.2f}%")
    print(f"  Recall:                {overall['recall'] * 100:.2f}%")
    print(f"  F1 Score:              {overall['f1']:.4f}")

    return {
        "overall": overall,
        "per_class": per_class,
        "confusion_matrix": formatted_conf
    }


def generate_comparative_plots(eval_results):
    domains = [
        ("deeppcb_untouched_test", "DeepPCB (Monochrome)"),
        ("dspcbsd_validation", "DsPCBSD+ (Colour Patches)"),
        ("pcb_defect_v2_validation", "PCB-Defect V2 (Val Tiled)"),
        ("pcb_defect_v2_external_colour", "PCB-Defect V2 (Ext Colour)")
    ]

    domain_names = [d[1] for d in domains]
    base_f1 = [eval_results["baseline_model"][d[0]]["overall"]["f1"] for d in domains]
    multi_f1 = [eval_results["multidomain_model"][d[0]]["overall"]["f1"] for d in domains]

    base_rec = [eval_results["baseline_model"][d[0]]["overall"]["recall"] * 100 for d in domains]
    multi_rec = [eval_results["multidomain_model"][d[0]]["overall"]["recall"] * 100 for d in domains]

    base_prec = [eval_results["baseline_model"][d[0]]["overall"]["precision"] * 100 for d in domains]
    multi_prec = [eval_results["multidomain_model"][d[0]]["overall"]["precision"] * 100 for d in domains]

    x = np.arange(len(domain_names))
    width = 0.35

    plt.figure(figsize=(14, 5))

    # Subplot 1: F1 Score
    plt.subplot(1, 2, 1)
    plt.bar(x - width/2, base_f1, width, label='DeepPCB Baseline', color='#708090', alpha=0.85)
    plt.bar(x + width/2, multi_f1, width, label='Unified Multi-Domain', color='#2E8B57', alpha=0.9)
    plt.ylabel('F1 Score (IoU >= 0.50)')
    plt.title('F1 Score by Domain: Baseline vs Multi-Domain')
    plt.xticks(x, domain_names, rotation=15, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()

    # Subplot 2: Recall
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, base_rec, width, label='DeepPCB Baseline', color='#B22222', alpha=0.85)
    plt.bar(x + width/2, multi_rec, width, label='Unified Multi-Domain', color='#1E90FF', alpha=0.9)
    plt.ylabel('Defect Recall (%)')
    plt.title('Defect Recall by Domain (Sensitivity)')
    plt.xticks(x, domain_names, rotation=15, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.legend()

    plt.tight_layout()
    chart_p = BENCHMARKS_DIR / "domain_specific_comparison.png"
    plt.savefig(chart_p, dpi=200)
    plt.close()
    print(f"[+] Comparative evaluation chart saved to: {chart_p}")


def run_full_evaluation():
    device = get_device()
    print(f"[*] Evaluation Device: {device} ({torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'})")

    test_img_dir = DATA_ROOT / "images" / "test"
    test_lbl_dir = DATA_ROOT / "labels" / "test"
    deeppcb_test_imgs = sorted(list(test_img_dir.glob("deeppcb_*.jpg")))

    val_img_dir = DATA_ROOT / "images" / "val"
    val_lbl_dir = DATA_ROOT / "labels" / "val"
    dspcbsd_val_imgs = sorted(list(val_img_dir.glob("dspcbsd_*.jpg")))
    pcbdefect_val_imgs = sorted(list(val_img_dir.glob("pcbdefect_*.jpg")))

    ext_img_dir = DATA_ROOT / "images" / "external_color_test"
    ext_lbl_dir = DATA_ROOT / "labels" / "external_color_test"
    pcbdefect_ext_imgs = sorted(list(ext_img_dir.glob("pcbdefect_*.jpg")))

    splits = {
        "deeppcb_untouched_test": (deeppcb_test_imgs, test_lbl_dir, "DeepPCB Untouched Test (Monochrome)"),
        "dspcbsd_validation": (dspcbsd_val_imgs, val_lbl_dir, "DsPCBSD+ Validation (Colour Patches)"),
        "pcb_defect_v2_validation": (pcbdefect_val_imgs, val_lbl_dir, "PCB-Defect V2 Validation (Tiled Native)"),
        "pcb_defect_v2_external_colour": (pcbdefect_ext_imgs, ext_lbl_dir, "PCB-Defect V2 External Colour Boards (Untouched)")
    }

    models_to_eval = [
        ("baseline_model", WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"),
        ("multidomain_model", WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt")
    ]

    all_results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "conf_threshold": 0.25,
        "iou_threshold": 0.50,
        "hardware": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device != "cpu" else "CPU"
        }
    }

    for model_key, model_path in models_to_eval:
        print("\n" + "#" * 70)
        print(f"LOADING MODEL: {model_key} ({model_path.name})")
        print("#" * 70)
        model = YOLO(str(model_path))
        all_results[model_key] = {}

        for split_key, (imgs, lbl_dir, sname) in splits.items():
            res = evaluate_split(model, imgs, lbl_dir, sname, conf_thresh=0.25, imgsz=640, device=device)
            all_results[model_key][split_key] = res

        # Domain 5: Own Camera
        own_cam_dir = WORKSPACE_ROOT / "datasets" / "own_camera"
        own_cam_val = own_cam_dir / "images" / "val"
        own_cam_imgs = [p for p in own_cam_val.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}] if own_cam_val.exists() else []
        if len(own_cam_imgs) == 0:
            print("\n=======================================================")
            print("DOMAIN 5 (own_camera): NOT RUN (Dataset empty - hardware pending)")
            print("=======================================================")
            all_results[model_key]["own_camera"] = {
                "status": "NOT RUN",
                "reason": "Dataset empty - hardware pending (Raspberry Pi Camera Module 3)",
                "image_count": 0
            }
        else:
            res_cam = evaluate_split(model, own_cam_imgs, own_cam_dir / "labels" / "val", "Domain 5 (Own Camera)", conf_thresh=0.25, imgsz=640, device=device)
            all_results[model_key]["own_camera"] = res_cam

    out_json = BENCHMARKS_DIR / "domain_specific_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[+] Full domain-specific evaluation report saved to: {out_json}")

    # Generate charts
    generate_comparative_plots(all_results)
    return all_results


if __name__ == "__main__":
    run_full_evaluation()
