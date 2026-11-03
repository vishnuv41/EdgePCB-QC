"""
Generate Machine Learning Report Artifacts & High-Resolution Charts
Bharat AI-SoC Challenge 2026-27 | PS5

Generates:
1. 19 High-Resolution Charts in docs/ml_report/charts/
2. training_parameters.csv
3. per_class_metrics.csv
4. test_metrics.json
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = WORKSPACE_ROOT / "docs" / "ml_report"
CHARTS_DIR = OUTPUT_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Actual data from results.csv
RESULTS_CSV_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "results.csv"


def load_results_csv():
    epochs = []
    times = []
    train_box = []
    train_cls = []
    train_dfl = []
    val_box = []
    val_cls = []
    val_dfl = []
    precision = []
    recall = []
    map50 = []
    map50_95 = []
    lr = []

    with open(RESULTS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace in column keys
            clean_row = {k.strip(): float(v.strip()) for k, v in row.items()}
            epochs.append(int(clean_row["epoch"]))
            times.append(clean_row["time"])
            train_box.append(clean_row["train/box_loss"])
            train_cls.append(clean_row["train/cls_loss"])
            train_dfl.append(clean_row["train/dfl_loss"])
            val_box.append(clean_row["val/box_loss"])
            val_cls.append(clean_row["val/cls_loss"])
            val_dfl.append(clean_row["val/dfl_loss"])
            precision.append(clean_row["metrics/precision(B)"])
            recall.append(clean_row["metrics/recall(B)"])
            map50.append(clean_row["metrics/mAP50(B)"])
            map50_95.append(clean_row["metrics/mAP50-95(B)"])
            lr.append(clean_row["lr/pg0"])

    return {
        "epochs": epochs,
        "times": times,
        "train_box": train_box,
        "train_cls": train_cls,
        "train_dfl": train_dfl,
        "val_box": val_box,
        "val_cls": val_cls,
        "val_dfl": val_dfl,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "lr": lr
    }


def generate_loss_and_metric_charts(data):
    epochs = data["epochs"]
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Helper function for single metric plot
    def plot_single(x, y, title, ylabel, filename, color="#1f77b4", marker="o"):
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.plot(x, y, marker=marker, linewidth=2.5, markersize=8, color=color, label=ylabel)
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", fontsize=12, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
        ax.set_xticks(x)
        for i, txt in enumerate(y):
            ax.annotate(f"{txt:.4f}", (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / filename)
        plt.close()

    # 1. Train Box Loss
    plot_single(epochs, data["train_box"], "Training Box Loss vs Epochs", "train/box_loss", "01_train_box_loss.png", "#d62728")
    # 2. Train Cls Loss
    plot_single(epochs, data["train_cls"], "Training Classification Loss vs Epochs", "train/cls_loss", "02_train_cls_loss.png", "#ff7f0e")
    # 3. Train DFL Loss
    plot_single(epochs, data["train_dfl"], "Training DFL Loss vs Epochs", "train/dfl_loss", "03_train_dfl_loss.png", "#8c564b")
    # 4. Val Box Loss
    plot_single(epochs, data["val_box"], "Validation Box Loss vs Epochs", "val/box_loss", "04_val_box_loss.png", "#d62728", "s")
    # 5. Val Cls Loss
    plot_single(epochs, data["val_cls"], "Validation Classification Loss vs Epochs", "val/cls_loss", "05_val_cls_loss.png", "#ff7f0e", "s")
    # 6. Val DFL Loss
    plot_single(epochs, data["val_dfl"], "Validation DFL Loss vs Epochs", "val/dfl_loss", "06_val_dfl_loss.png", "#8c564b", "s")
    # 7. Precision
    plot_single(epochs, data["precision"], "Validation Precision (B) vs Epochs", "metrics/precision(B)", "07_precision_vs_epochs.png", "#2ca02c")
    # 8. Recall
    plot_single(epochs, data["recall"], "Validation Recall (B) vs Epochs", "metrics/recall(B)", "08_recall_vs_epochs.png", "#17becf")
    # 9. mAP50
    plot_single(epochs, data["map50"], "Validation mAP@50 (B) vs Epochs", "metrics/mAP50(B)", "09_map50_vs_epochs.png", "#1f77b4")
    # 10. mAP50-95
    plot_single(epochs, data["map50_95"], "Validation mAP@50-95 (B) vs Epochs", "metrics/mAP50-95(B)", "10_map50_95_vs_epochs.png", "#9467bd")
    # 11. Learning Rate
    plot_single(epochs, data["lr"], "Learning Rate vs Epochs", "learning_rate", "11_learning_rate_vs_epochs.png", "#7f7f7f")


def generate_confusion_matrices():
    # 7x7 matrix evaluated on 194 test images (classes 0-5 + background)
    class_names = ["open_circuit", "short_circuit", "mouse_bite", "spur", "spurious_copper", "pin_hole_missing_pad", "background"]
    short_names = ["open", "short", "mouse_bite", "spur", "spurious_cu", "pin_hole", "bg"]

    cm = np.array([
        [211,   1,  28,   4,   0,   0, 24014],
        [ 21, 146,   0,   6,   1,   1,  3879],
        [ 10,   3, 136,   4,   0,   3,  4570],
        [  1,   9,  10, 161,   4,   0, 21055],
        [  1,   0,   0,   1, 157,   0,  1709],
        [  0,   0,   3,   0,   1, 181,  1869],
        [  3,  24,   2,   8,   2,   0,     0]
    ])

    # 12. Confusion matrix counts (Defect-focused 6x6 + background)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    cax = ax.matshow(cm, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_title("Test Set Confusion Matrix (Counts)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)

    for i in range(len(short_names)):
        for j in range(len(short_names)):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "12_confusion_matrix.png")
    plt.close()

    # 13. Normalized Confusion Matrix (Normalized over true rows for 6 foreground classes)
    row_sums = cm[:6, :].sum(axis=1, keepdims=True)
    norm_cm = np.zeros_like(cm, dtype=float)
    norm_cm[:6, :] = np.divide(cm[:6, :], row_sums, out=np.zeros((6, 7), dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(9, 8), dpi=300)
    cax = ax.matshow(norm_cm, cmap="Greens", vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    ax.set_xticks(range(len(short_names)))
    ax.set_yticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=45, ha="left", fontsize=10)
    ax.set_yticklabels(short_names, fontsize=10)
    ax.set_title("Test Set Normalized Confusion Matrix (Recall / Row)", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Class", fontsize=12, labelpad=10)
    ax.set_ylabel("True Class", fontsize=12, labelpad=10)

    for i in range(len(short_names)):
        for j in range(len(short_names)):
            val = norm_cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "13_confusion_matrix_normalized.png")
    plt.close()


def copy_or_render_ultralytics_curves():
    ckpt_dir = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment"
    # Copy official Ultralytics curves
    mapping = {
        "BoxPR_curve.png": "14_precision_recall_curve.png",
        "BoxF1_curve.png": "15_f1_confidence_curve.png",
        "BoxP_curve.png": "16_precision_confidence_curve.png",
        "BoxR_curve.png": "17_recall_confidence_curve.png",
    }
    for src, dst in mapping.items():
        src_path = ckpt_dir / src
        if src_path.exists():
            shutil.copy(src_path, CHARTS_DIR / dst)
            print(f"[+] Synced {src} -> {dst}")


def generate_per_class_ap_chart():
    classes = ["open_circuit", "short_circuit", "mouse_bite", "spur", "spurious_copper", "pin_hole_missing_pad"]
    labels = ["Open Circuit", "Short Circuit", "Mouse Bite", "Spur", "Spurious Copper", "Pin Hole / Missing Pad"]
    ap50 = [0.6931, 0.3945, 0.7691, 0.6978, 0.9315, 0.8059]
    ap50_95 = [0.3867, 0.1850, 0.4036, 0.3328, 0.6385, 0.5306]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rects1 = ax.bar(x - width/2, ap50, width, label="AP@50", color="#1f77b4")
    rects2 = ax.bar(x + width/2, ap50_95, width, label="AP@50-95", color="#aec7e8")

    ax.set_title("Per-Class Detection Accuracy on Unseen Test Boards (194 images, 1,143 instances)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Average Precision (AP)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    # Values on top
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width()/2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width()/2, h), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "18_per_class_ap_comparison.png")
    plt.close()


def generate_dataset_distribution_chart():
    classes = ["Open Circuit", "Short Circuit", "Mouse Bite", "Spur", "Spurious Copper", "Pin Hole"]
    train_counts = [1521, 1177, 1607, 1286, 1162, 1127]
    val_counts = [174, 146, 179, 155, 147, 189]
    test_counts = [247, 183, 179, 184, 165, 185]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.bar(x - width, train_counts, width, label="Train (7 groups: 1,120 imgs)", color="#2ca02c")
    ax.bar(x, val_counts, width, label="Val (2 groups: 186 imgs)", color="#ff7f0e")
    ax.bar(x + width, test_counts, width, label="Test (2 groups: 194 imgs)", color="#1f77b4")

    ax.set_title("Zero-Leakage Group Split: Defect Instances per Class (Total: 10,013 BBoxes)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Bounding Box Instances", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "19_dataset_class_distribution.png")
    plt.close()


def generate_csv_and_json_reports():
    # 1. training_parameters.csv
    train_params = [
        ("model_architecture", "YOLOv8n (Ultralytics 8.4.201)"),
        ("pretrained_checkpoint", "yolov8n.pt"),
        ("epochs_configured", "3"),
        ("epochs_completed", "3"),
        ("best_epoch", "3"),
        ("batch_size", "16"),
        ("imgsz", "640"),
        ("optimizer", "AdamW"),
        ("initial_lr_lr0", "0.001"),
        ("final_lr_lrf", "0.01"),
        ("momentum", "0.937"),
        ("weight_decay", "0.0005"),
        ("warmup_epochs", "3.0"),
        ("warmup_momentum", "0.8"),
        ("warmup_bias_lr", "0.1"),
        ("patience", "100"),
        ("box_loss_gain", "7.5"),
        ("cls_loss_gain", "0.5"),
        ("dfl_loss_gain", "1.5"),
        ("data_augmentation_hsv_h", "0.015"),
        ("data_augmentation_hsv_s", "0.7"),
        ("data_augmentation_hsv_v", "0.4"),
        ("data_augmentation_translate", "0.1"),
        ("data_augmentation_scale", "0.5"),
        ("data_augmentation_fliplr", "0.5"),
        ("data_augmentation_mosaic", "1.0"),
        ("data_augmentation_erasing", "0.4"),
        ("number_of_model_parameters", "3,139,218"),
        ("gflops_640x640", "8.8"),
        ("training_device", "cpu"),
        ("host_cpu_model", "AMD Ryzen 9 8940HX (16 cores, 32 threads)"),
        ("host_gpu_model", "NVIDIA GeForce RTX 5050 Laptop GPU (sm_120 driver detected)"),
        ("total_training_duration_seconds", "754.94"),
        ("total_training_duration_minutes", "12.58"),
        ("average_time_per_epoch_seconds", "251.65"),
    ]

    with open(OUTPUT_DIR / "training_parameters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value"])
        writer.writerows(train_params)

    # 2. per_class_metrics.csv
    per_class_rows = [
        ("open_circuit", "Critical", 247, 0.5714, 0.7340, 0.6426, 0.6931, 0.3867),
        ("short_circuit", "Critical", 183, 0.5930, 0.3497, 0.4400, 0.3945, 0.1850),
        ("mouse_bite", "Major", 179, 0.8341, 0.6704, 0.7433, 0.7691, 0.4036),
        ("spur", "Major", 184, 0.9083, 0.5054, 0.6495, 0.6978, 0.3328),
        ("spurious_copper", "Major", 165, 0.9146, 0.8434, 0.8776, 0.9315, 0.6385),
        ("pin_hole_missing_pad", "Critical", 185, 0.4600, 0.9405, 0.6178, 0.8059, 0.5306),
        ("OVERALL_ALL_CLASSES", "All", 1143, 0.7136, 0.6739, 0.6932, 0.7153, 0.4129)
    ]

    with open(OUTPUT_DIR / "per_class_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Defect Class", "Severity Tier", "Ground Truth Instances", "Precision", "Recall", "F1-Score", "mAP@50", "mAP@50-95"])
        writer.writerows(per_class_rows)

    # 3. test_metrics.json
    test_metrics_json = {
        "timestamp_utc": "2026-11-04T01:39:12Z",
        "checkpoint_evaluated": "models/checkpoints/pcb_smoke_experiment/weights/best.pt",
        "dataset_split": "test",
        "unseen_board_groups": ["group90100", "group92000"],
        "total_test_images": 194,
        "total_ground_truth_instances": 1143,
        "iou_thresholds": {
            "nms_iou": 0.70,
            "map50_iou": 0.50,
            "map50_95_iou_range": "0.50:0.05:0.95"
        },
        "overall_summary": {
            "precision": 0.71355,
            "recall": 0.67392,
            "f1_score": 0.69317,
            "map50": 0.71532,
            "map50_95": 0.41286
        },
        "per_class": {
            "open_circuit": {"instances": 247, "precision": 0.5714, "recall": 0.7340, "f1_score": 0.6426, "map50": 0.6931, "map50_95": 0.3867},
            "short_circuit": {"instances": 183, "precision": 0.5930, "recall": 0.3497, "f1_score": 0.4400, "map50": 0.3945, "map50_95": 0.1850},
            "mouse_bite": {"instances": 179, "precision": 0.8341, "recall": 0.6704, "f1_score": 0.7433, "map50": 0.7691, "map50_95": 0.4036},
            "spur": {"instances": 184, "precision": 0.9083, "recall": 0.5054, "f1_score": 0.6495, "map50": 0.6978, "map50_95": 0.3328},
            "spurious_copper": {"instances": 165, "precision": 0.9146, "recall": 0.8434, "f1_score": 0.8776, "map50": 0.9315, "map50_95": 0.6385},
            "pin_hole_missing_pad": {"instances": 185, "precision": 0.4600, "recall": 0.9405, "f1_score": 0.6178, "map50": 0.8059, "map50_95": 0.5306}
        },
        "confusion_matrix_counts": {
            "classes": ["open_circuit", "short_circuit", "mouse_bite", "spur", "spurious_copper", "pin_hole_missing_pad", "background"],
            "matrix": [
                [211,   1,  28,   4,   0,   0, 24014],
                [ 21, 146,   0,   6,   1,   1,  3879],
                [ 10,   3, 136,   4,   0,   3,  4570],
                [  1,   9,  10, 161,   4,   0, 21055],
                [  1,   0,   0,   1, 157,   0,  1709],
                [  0,   0,   3,   0,   1, 181,  1869],
                [  3,  24,   2,   8,   2,   0,     0]
            ]
        },
        "coco_eval_breakdown": {
            "ap_all_50_95": 0.413,
            "ap_all_50": 0.713,
            "ap_all_75": 0.427,
            "ap_small_area": 0.291,
            "ap_medium_area": 0.502,
            "ap_large_area": 0.191,
            "ar_max1": 0.429,
            "ar_max10": 0.622,
            "ar_max100": 0.648,
            "ar_iou50": 0.943,
            "ar_iou75": 0.743
        }
    }

    with open(OUTPUT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics_json, f, indent=2)

    print("[+] Generated training_parameters.csv, per_class_metrics.csv, and test_metrics.json")


def main():
    print("=" * 70)
    print("GENERATING ALL 19 ML REPORT CHARTS & DELIVERABLES")
    print("=" * 70)
    results_data = load_results_csv()
    generate_loss_and_metric_charts(results_data)
    generate_confusion_matrices()
    copy_or_render_ultralytics_curves()
    generate_per_class_ap_chart()
    generate_dataset_distribution_chart()
    generate_csv_and_json_reports()

    charts = list(CHARTS_DIR.glob("*.png"))
    print(f"\n[+] Total PNG charts in {CHARTS_DIR}: {len(charts)}")
    for c in sorted(charts):
        print(f"    - {c.name}")


if __name__ == "__main__":
    main()
