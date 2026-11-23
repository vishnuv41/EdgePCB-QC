"""
Regression Test Suite: Coordinate Scaling Across Non-640 Image Dimensions
EdgePCB-QC — Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Verifies load_yolo_labels scales normalized coordinates to actual image dimensions:
   - 226x226 image (DsPCBSD+ typical crop)
   - 300x250 image (rectangular non-square patch)
   - 640x640 image (DeepPCB and PCB-Defect V2 tile)
2. Verifies IoU matching on non-640 images with controlled ground truth and prediction boxes.
3. Tests regression: proves hardcoding 640x640 on a 226x226 image fails (IoU=0),
   while using actual dimensions produces accurate TP/FP/FN matching.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from scripts.evaluate_candidate_model import load_yolo_labels, match_boxes
from inspection.tiling import compute_iou


def test_load_yolo_labels_scales_to_actual_dimensions_640():
    """Verifies 640x640 image coordinate conversion."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f:
        # class_id, xc, yc, w, h
        f.write("0 0.5 0.5 0.1 0.2\n")
        lbl_path = Path(f.name)

    try:
        boxes = load_yolo_labels(lbl_path, 640, 640)
        assert len(boxes) == 1
        b = boxes[0]
        assert b["class_id"] == 0
        assert pytest.approx(b["bbox"][0], 0.01) == 288.0
        assert pytest.approx(b["bbox"][1], 0.01) == 256.0
        assert pytest.approx(b["bbox"][2], 0.01) == 352.0
        assert pytest.approx(b["bbox"][3], 0.01) == 384.0
        assert pytest.approx(b["width"], 0.01) == 64.0
        assert pytest.approx(b["height"], 0.01) == 128.0
    finally:
        if lbl_path.exists():
            lbl_path.unlink()


def test_load_yolo_labels_scales_to_actual_dimensions_226():
    """Verifies 226x226 image coordinate conversion."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f:
        f.write("1 0.5 0.5 0.1 0.2\n")
        lbl_path = Path(f.name)

    try:
        boxes = load_yolo_labels(lbl_path, 226, 226)
        assert len(boxes) == 1
        b = boxes[0]
        assert b["class_id"] == 1
        assert pytest.approx(b["bbox"][0], 0.01) == 101.7
        assert pytest.approx(b["bbox"][1], 0.01) == 90.4
        assert pytest.approx(b["bbox"][2], 0.01) == 124.3
        assert pytest.approx(b["bbox"][3], 0.01) == 135.6
        assert pytest.approx(b["width"], 0.01) == 22.6
        assert pytest.approx(b["height"], 0.01) == 45.2
    finally:
        if lbl_path.exists():
            lbl_path.unlink()


def test_load_yolo_labels_scales_to_actual_dimensions_300_250():
    """Verifies non-square 300x250 image coordinate conversion."""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f:
        f.write("2 0.5 0.5 0.1 0.2\n")
        lbl_path = Path(f.name)

    try:
        # width = 300, height = 250
        boxes = load_yolo_labels(lbl_path, 300, 250)
        assert len(boxes) == 1
        b = boxes[0]
        assert b["class_id"] == 2
        assert pytest.approx(b["bbox"][0], 0.01) == 135.0  # (0.5 - 0.05) * 300
        assert pytest.approx(b["bbox"][1], 0.01) == 100.0  # (0.5 - 0.10) * 250
        assert pytest.approx(b["bbox"][2], 0.01) == 165.0  # (0.5 + 0.05) * 300
        assert pytest.approx(b["bbox"][3], 0.01) == 150.0  # (0.5 + 0.10) * 250
        assert pytest.approx(b["width"], 0.01) == 30.0
        assert pytest.approx(b["height"], 0.01) == 50.0
    finally:
        if lbl_path.exists():
            lbl_path.unlink()


def test_controlled_iou_matching_non_640():
    """Verifies IoU matching and TP/FP/FN tallying on 226x226 and 300x250 images."""
    # Ground truth in 226x226 space
    gt_boxes = [
        {"class_id": 0, "bbox": [100.0, 100.0, 140.0, 140.0]},  # 40x40 box
        {"class_id": 1, "bbox": [20.0, 20.0, 60.0, 60.0]}
    ]

    # Pred 1 matches GT 0 with controlled IoU > 0.50
    # Pred 2 is a false positive
    pred_boxes = [
        {"class_id": 0, "confidence": 0.85, "bbox": [105.0, 105.0, 145.0, 145.0]},  # shifted by 5px
        {"class_id": 3, "confidence": 0.60, "bbox": [180.0, 180.0, 210.0, 210.0]}
    ]

    # Compute controlled IoU
    iou = compute_iou(gt_boxes[0]["bbox"], pred_boxes[0]["bbox"])
    # intersection: 35x35 = 1225. Union: 1600 + 1600 - 1225 = 1975. IoU = 1225 / 1975 = 0.6202
    assert iou > 0.50

    matches, unmatched_gt, unmatched_pred = match_boxes(gt_boxes, pred_boxes, iou_thresh=0.50)
    assert len(matches) == 1
    assert matches[0][0] == 0  # gt index 0
    assert matches[0][1] == 0  # pred index 0
    assert len(unmatched_gt) == 1  # gt index 1 was missed (FN)
    assert unmatched_gt[0] == 1
    assert len(unmatched_pred) == 1  # pred index 1 was false positive (FP)
    assert unmatched_pred[0] == 1


def test_coordinate_scaling_bug_demonstration():
    """
    Demonstrates that the coordinate scaling bug caused IoU = 0.0 on DsPCBSD+,
    and proves that the fix restores the true match.
    """
    # Normalized label representing a defect on a 226x226 DsPCBSD+ image
    # Center = (113, 113), Width = 20, Height = 20
    # Normalized: xc = 113/226 = 0.5, yc = 0.5, w = 20/226 = 0.088495, h = 0.088495
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".txt") as f:
        f.write("3 0.500000 0.500000 0.088495 0.088495\n")
        lbl_path = Path(f.name)

    try:
        # Prediction from YOLO in native 226x226 space:
        pred_box = [{"class_id": 3, "confidence": 0.75, "bbox": [103.0, 103.0, 123.0, 123.0]}]

        # BUGGY behavior: scaling with 640x640
        buggy_gt = load_yolo_labels(lbl_path, 640, 640)
        # GT box becomes centered at (320, 320) with width 56.6px
        buggy_matches, buggy_unmatched_gt, buggy_unmatched_pred = match_boxes(buggy_gt, pred_box, 0.50)
        assert len(buggy_matches) == 0, "Buggy code should have produced 0 matches due to coordinate mismatch"
        assert len(buggy_unmatched_gt) == 1
        assert len(buggy_unmatched_pred) == 1

        # CORRECTED behavior: scaling with actual dimensions (226, 226)
        fixed_gt = load_yolo_labels(lbl_path, 226, 226)
        fixed_matches, fixed_unmatched_gt, fixed_unmatched_pred = match_boxes(fixed_gt, pred_box, 0.50)
        assert len(fixed_matches) == 1, "Corrected code must produce a valid true positive match"
        assert len(fixed_unmatched_gt) == 0
        assert len(fixed_unmatched_pred) == 0
    finally:
        if lbl_path.exists():
            lbl_path.unlink()
