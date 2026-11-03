"""
Unit Tests for Tiled High-Resolution Inference Module
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
- Intersection over Union (IoU) calculation accuracy
- Global Non-Maximum Suppression (NMS) duplicate suppression
- Image slicing geometry and coordinate bounds
- TiledInferenceRunner coordinate translation to full board coordinates
"""

import pytest
import numpy as np
from inspection.tiling import compute_iou, global_nms, TiledInferenceRunner


def test_compute_iou():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [0.0, 0.0, 10.0, 10.0]
    assert compute_iou(box1, box2) == 1.0

    box3 = [20.0, 20.0, 30.0, 30.0]
    assert compute_iou(box1, box3) == 0.0

    box4 = [5.0, 0.0, 15.0, 10.0]
    iou = compute_iou(box1, box4)
    # Intersection = 5 * 10 = 50. Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    assert abs(iou - (1.0 / 3.0)) < 1e-4


def test_global_nms_suppression():
    # Two identical bboxes of the same class with different confidences
    dets = [
        {"class": "open_circuit", "confidence": 0.85, "bbox": [10.0, 10.0, 50.0, 50.0]},
        {"class": "open_circuit", "confidence": 0.92, "bbox": [11.0, 10.0, 51.0, 50.0]},
        # Different class at same location should NOT be suppressed
        {"class": "spur", "confidence": 0.70, "bbox": [10.0, 10.0, 50.0, 50.0]}
    ]
    fused = global_nms(dets, iou_threshold=0.5)
    assert len(fused) == 2  # Best open_circuit (0.92) kept, spur kept
    classes = [d["class"] for d in fused]
    assert "open_circuit" in classes
    assert "spur" in classes
    assert fused[0]["confidence"] == 0.92


def test_image_tiling_geometry():
    class DummyInspector:
        def inspect(self, img):
            return [{"class": "mouse_bite", "confidence": 0.90, "bbox": [10.0, 10.0, 30.0, 30.0]}]

    runner = TiledInferenceRunner(DummyInspector(), tile_size=640, overlap=0.20)
    # Create synthetic 1200x1200 image
    img = np.zeros((1200, 1200, 3), dtype=np.uint8)
    tiles = runner.slice_image(img)

    assert len(tiles) >= 4  # Should create at least a 2x2 grid
    for tile_img, ox, oy in tiles:
        assert tile_img.shape[0] == 640
        assert tile_img.shape[1] == 640
        assert ox >= 0 and oy >= 0
        assert ox + 640 <= 1200
        assert oy + 640 <= 1200

    # Run tiled inspection
    detections = runner.inspect_tiled(img)
    assert len(detections) > 0
    # Confirm bounding boxes are offset to the global coordinates
    for d in detections:
        bbox = d["bbox"]
        assert 0.0 <= bbox[0] < bbox[2] <= 1200.0
        assert 0.0 <= bbox[1] < bbox[3] <= 1200.0
