"""
Tests for Unified Inference Manager & Hardware Backends
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import pytest
import numpy as np
from pathlib import Path
from hardware.inference.inference_manager import InferenceManager
from hardware.inference.yolo_postprocess import letterbox_image, unletterbox_boxes, nms_numpy


def test_yolo_postprocess_letterbox():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, scale, (pad_l, pad_t) = letterbox_image(img, target_shape=(640, 640))
    assert padded.shape == (640, 640, 3)
    assert scale == 1.0
    assert pad_l == 0
    assert pad_t == 80  # (640 - 480) // 2


def test_yolo_postprocess_nms():
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],  # High overlap with box 0
        [100, 100, 150, 150]  # Separate box
    ], dtype=np.float32)
    scores = np.array([0.90, 0.85, 0.95], dtype=np.float32)

    keep = nms_numpy(boxes, scores, iou_threshold=0.45)
    assert len(keep) == 2
    assert 2 in keep  # highest score
    assert 0 in keep  # kept over box 1


def test_inference_manager_auto_resolution():
    im = InferenceManager(backend_mode="auto")
    status = im.get_status_info()

    assert status["configured_mode"] == "auto"
    # When Hailo hardware is absent, auto cleanly resolves to cpu
    assert status["active_backend"] == "cpu"
    assert "CPU" in status["active_backend_label"]
    assert status["hardware_accelerated"] is False


def test_inference_manager_cpu_inspection():
    im = InferenceManager(backend_mode="cpu")
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)

    result = im.inspect(dummy_img)
    assert "backend" in result
    assert "CPU" in result["backend"]
    assert "detections" in result
    assert isinstance(result["detections"], list)
    assert "latency_ms" in result
    assert result["latency_ms"] > 0
    assert result["hardware_accelerated"] is False


def test_inference_manager_hailo_handling_without_hardware():
    im = InferenceManager(backend_mode="hailo")
    status = im.get_status_info()
    assert status["hailo_backend_status"]["hardware_detected"] is False

    # Inspecting should trigger automatic fallback to CPU baseline with transparent note
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    result = im.inspect(dummy_img)
    assert "CPU" in result["backend"]
    assert result.get("fallback_triggered") is True
