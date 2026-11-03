"""
Unit Tests for Inspection Reliability & Fail-Safe Invariants
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import cv2
import numpy as np
import pytest
from pathlib import Path
from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner
from decision_engine.decision import DecisionEngine

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMG = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"


def test_optical_robustness_failsafe():
    inspector = CPUInspector()
    engine = DecisionEngine()
    assert SAMPLE_IMG.exists()
    img = cv2.imread(str(SAMPLE_IMG))

    # Test overexposure (+30) and dimming (-30)
    overexposed = cv2.convertScaleAbs(img, alpha=1.15, beta=30)
    dimmed = cv2.convertScaleAbs(img, alpha=0.85, beta=-30)

    dets_bright = inspector.inspect(overexposed)
    dets_dim = inspector.inspect(dimmed)

    assert len(dets_bright) > 0
    assert len(dets_dim) > 0

    dec_bright = engine.evaluate({"board_id": "TEST-BRIGHT", "defects": dets_bright, "coverage": 0.98})
    dec_dim = engine.evaluate({"board_id": "TEST-DIM", "defects": dets_dim, "coverage": 0.98})

    # Under no circumstances should a board with defects pass silently
    assert dec_bright["decision"] in ["INSPECT", "REJECT"]
    assert dec_dim["decision"] in ["INSPECT", "REJECT"]


def test_tiled_multi_panel_fusion():
    inspector = CPUInspector()
    tiled_runner = TiledInferenceRunner(inspector, tile_size=640, overlap=0.20)
    assert SAMPLE_IMG.exists()
    img = cv2.imread(str(SAMPLE_IMG))
    h, w = img.shape[:2]

    # Create 2x2 panel (1280x1280)
    panel = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    panel[0:h, 0:w] = img
    panel[0:h, w:w*2] = img
    panel[h:h*2, 0:w] = img
    panel[h:h*2, w:w*2] = img

    fused_dets = tiled_runner.inspect_tiled(panel)
    assert len(fused_dets) > 0
    # Coordinates must map into the 1280x1280 canvas
    for d in fused_dets:
        box = d["bbox"]
        assert 0.0 <= box[0] <= 1280.0
        assert 0.0 <= box[1] <= 1280.0
        assert 0.0 <= box[2] <= 1280.0
        assert 0.0 <= box[3] <= 1280.0
