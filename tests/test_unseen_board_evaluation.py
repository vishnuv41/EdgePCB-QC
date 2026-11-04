"""
Automated Regression Tests for Unseen Bare PCB Evaluation Pipeline
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Ingestion and evaluation of unseen bare PCB image files.
2. Verified zero-detection on clean template correctly produces zero false defect boxes.
3. Zero-detection without aligned reference or with low coverage routes to INSPECT, never false PASS.
4. Defective board triggers appropriate critical defect rule (REJECT or INSPECT).
5. Whole-board multi-megapixel image executes tiled inference without coordinate distortion.
6. Electrical verification status is strictly unverified by optical AOI.
"""

from pathlib import Path
import json
import pytest
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
from inspection.evaluate_unseen_board import inspect_single_board
from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner
from decision_engine.decision import DecisionEngine

MODEL_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
CLEAN_TEMPLATE = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_temp.jpg"
DEFECTIVE_BOARD = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
WHOLE_BOARD = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train" / "images" / "68-2-1.jpg"


@pytest.fixture(scope="module")
def setup_subsystems():
    inspector = CPUInspector(model_path=MODEL_PATH, conf_threshold=0.25, device="cpu")
    tiled_runner = TiledInferenceRunner(
        inspector=inspector,
        tile_size=640,
        overlap=0.40,
        iou_threshold=0.30,
        edge_padding=True,
        class_agnostic_nms=False
    )
    decision_engine = DecisionEngine()
    return inspector, tiled_runner, decision_engine


def test_clean_template_zero_detections_safety(setup_subsystems):
    inspector, tiled_runner, decision_engine = setup_subsystems
    if not CLEAN_TEMPLATE.exists():
        pytest.skip("DeepPCB raw dataset not present")

    res = inspect_single_board(
        CLEAN_TEMPLATE, inspector, tiled_runner, decision_engine, preset="consumer"
    )
    # A clean PCB template should produce zero defect boxes
    assert res["defects_localized_count"] == 0, f"Expected 0 defects on clean template, got {res['defects_localized_count']}"
    # Without verified reference alignment or full coverage, it must NOT falsely PASS
    assert res["decision"] == "INSPECT", f"Zero detections without complete coverage must route to INSPECT, got {res['decision']}"
    assert res["electrical_functionality_verified"] is False
    assert res["electrical_test_required_for_final_signoff"] is True


def test_defective_board_localization_and_rejection(setup_subsystems):
    inspector, tiled_runner, decision_engine = setup_subsystems
    if not DEFECTIVE_BOARD.exists():
        pytest.skip("DeepPCB raw dataset not present")

    res = inspect_single_board(
        DEFECTIVE_BOARD, inspector, tiled_runner, decision_engine, preset="consumer"
    )
    assert res["defects_localized_count"] > 0, "Failed to localize defects on known defective board"
    assert res["decision"] in ["REJECT", "INSPECT"]
    assert res["requires_electrical_test_for_suspected_defect"] is True
    assert res["electrical_functionality_verified"] is False


def test_whole_board_tiled_evaluation(setup_subsystems):
    inspector, tiled_runner, decision_engine = setup_subsystems
    if not WHOLE_BOARD.exists():
        pytest.skip("PCB-Defect V2 raw dataset not present")

    res = inspect_single_board(
        WHOLE_BOARD, inspector, tiled_runner, decision_engine, preset="consumer"
    )
    assert res["dimensions"]["width"] == 2048
    assert res["dimensions"]["height"] == 1651
    assert res["defects_localized_count"] > 0
    # Boundary clamping invariant: all bounding boxes must reside inside [0, w] and [0, h]
    for d in res["defects_manifest"]:
        bbox = d["bbox"]
        assert 0 <= bbox[0] <= 2048
        assert 0 <= bbox[1] <= 1651
        assert 0 <= bbox[2] <= 2048
        assert 0 <= bbox[3] <= 1651
