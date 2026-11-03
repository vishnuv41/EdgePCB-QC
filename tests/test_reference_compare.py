"""
Unit Tests for Golden Reference Comparison Module
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
- Loading template and test image pair
- Alignment via feature matching & homography
- Difference map computation
- Discrepancy bounding box extraction
- Integration with Decision Engine (producing INSPECT verdict for anomalies)
"""

import pytest
from pathlib import Path
from inspection.reference_compare import (
    load_reference,
    load_test_image,
    align_images,
    compare_images,
    highlight_differences
)
from decision_engine.decision import DecisionEngine

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_TEMP = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_temp.jpg"
SAMPLE_TEST = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"


@pytest.mark.skipif(not SAMPLE_TEMP.exists(), reason="DeepPCB sample images not found")
def test_reference_comparison_flow():
    ref_img = load_reference(SAMPLE_TEMP)
    test_img = load_test_image(SAMPLE_TEST)

    assert ref_img is not None
    assert test_img is not None
    assert ref_img.shape == (640, 640, 3)
    assert test_img.shape == (640, 640, 3)

    aligned_test, H = align_images(test_img, ref_img)
    assert aligned_test.shape == ref_img.shape
    assert H.shape == (3, 3)

    diff, mask = compare_images(aligned_test, ref_img)
    assert diff.shape == (640, 640)
    assert mask.shape == (640, 640)

    annotated, anomalies = highlight_differences(aligned_test, mask)
    assert annotated.shape == (640, 640, 3)
    # The defective test image has known artificially injected defects
    assert len(anomalies) > 0, "Expected discrepancy regions detected between test and template"

    # Feed detected anomalies into Decision Engine
    engine = DecisionEngine()
    payload = {
        "board_id": "DEEPPCB-00041000",
        "defects": anomalies,
        "coverage": 1.0
    }
    decision = engine.evaluate(payload, preset="consumer")
    # Discrepancies without high-confidence electrical confirmation trigger INSPECT or REJECT
    assert decision["decision"] in ["INSPECT", "REJECT"]
