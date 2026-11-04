"""
Unit & Integration Tests for Conservative Inspection Policy
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Acceptable quality within-domain board with zero defects -> PASS
2. Unfamiliar color domain board with zero detections -> INSPECT (NOT PASS, NOT REJECT)
3. Defocused / blurred board with zero detections -> INSPECT (NOT PASS)
4. Underexposed / dark board with zero detections -> INSPECT (NOT PASS)
5. Overexposed / glare board with zero detections -> INSPECT (NOT PASS)
6. High-confidence critical defect on color board -> REJECT (preserves defect rejection)
7. Electrical testing requirement preserved on open/short
8. Presence and correctness of audit fields (domain_status, image_quality_status, review_required, reason_for_inspect)
"""

import pytest
import numpy as np
from decision_engine.decision import DecisionEngine
from inspection.quality_assessment import assess_inspection_image


@pytest.fixture
def decision_engine():
    return DecisionEngine()


def test_clean_within_domain_pass(decision_engine):
    payload = {
        "board_id": "BOARD_CLEAN_01",
        "defects": [],
        "coverage": 1.0,
        "domain_status": "within_domain",
        "image_quality_status": "acceptable"
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "PASS"
    assert rec["rule_id"] == "RULE_008_ZERO_CRITICAL_DEFECTS_PASS"
    assert rec["domain_status"] == "within_domain"
    assert rec["image_quality_status"] == "acceptable"
    assert rec["review_required"] is False
    assert rec["reason_for_inspect"] is None


def test_unfamiliar_color_domain_routes_to_inspect(decision_engine):
    # Simulated color PCB where model detected 0 defects due to domain shift
    payload = {
        "board_id": "BOARD_COLOR_UNKNOWN",
        "defects": [],
        "coverage": 1.0,
        "domain_status": "unfamiliar_domain",
        "image_quality_status": "acceptable",
        "reason_for_inspect": "Unfamiliar image domain (color_pcb): Model only validated on monochrome"
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "INSPECT"
    assert rec["rule_id"] == "RULE_005B_UNFAMILIAR_IMAGE_DOMAIN"
    assert rec["review_required"] is True
    assert "unfamiliar" in rec["reason_for_inspect"].lower()
    # Ensure color PCB is NOT rejected simply because it is color
    assert rec["decision"] != "REJECT"


def test_blurred_image_routes_to_inspect(decision_engine):
    payload = {
        "board_id": "BOARD_BLURRED",
        "defects": [],
        "coverage": 1.0,
        "domain_status": "within_domain",
        "image_quality_status": "blurred",
        "reason_for_inspect": "Optical blur detected (Laplacian variance 12.5 < 80.0)"
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "INSPECT"
    assert rec["rule_id"] == "RULE_005A_DEGRADED_IMAGE_QUALITY"
    assert rec["review_required"] is True
    assert rec["image_quality_status"] == "blurred"


def test_underexposed_image_routes_to_inspect(decision_engine):
    payload = {
        "board_id": "BOARD_DARK",
        "defects": [],
        "coverage": 1.0,
        "domain_status": "within_domain",
        "image_quality_status": "underexposed",
        "reason_for_inspect": "Severe underexposure (Mean brightness 18.2 < 30.0)"
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "INSPECT"
    assert rec["rule_id"] == "RULE_005A_DEGRADED_IMAGE_QUALITY"
    assert rec["review_required"] is True


def test_color_pcb_with_critical_defect_rejects(decision_engine):
    # Defect on color PCB must still be rejected if critical defect is detected
    payload = {
        "board_id": "BOARD_COLOR_DEFECTIVE",
        "defects": [
            {"class": "short_circuit", "confidence": 0.88, "severity": "critical"}
        ],
        "coverage": 1.0,
        "domain_status": "unfamiliar_domain",
        "image_quality_status": "acceptable"
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "REJECT"
    assert rec["rule_id"] == "RULE_001_HIGH_CONF_CRITICAL_DEFECT"
    assert rec["requires_electrical_test"] is True


def test_quality_assessment_synthetic_blur():
    # Create sharp synthetic image with edges
    sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
    sharp_img[50:150, 50:150] = 255
    res_sharp = assess_inspection_image(sharp_img)
    assert res_sharp["is_quality_acceptable"] is True
    assert res_sharp["image_quality_status"] == "acceptable"

    # Heavily blur the image
    import cv2
    blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), 15.0)
    res_blur = assess_inspection_image(blurred_img)
    assert res_blur["is_quality_acceptable"] is False
    assert res_blur["image_quality_status"] == "blurred"
    assert res_blur["review_required"] is True


def test_quality_assessment_color_detection():
    # Monochrome image (R=G=B)
    mono_img = np.full((100, 100, 3), 128, dtype=np.uint8)
    res_mono = assess_inspection_image(mono_img, supported_domains=["monochrome_high_contrast"])
    assert res_mono["domain_status"] == "within_domain"

    # Color image (Green solder mask: high G, low R, B)
    color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    color_img[:, :, 1] = 180  # Green
    color_img[:, :, 0] = 20   # Blue
    color_img[:, :, 2] = 20   # Red
    res_color = assess_inspection_image(color_img, supported_domains=["monochrome_high_contrast"])
    assert res_color["domain_status"] == "unfamiliar_domain"
    assert res_color["review_required"] is True


def test_missing_reference_routes_to_inspect(decision_engine):
    """Scenario 10 regression test: Board with no defects and missing reference must route to INSPECT."""
    qa = {
        "image_quality_status": "acceptable",
        "domain_status": "within_domain",
        "has_valid_reference": False,
        "reference_alignment_score": None,
        "reason_for_inspect": "Missing or unverified golden reference template for inspection verification."
    }
    payload = {
        "board_id": "BOARD_NO_REF",
        "defects": [],
        "coverage": 1.0,
        "quality_assessment": qa
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "INSPECT"
    assert rec["rule_id"] == "RULE_005C_ALIGNMENT_UNCERTAIN"
    assert rec["review_required"] is True
    assert rec["decision"] != "PASS"
    assert rec["decision"] != "REJECT"


def test_misaligned_reference_routes_to_inspect(decision_engine):
    """Scenario 10 regression test: Board with poor alignment score (< 0.60) must route to INSPECT."""
    qa = {
        "image_quality_status": "acceptable",
        "domain_status": "within_domain",
        "has_valid_reference": True,
        "reference_alignment_score": 0.42,
        "reason_for_inspect": "Uncertain reference alignment score (0.42 < 0.60)."
    }
    payload = {
        "board_id": "BOARD_POOR_ALIGN",
        "defects": [],
        "coverage": 1.0,
        "quality_assessment": qa
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "INSPECT"
    assert rec["rule_id"] == "RULE_005C_ALIGNMENT_UNCERTAIN"
    assert rec["review_required"] is True
    assert rec["decision"] != "PASS"


def test_missing_reference_with_critical_defect_rejects(decision_engine):
    """When a high-confidence critical defect is found, REJECT still takes precedence even if reference is missing."""
    qa = {
        "image_quality_status": "acceptable",
        "domain_status": "within_domain",
        "has_valid_reference": False,
        "reference_alignment_score": 0.42
    }
    payload = {
        "board_id": "BOARD_DEFECTIVE_NO_REF",
        "defects": [
            {"class": "short_circuit", "confidence": 0.88, "severity": "critical"}
        ],
        "coverage": 1.0,
        "quality_assessment": qa
    }
    rec = decision_engine.evaluate(payload)
    assert rec["decision"] == "REJECT"
    assert rec["rule_id"] == "RULE_001_HIGH_CONF_CRITICAL_DEFECT"
    assert rec["requires_electrical_test"] is True

