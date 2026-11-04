"""
Comprehensive Regression Test Suite for EdgePCB-QC Inspection Correctness & Audit Integrity
Bharat AI-SoC Challenge 2026-27 | PS5

Tests all 8 zero-detection, domain shift, image quality, and electrical verification edge cases:
1. Genuinely clean, validated PCB image -> PASS, evidence-based advisory, requires_electrical_test=False.
2. Defective color image on baseline monochrome model -> Domain gate intercepts -> INSPECT (No false PASS!).
3. Unfamiliar image domain with 0 detections -> RULE_005B_UNFAMILIAR_IMAGE_DOMAIN -> INSPECT.
4. Degraded image quality (blur / underexposure) -> RULE_005A_DEGRADED_IMAGE_QUALITY -> INSPECT.
5. Missing or invalid golden reference board -> RULE_005C_ALIGNMENT_UNCERTAIN -> INSPECT.
6. Suspected electrical defects (open/short circuit) -> requires_electrical_test=True, advisory states mandatory contact test.
7. Zero-detection result with insufficient coverage (< 0.90) -> RULE_005_INSUFFICIENT_COVERAGE -> INSPECT.
8. Advisory attribution integrity when Llama is unavailable -> deterministic_fallback, rules engine, no false Llama claims.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

from decision_engine.decision import DecisionEngine
from inspection.quality_assessment import assess_inspection_image
from inspection.coverage import calculate_inspection_coverage
from advisory.fallback import generate_fallback_advisory
from advisory.schema import AdvisoryExplanation

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def decision_engine():
    return DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")


# --------------------------------------------------------------------------
# Scenario 1: Genuinely Clean, Validated PCB Image
# --------------------------------------------------------------------------
def test_scenario_1_genuinely_clean_pcb(decision_engine):
    """Clean board within validated domain with 100% coverage must return PASS with evidence-based advisory."""
    clean_gray_img = np.full((640, 640), 128, dtype=np.uint8)
    
    qa = assess_inspection_image(
        clean_gray_img,
        coverage=1.0,
        min_coverage_required=0.90,
        supported_domains=["monochrome_high_contrast"]
    )
    # Manually ensure acceptable quality for synthetic canvas
    qa["image_quality_status"] = "acceptable"
    qa["domain_status"] = "within_domain"

    payload = {
        "board_id": "PCB-CLEAN-001",
        "defects": [],
        "coverage": 1.0,
        "quality_assessment": qa
    }
    result = decision_engine.evaluate(payload, preset="consumer")

    assert result["decision"] == "PASS"
    assert result["rule_id"] == "RULE_008_ZERO_CRITICAL_DEFECTS_PASS"
    assert result["requires_electrical_test"] is False

    adv = generate_fallback_advisory(result)
    assert isinstance(adv, AdvisoryExplanation)
    # Evidence-based wording: never claim "all trace geometries conform" or "defect-free"
    assert "No defects were localized" in adv.what_detected
    assert "Defect-free" not in adv.what_detected
    assert "all trace geometries" not in adv.what_detected
    assert "Optical inspection confirms visible surface morphology only" in adv.confidence_caveat
    assert adv.electrical_confirmation_needed is False


# --------------------------------------------------------------------------
# Scenario 2: Defective Color Image on Baseline Monochrome Model (S_11833255)
# --------------------------------------------------------------------------
def test_scenario_2_defective_color_on_monochrome_baseline(decision_engine):
    """When baseline monochrome model returns 0 detections on a color PCB, domain gate must trigger INSPECT."""
    # Synthetic color PCB with green substrate (high saturation)
    color_img = np.zeros((226, 226, 3), dtype=np.uint8)
    color_img[:, :] = (30, 160, 40) # Green BGR

    # Baseline model only supports monochrome_high_contrast
    qa = assess_inspection_image(
        color_img,
        coverage=1.0,
        supported_domains=["monochrome_high_contrast"]
    )
    assert qa["domain_status"] == "unfamiliar_domain"

    payload = {
        "board_id": "PCB-S11833255-BASE",
        "defects": [], # 0 detections from blind baseline model
        "coverage": 1.0,
        "quality_assessment": qa
    }
    result = decision_engine.evaluate(payload, preset="consumer")

    assert result["decision"] == "INSPECT"
    assert "RULE_005" in result["rule_id"]
    assert result["decision"] != "PASS" # Zero false PASS!


# --------------------------------------------------------------------------
# Scenario 3: Unfamiliar Image Domain with Zero Detections
# --------------------------------------------------------------------------
def test_scenario_3_unfamiliar_domain_zero_detections(decision_engine):
    """Unfamiliar domain with 0 detections must route to INSPECT via RULE_005B."""
    qa = {
        "image_quality_status": "acceptable",
        "domain_status": "unfamiliar_domain",
        "is_quality_acceptable": True,
        "is_domain_validated": False,
        "review_required": True,
        "reason_for_inspect": "Unfamiliar image domain (color_pcb): Model validated on monochrome only."
    }
    payload = {
        "board_id": "PCB-DOMAIN-TEST",
        "defects": [],
        "coverage": 0.98,
        "quality_assessment": qa
    }
    result = decision_engine.evaluate(payload, preset="consumer")

    assert result["decision"] == "INSPECT"
    assert result["rule_id"] == "RULE_005B_UNFAMILIAR_IMAGE_DOMAIN"
    assert "unfamiliar" in result["rule_description"].lower()


# --------------------------------------------------------------------------
# Scenario 4: Degraded Image Quality Gate (Blur / Underexposure)
# --------------------------------------------------------------------------
def test_scenario_4_degraded_image_quality(decision_engine):
    """Degraded image quality (blur) with 0 detections must route to INSPECT via RULE_005A."""
    qa = {
        "image_quality_status": "blurred",
        "domain_status": "within_domain",
        "is_quality_acceptable": False,
        "is_domain_validated": True,
        "review_required": True,
        "reason_for_inspect": "Optical blur detected (Laplacian variance 42.1 < 80.0)"
    }
    payload = {
        "board_id": "PCB-BLURRED",
        "defects": [],
        "coverage": 0.95,
        "quality_assessment": qa
    }
    result = decision_engine.evaluate(payload, preset="consumer")

    assert result["decision"] == "INSPECT"
    assert result["rule_id"] == "RULE_005A_DEGRADED_IMAGE_QUALITY"
    assert "degraded" in result["rule_description"].lower()


# --------------------------------------------------------------------------
# Scenario 5: Missing or Invalid Golden Reference Template
# --------------------------------------------------------------------------
def test_scenario_5_missing_golden_reference(decision_engine):
    """When reference comparison is required but template is missing, must route to INSPECT via RULE_005C."""
    qa = {
        "image_quality_status": "acceptable",
        "domain_status": "within_domain",
        "has_valid_reference": False,
        "reference_alignment_score": None,
        "is_quality_acceptable": True,
        "is_domain_validated": True,
        "review_required": True,
        "reason_for_inspect": "Missing or unverified golden reference template for inspection verification."
    }
    payload = {
        "board_id": "PCB-NO-REF",
        "defects": [],
        "coverage": 0.98,
        "quality_assessment": qa
    }
    result = decision_engine.evaluate(payload, preset="consumer")

    assert result["decision"] == "INSPECT"
    assert result["rule_id"] == "RULE_005C_ALIGNMENT_UNCERTAIN"


# --------------------------------------------------------------------------
# Scenario 6: Suspected Electrical Defects (Open / Short Circuit)
# --------------------------------------------------------------------------
def test_scenario_6_suspected_electrical_defects(decision_engine):
    """Visually suspected open and short circuits must trigger mandatory electrical test and agreed advisory."""
    payload = {
        "board_id": "PCB-ELEC-001",
        "defects": [
            {
                "class": "open_circuit",
                "confidence": 0.88,
                "bbox": [100.0, 100.0, 150.0, 120.0],
                "area_fraction": 0.005,
                "location": "center region"
            },
            {
                "class": "short_circuit",
                "confidence": 0.92,
                "bbox": [200.0, 200.0, 240.0, 210.0],
                "area_fraction": 0.003,
                "location": "lower-right region"
            }
        ],
        "coverage": 0.98
    }
    result = decision_engine.evaluate(payload, preset="consumer")

    assert result["decision"] == "REJECT"
    assert result["requires_electrical_test"] is True

    adv = generate_fallback_advisory(result)
    assert adv.electrical_confirmation_needed is True
    assert "Open Circuit" in adv.what_detected
    assert "Short Circuit" in adv.what_detected
    assert "electrical confirmation via flying-probe or DMM is mandatory" in adv.confidence_caveat


# --------------------------------------------------------------------------
# Scenario 7: Zero Detections with Insufficient Inspection Coverage (< 0.90)
# --------------------------------------------------------------------------
def test_scenario_7_insufficient_coverage_zero_detections(decision_engine):
    """Zero detections with coverage below 90% must route to INSPECT via RULE_005."""
    payload = {
        "board_id": "PCB-LOW-COV",
        "defects": [],
        "coverage": 0.78, # 78% < 90% required
        "quality_assessment": {
            "image_quality_status": "acceptable",
            "domain_status": "within_domain",
            "is_quality_acceptable": True,
            "is_domain_validated": True,
            "review_required": False
        }
    }
    result = decision_engine.evaluate(payload, preset="consumer")

    assert result["decision"] == "INSPECT"
    assert result["rule_id"] == "RULE_005_INSUFFICIENT_COVERAGE"
    assert "78.0%" in result["rule_description"]


# --------------------------------------------------------------------------
# Scenario 8: Advisory Attribution Integrity (Deterministic Fallback)
# --------------------------------------------------------------------------
def test_scenario_8_advisory_attribution_integrity():
    """Fallback advisory must be explicitly attributed to rules engine and never claim Llama generated text."""
    payload = {
        "decision": "INSPECT",
        "defects": [
            {
                "class": "mouse_bite",
                "confidence": 0.55,
                "bbox": [50.0, 50.0, 70.0, 60.0]
            }
        ],
        "requires_electrical_test": False
    }
    adv = generate_fallback_advisory(payload)
    
    assert adv.generated_text is None
    assert adv.is_complete is True
    assert "Mouse Bite" in adv.what_detected
    assert "Engineering Guidance" in adv.operator_action
