"""
Safety and Regression Tests for PCB-5655 / PCB-7331 Zero-Detection PASS Prevention,
Frame-Edge Clipping Gates, and Optical AOI Electrical Verification Constraints.
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import pytest
import numpy as np
from pathlib import Path
from decision_engine.decision import DecisionEngine
from decision_engine.rules import evaluate_rules
from database.models import DatabaseManager, InspectionRecord


@pytest.fixture
def engine():
    return DecisionEngine()


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_safety_inspections.db"
    db_manager = DatabaseManager(db_path=db_file)
    yield db_manager
    db_manager.close()


class TestPCB5655SafetyRegression:
    """
    Verifies that zero detections cannot trigger an unvalidated PASS when
    frame-edge clipping, degraded quality, or domain unfamiliarity are present.
    """

    def test_clean_board_valid_domain_passes(self, engine):
        """Zero detections + full coverage + acceptable quality -> qualified PASS."""
        payload = {
            "board_id": "PCB-CLEAN-001",
            "defects": [],
            "coverage": 1.0,
            "domain_status": "within_domain",
            "image_quality_status": "acceptable",
            "clipping_warnings": 0,
            "quality_assessment": {
                "domain_status": "within_domain",
                "image_quality_status": "acceptable",
                "is_quality_acceptable": True,
                "is_domain_validated": True,
                "clipping_warnings": 0,
                "has_valid_reference": True,
                "reference_alignment_score": 0.95
            }
        }
        record = engine.evaluate(payload, preset="consumer")
        assert record["decision"] == "PASS"
        assert record["rule_id"] == "RULE_008_ZERO_CRITICAL_DEFECTS_PASS"
        assert "visual clearance" in record["rule_description"].lower()
        # Optical AOI never verifies electrical functionality
        assert record["electrical_functionality_verified"] is False
        assert record["electrical_test_required_for_final_signoff"] is True

    def test_pcb_5655_clipping_gate_prevents_false_pass(self, engine):
        """
        Exact PCB-5655 / PCB-7331 scenario:
        Zero detections + 93% coverage + 2 clipping warnings -> strictly INSPECT.
        Never PASS.
        """
        payload = {
            "board_id": "PCB-5655",
            "defects": [],
            "coverage": 0.93,
            "domain_status": "within_domain",
            "image_quality_status": "acceptable",
            "clipping_warnings": 2,
            "quality_assessment": {
                "domain_status": "within_domain",
                "image_quality_status": "acceptable",
                "is_quality_acceptable": True,
                "is_domain_validated": True,
                "clipping_warnings": 2,
                "clipped_edges": 2,
                "reason_for_inspect": "Frame-edge clipping detected: 2 edge(s) truncated"
            }
        }
        record = engine.evaluate(payload, preset="consumer")
        assert record["decision"] == "INSPECT", f"PCB-5655 must yield INSPECT, got {record['decision']}"
        assert record["rule_id"] == "RULE_005D_FRAME_EDGE_CLIPPING"
        assert "clipping warning detected" in record["rule_description"]
        assert record["review_required"] is True
        assert record["clipping_warnings"] == 2

    def test_degraded_image_quality_yields_inspect(self, engine):
        """Zero detections + degraded image quality -> strictly INSPECT."""
        payload = {
            "board_id": "PCB-BLURRY-002",
            "defects": [],
            "coverage": 1.0,
            "domain_status": "within_domain",
            "image_quality_status": "severely_blurred",
            "clipping_warnings": 0,
            "quality_assessment": {
                "domain_status": "within_domain",
                "image_quality_status": "severely_blurred",
                "is_quality_acceptable": False,
                "is_domain_validated": True,
                "reason_for_inspect": "Laplacian sharpness variance 18.2 below threshold 100.0"
            }
        }
        record = engine.evaluate(payload, preset="consumer")
        assert record["decision"] == "INSPECT"
        assert record["rule_id"] == "RULE_005A_DEGRADED_IMAGE_QUALITY"
        assert "degraded" in record["rule_description"].lower()

    def test_unfamiliar_domain_yields_inspect(self, engine):
        """Zero detections + unfamiliar domain -> strictly INSPECT."""
        payload = {
            "board_id": "PCB-UNFAMILIAR-003",
            "defects": [],
            "coverage": 1.0,
            "domain_status": "unfamiliar_domain",
            "image_quality_status": "acceptable",
            "clipping_warnings": 0,
            "quality_assessment": {
                "domain_status": "unfamiliar_domain",
                "image_quality_status": "acceptable",
                "is_quality_acceptable": True,
                "is_domain_validated": False,
                "reason_for_inspect": "Colour distribution shift from training baseline"
            }
        }
        record = engine.evaluate(payload, preset="consumer")
        assert record["decision"] == "INSPECT"
        assert record["rule_id"] == "RULE_005B_UNFAMILIAR_IMAGE_DOMAIN"

    def test_missing_or_poor_reference_alignment_yields_inspect(self, engine):
        """Zero detections + reference alignment uncertain -> strictly INSPECT."""
        payload = {
            "board_id": "PCB-MISALIGNED-004",
            "defects": [],
            "coverage": 1.0,
            "domain_status": "within_domain",
            "image_quality_status": "acceptable",
            "clipping_warnings": 0,
            "quality_assessment": {
                "domain_status": "within_domain",
                "image_quality_status": "acceptable",
                "is_quality_acceptable": True,
                "is_domain_validated": True,
                "has_valid_reference": False,
                "reference_alignment_score": 0.42,
                "reason_for_inspect": "Reference template missing or alignment uncertain"
            }
        }
        record = engine.evaluate(payload, preset="consumer")
        assert record["decision"] == "INSPECT"
        assert record["rule_id"] == "RULE_005C_ALIGNMENT_UNCERTAIN"


class TestElectricalVerificationConstraints:
    """
    Verifies that optical AOI never asserts electrical functionality,
    and flags suspected opens and shorts for physical contact testing.
    """

    def test_suspected_short_circuit_flags_electrical_testing(self, engine):
        payload = {
            "board_id": "PCB-SHORT-005",
            "defects": [
                {
                    "class": "short_circuit",
                    "confidence": 0.88,
                    "bbox": [100, 100, 50, 20]
                }
            ],
            "coverage": 1.0,
            "domain_status": "within_domain",
            "image_quality_status": "acceptable"
        }
        record = engine.evaluate(payload, preset="consumer")
        assert record["decision"] == "REJECT"
        assert record["requires_electrical_test"] is True
        assert record["requires_electrical_test_for_suspected_defect"] is True
        assert record["electrical_functionality_verified"] is False
        assert record["electrical_test_required_for_final_signoff"] is True

    def test_suspected_open_circuit_flags_electrical_testing(self, engine):
        payload = {
            "board_id": "PCB-OPEN-006",
            "defects": [
                {
                    "class": "open_circuit",
                    "confidence": 0.65,
                    "bbox": [200, 200, 30, 10]
                }
            ],
            "coverage": 1.0,
            "domain_status": "within_domain",
            "image_quality_status": "acceptable"
        }
        record = engine.evaluate(payload, preset="consumer")
        assert record["decision"] == "INSPECT"
        assert record["requires_electrical_test"] is True
        assert record["requires_electrical_test_for_suspected_defect"] is True
        assert record["electrical_functionality_verified"] is False
        assert record["electrical_test_required_for_final_signoff"] is True


class TestAuditAndDatabasePersistence:
    """Verifies that clipping warnings, electrical flags, and operator overrides persist accurately."""

    def test_clipping_and_electrical_persistence_roundtrip(self, test_db, engine):
        payload = {
            "board_id": "PCB-7331",
            "defects": [],
            "coverage": 0.93,
            "domain_status": "within_domain",
            "image_quality_status": "acceptable",
            "clipping_warnings": 2,
            "quality_assessment": {
                "clipping_warnings": 2,
                "domain_status": "within_domain",
                "image_quality_status": "acceptable"
            }
        }
        record = engine.evaluate(payload, preset="consumer")
        insp_id = test_db.log_inspection(
            board_id=record["board_id"],
            image_path="test_images/PCB-7331.jpg",
            decision_record=record,
            timings={"vision_ms": 12.5, "advisory_ms": 0.2, "total_ms": 12.7}
        )

        retrieved = test_db.get_inspection(insp_id)
        assert retrieved is not None
        assert retrieved["board_id"] == "PCB-7331"
        assert retrieved["decision"] == "INSPECT"
        assert retrieved["clipping_warnings"] == 2
        assert retrieved["electrical_functionality_verified"] is False
        assert retrieved["electrical_test_required_for_final_signoff"] is True

    def test_operator_override_audit_trail(self, test_db, engine):
        payload = {
            "board_id": "PCB-OVERRIDE-001",
            "defects": [],
            "coverage": 0.93,
            "clipping_warnings": 1,
            "quality_assessment": {"clipping_warnings": 1}
        }
        record = engine.evaluate(payload, preset="consumer")
        insp_id = test_db.log_inspection(
            board_id=record["board_id"],
            image_path="test_images/PCB-OVERRIDE-001.jpg",
            decision_record=record
        )

        # Operator inspects physically and confirms edge is clean tooling margin
        success = test_db.record_override(
            inspection_id=insp_id,
            operator_id="QA_TECH_402",
            new_decision="PASS",
            reason="Tooling strip edge checked under 40x optical microscope. No active copper in clipped boundary."
        )
        assert success is True

        retrieved = test_db.get_inspection(insp_id)
        assert retrieved["decision"] == "INSPECT"  # original machine decision preserved
        assert retrieved["final_decision"] == "PASS"  # updated final decision
        assert retrieved["operator_override"] is not None
        assert retrieved["operator_override"]["operator_id"] == "QA_TECH_402"
        assert retrieved["operator_override"]["override_decision"] == "PASS"
        assert "40x optical microscope" in retrieved["operator_override"]["reason"]
