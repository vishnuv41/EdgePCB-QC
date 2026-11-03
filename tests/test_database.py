"""
Unit Tests for SQLite Persistence and Inspection Logging
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
- SQLite table creation and record insertion
- Retrieval of full inspection payloads
- Operator override audit trail logging
- Analytics and Pareto defect distribution calculation
"""

import pytest
from database.models import DatabaseManager


@pytest.fixture
def db_mgr(tmp_path):
    db_file = tmp_path / "test_inspections.db"
    mgr = DatabaseManager(db_file)
    yield mgr
    mgr.close()


def test_log_and_get_inspection(db_mgr):
    decision_rec = {
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Detected critical defect",
        "preset_applied": "consumer",
        "requires_electrical_test": True,
        "coverage": 0.99,
        "config_version": "1.0",
        "defects": [
            {"class": "open_circuit", "confidence": 0.91, "severity": "critical"}
        ],
        "summary_counts": {"critical": 1, "major": 0, "minor": 0}
    }

    advisory_rec = {
        "what_detected": "Open Circuit detected",
        "why_it_matters": "Broken trace",
        "operator_action": "Scrap board",
        "electrical_confirmation_needed": True,
        "confidence_caveat": "Visual only"
    }

    timings = {"vision_ms": 14.5, "advisory_ms": 32.1, "total_ms": 46.6}

    insp_id = db_mgr.log_inspection(
        board_id="PCB-TEST-001",
        image_path="/path/to/test.jpg",
        decision_record=decision_rec,
        advisory_record=advisory_rec,
        timings=timings
    )

    assert insp_id is not None

    record = db_mgr.get_inspection(insp_id)
    assert record is not None
    assert record["board_id"] == "PCB-TEST-001"
    assert record["decision"] == "REJECT"
    assert record["final_decision"] == "REJECT"
    assert record["requires_electrical_test"] is True
    assert len(record["defects"]) == 1
    assert record["vision_latency_ms"] == 14.5
    assert record["advisory_latency_ms"] == 32.1


def test_operator_override(db_mgr):
    decision_rec = {
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Critical defect",
        "defects": [{"class": "spur", "confidence": 0.88}],
        "summary_counts": {"critical": 0, "major": 1, "minor": 0}
    }

    insp_id = db_mgr.log_inspection("PCB-OVERRIDE", "/img.jpg", decision_rec)
    success = db_mgr.record_override(
        inspection_id=insp_id,
        operator_id="OP-007",
        new_decision="INSPECT",
        reason="Superficial hair on lens, not on copper."
    )
    assert success is True

    record = db_mgr.get_inspection(insp_id)
    assert record["decision"] == "REJECT"  # Original
    assert record["final_decision"] == "INSPECT"  # Overridden
    assert record["operator_override"] is not None
    assert record["operator_override"]["operator_id"] == "OP-007"
    assert record["operator_override"]["override_decision"] == "INSPECT"


def test_analytics_summary(db_mgr):
    # Log 1 Pass, 1 Reject
    pass_rec = {
        "decision": "PASS",
        "rule_id": "RULE_008_ZERO_CRITICAL_DEFECTS_PASS",
        "rule_description": "Clean",
        "defects": [],
        "summary_counts": {"critical": 0, "major": 0, "minor": 0}
    }
    reject_rec = {
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Reject",
        "defects": [
            {"class": "open_circuit", "confidence": 0.9},
            {"class": "spur", "confidence": 0.8}
        ],
        "summary_counts": {"critical": 1, "major": 1, "minor": 0}
    }

    db_mgr.log_inspection("PCB-1", "/p1.jpg", pass_rec)
    db_mgr.log_inspection("PCB-2", "/p2.jpg", reject_rec)

    stats = db_mgr.get_defect_statistics()
    assert stats["total_inspections"] == 2
    assert stats["decisions"]["PASS"] == 1
    assert stats["decisions"]["REJECT"] == 1
    assert stats["defect_frequencies"].get("open_circuit") == 1
    assert stats["defect_frequencies"].get("spur") == 1
