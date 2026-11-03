"""
Unit Tests for Deterministic Decision Engine
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
- Clean board PASS evaluation
- High-confidence critical defect REJECT
- Uncertain/medium-confidence critical defect INSPECT
- Low-coverage image flag INSPECT
- Automotive preset strictness vs Consumer
- Operator audit trail and override
"""

import pytest
from decision_engine.decision import DecisionEngine


@pytest.fixture
def engine():
    return DecisionEngine()


def test_clean_board_pass(engine):
    payload = {
        "board_id": "PCB-CLEAN-001",
        "defects": [],
        "coverage": 1.0
    }
    res = engine.evaluate(payload, preset="consumer")
    assert res["decision"] == "PASS"
    assert res["rule_id"] == "RULE_008_ZERO_CRITICAL_DEFECTS_PASS"
    assert res["requires_electrical_test"] is False


def test_high_conf_critical_reject(engine):
    payload = {
        "board_id": "PCB-BAD-001",
        "defects": [
            {"class": "open_circuit", "confidence": 0.88, "area_fraction": 0.01}
        ],
        "coverage": 0.98
    }
    res = engine.evaluate(payload, preset="consumer")
    assert res["decision"] == "REJECT"
    assert res["rule_id"] == "RULE_001_HIGH_CONF_CRITICAL_DEFECT"
    assert res["requires_electrical_test"] is True
    assert res["summary_counts"]["critical"] == 1


def test_suspected_critical_inspect(engine):
    payload = {
        "board_id": "PCB-SUSPECT-001",
        "defects": [
            {"class": "short_circuit", "confidence": 0.55, "area_fraction": 0.01}
        ],
        "coverage": 0.95
    }
    res = engine.evaluate(payload, preset="consumer")
    # For consumer, reject_conf is 0.75, so 0.55 triggers INSPECT
    assert res["decision"] == "INSPECT"
    assert res["rule_id"] == "RULE_003_SUSPECTED_CRITICAL_DEFECT"
    assert res["requires_electrical_test"] is True


def test_insufficient_coverage_inspect(engine):
    payload = {
        "board_id": "PCB-PARTIAL-001",
        "defects": [],
        "coverage": 0.70  # Below 90% required
    }
    res = engine.evaluate(payload, preset="consumer")
    assert res["decision"] == "INSPECT"
    assert res["rule_id"] == "RULE_005_INSUFFICIENT_COVERAGE"


def test_preset_strictness_automotive_vs_consumer(engine):
    payload = {
        "board_id": "PCB-EDGE-001",
        "defects": [
            {"class": "spur", "confidence": 0.65, "area_fraction": 0.02}
        ],
        "coverage": 0.98
    }
    # For consumer (reject_conf 0.75), 0.65 spur triggers INSPECT
    consumer_res = engine.evaluate(payload, preset="consumer")
    assert consumer_res["decision"] == "INSPECT"

    # For automotive (reject_conf 0.50, max_major 0), 0.65 spur triggers REJECT
    auto_res = engine.evaluate(payload, preset="automotive")
    assert auto_res["decision"] == "REJECT"


def test_operator_override_audit(engine):
    payload = {
        "board_id": "PCB-OVERRIDE-001",
        "defects": [
            {"class": "open_circuit", "confidence": 0.85, "area_fraction": 0.01}
        ],
        "coverage": 1.0
    }
    initial_res = engine.evaluate(payload, preset="consumer")
    assert initial_res["decision"] == "REJECT"

    override_res = engine.apply_override(
        decision_record=initial_res,
        operator_id="TECH-42",
        new_decision="INSPECT",
        reason="Visual artifact from shadow under microscope lens."
    )

    assert override_res["decision"] == "INSPECT"
    assert override_res["operator_override"] is not None
    assert override_res["operator_override"]["operator_id"] == "TECH-42"
    assert override_res["operator_override"]["original_decision"] == "REJECT"
    assert override_res["operator_override"]["override_decision"] == "INSPECT"
