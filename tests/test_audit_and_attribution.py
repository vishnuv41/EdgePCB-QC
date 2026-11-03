"""
Unit & Integration Tests for Inspection Accuracy, Advisory Attribution, and Audit Integrity
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Fallback attribution (never claims Llama on fallback output)
2. Actual ExecuTorch attribution (records genuine backend and model ID)
3. Open-circuit electrical verification requirement
4. Short-circuit electrical verification requirement
5. Low-confidence defect handling (flagged for review, not auto-major)
6. Operator override audit integrity (preserves original decision, tracks justification)
7. Historical model-version preservation and legacy reconciliation
"""

import pytest
import json
import sqlite3
from pathlib import Path
from decision_engine.decision import DecisionEngine
from decision_engine.rules import evaluate_rules
from advisory.fallback import generate_fallback_advisory
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from database.models import DatabaseManager, InspectionRecord


def test_fallback_attribution(tmp_path):
    """Test 1: Fallback advisory is explicitly attributed to deterministic fallback, never Llama."""
    db_path = tmp_path / "test_fallback_attr.db"
    db_mgr = DatabaseManager(db_path)
    runner = ExecuTorchLlamaRunner()

    payload = {
        "board_id": "PCB-TEST-ATTR-01",
        "decision": "INSPECT",
        "rule_id": "RULE_004_MAJOR_DEFECT_FLAGGED",
        "rule_description": "Detected major defect",
        "defects": [{"class": "spur", "confidence": 0.60, "severity": "major"}]
    }

    # Generate in fast mode (fallback)
    advisory_res = runner.generate_advisory(payload, mode="fast")
    assert advisory_res["backend"] == "deterministic_fallback"
    assert advisory_res["model_version"] == "None (Deterministic Expert Fallback)"

    insp_id = db_mgr.log_inspection(
        board_id="PCB-TEST-ATTR-01",
        image_path="test_img.jpg",
        decision_record=payload,
        advisory_record=advisory_res["advisory"],
        advisory_metrics=advisory_res["metrics"],
        timings={"vision_ms": 30.0, "advisory_ms": 0.15, "total_ms": 30.15}
    )

    rec = db_mgr.get_inspection(insp_id)
    assert rec["advisory_backend"] == "deterministic_fallback"
    assert rec["llm_version"] == "None (Deterministic Expert Fallback)"
    assert rec["advisory_model_id"] == "ipc_expert_rules_engine"
    assert rec["advisory_status"] == "active"
    assert rec["advisory_latency_ms"] == 0.15
    assert "Llama" not in rec["llm_version"]

    db_mgr.close()


def test_actual_executorch_attribution(tmp_path):
    """Test 2: Genuine ExecuTorch execution is accurately recorded with Llama model version."""
    db_path = tmp_path / "test_executorch_attr.db"
    db_mgr = DatabaseManager(db_path)

    payload = {
        "board_id": "PCB-TEST-EXEC-02",
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Detected critical defect",
        "defects": [{"class": "open_circuit", "confidence": 0.95, "severity": "critical"}]
    }

    # Simulated metrics from genuine ExecuTorch execution
    exec_metrics = {
        "backend": "executorch_llama_3.2_1b",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "model_version": "Llama-3.2-1B-Instruct",
        "status": "executed",
        "model_artifact": "llama3_2_1b_fp16.pte",
        "tokens_evaluated": 45,
        "latency_ms": 35420.0,
        "ttft_ms": 35400.0,
        "tokens_per_second": 0.82,
        "peak_ram_mb": 3021.0
    }

    insp_id = db_mgr.log_inspection(
        board_id="PCB-TEST-EXEC-02",
        image_path="test_img_exec.jpg",
        decision_record=payload,
        advisory_record={"what_detected": "Open Circuit", "operator_action": "Quarantine"},
        advisory_metrics=exec_metrics,
        timings={"vision_ms": 35.0, "advisory_ms": 35420.0, "total_ms": 35455.0}
    )

    rec = db_mgr.get_inspection(insp_id)
    assert rec["advisory_backend"] == "executorch_llama_3.2_1b"
    assert rec["llm_version"] == "Llama-3.2-1B-Instruct"
    assert rec["advisory_model_id"] == "meta-llama/Llama-3.2-1B-Instruct"
    assert rec["advisory_status"] == "executed"
    assert rec["advisory_latency_ms"] == 35420.0

    db_mgr.close()


def test_open_circuit_electrical_verification_requirement():
    """Test 3: Visually suspected open circuit mandates electrical verification across rules and advisory."""
    engine = DecisionEngine()
    payload = {
        "board_id": "PCB-OPEN-TEST",
        "coverage": 0.98,
        "defects": [
            {
                "class": "open_circuit",
                "confidence": 0.88,
                "bbox": [100.0, 100.0, 150.0, 120.0],
                "area_fraction": 0.002
            }
        ]
    }

    decision_res = engine.evaluate(payload, preset="consumer")
    assert decision_res["decision"] == "REJECT"
    assert decision_res["requires_electrical_test"] is True

    adv = generate_fallback_advisory(decision_res)
    assert adv.electrical_confirmation_needed is True
    assert "continuity" in adv.operator_action.lower()
    assert "[Engineering Guidance" in adv.operator_action


def test_short_circuit_electrical_verification_requirement():
    """Test 4: Visually suspected short circuit mandates electrical isolation verification without scraping."""
    engine = DecisionEngine()
    payload = {
        "board_id": "PCB-SHORT-TEST",
        "coverage": 0.98,
        "defects": [
            {
                "class": "short_circuit",
                "confidence": 0.85,
                "bbox": [200.0, 200.0, 230.0, 220.0],
                "area_fraction": 0.001
            }
        ]
    }

    decision_res = engine.evaluate(payload, preset="consumer")
    assert decision_res["decision"] == "REJECT"
    assert decision_res["requires_electrical_test"] is True

    adv = generate_fallback_advisory(decision_res)
    assert adv.electrical_confirmation_needed is True
    assert "isolation" in adv.operator_action.lower()
    # Verify no unguided scraping advice
    assert "manual scraping" not in adv.operator_action.lower() or "do not attempt unguided manual scraping" in adv.operator_action.lower()


def test_low_confidence_defect_handling():
    """Test 5: Low-confidence detections are flagged for review rather than auto-rejected as major."""
    engine = DecisionEngine()
    payload = {
        "board_id": "PCB-LOW-CONF",
        "coverage": 0.98,
        "defects": [
            {
                "class": "spur",
                "confidence": 0.35,  # below consumer inspect_conf 0.40
                "bbox": [50.0, 50.0, 70.0, 70.0],
                "area_fraction": 0.0004
            }
        ]
    }

    decision_res = engine.evaluate(payload, preset="consumer")
    # Low confidence anomaly must trigger INSPECT review, NOT REJECT
    assert decision_res["decision"] == "INSPECT"
    assert decision_res["rule_id"] == "RULE_007_UNCERTAIN_SURFACE_ANOMALIES"
    defect = decision_res["defects"][0]
    assert defect["review_required"] is True
    assert defect["is_confirmed"] is False
    assert defect["severity"] == "minor"


def test_operator_override_audit_integrity(tmp_path):
    """Test 6: Operator override preserves original decision, requires technical justification, and is auditable."""
    db_path = tmp_path / "test_override_audit.db"
    db_mgr = DatabaseManager(db_path)

    payload = {
        "board_id": "PCB-OVERRIDE-01",
        "decision": "REJECT",
        "rule_id": "RULE_002_EXCESS_MAJOR_DEFECTS",
        "rule_description": "Detected excess major defects",
        "defects": [{"class": "mouse_bite", "confidence": 0.82, "severity": "major"}]
    }

    insp_id = db_mgr.log_inspection(
        board_id="PCB-OVERRIDE-01",
        image_path="test_img.jpg",
        decision_record=payload
    )

    # Apply valid override
    justification = "Optical reticle confirms remaining trace width exceeds 85% design allowance; safe for production."
    ok = db_mgr.record_override(
        inspection_id=insp_id,
        operator_id="QC-ENGINEER-42",
        new_decision="PASS",
        reason=justification
    )
    assert ok is True

    rec = db_mgr.get_inspection(insp_id)
    assert rec["decision"] == "REJECT"  # Original automated decision preserved
    assert rec["final_decision"] == "PASS"  # Overridden final decision
    override_audit = rec["operator_override"]
    assert override_audit["operator_id"] == "QC-ENGINEER-42"
    assert override_audit["original_decision"] == "REJECT"
    assert override_audit["override_decision"] == "PASS"
    assert override_audit["reason"] == justification
    assert "timestamp" in override_audit

    db_mgr.close()


def test_historical_model_version_preservation(tmp_path):
    """Test 7: Historical records with fast fallback are reconciled and never attribute fallback to Llama."""
    db_path = tmp_path / "test_legacy_reconcile.db"

    # Create raw legacy SQLite database mimicking pre-fix PCB-7199
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE inspection_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            inspection_id VARCHAR(64) UNIQUE,
            board_id VARCHAR(64),
            timestamp DATETIME,
            image_path VARCHAR(512),
            decision VARCHAR(16),
            final_decision VARCHAR(16),
            rule_id VARCHAR(64),
            rule_description TEXT,
            preset_applied VARCHAR(32),
            requires_electrical_test BOOLEAN,
            coverage FLOAT,
            defects_json TEXT,
            defect_count INTEGER,
            summary_counts_json TEXT,
            llm_explanation_json TEXT,
            operator_override_json TEXT,
            vision_model_version VARCHAR(32),
            llm_version VARCHAR(32),
            config_version VARCHAR(32),
            vision_latency_ms FLOAT,
            advisory_latency_ms FLOAT,
            total_latency_ms FLOAT
        )
    """)
    conn.execute("""
        INSERT INTO inspection_records (
            inspection_id, board_id, timestamp, image_path, decision, final_decision,
            rule_id, rule_description, preset_applied, requires_electrical_test, coverage,
            defects_json, defect_count, summary_counts_json, llm_explanation_json,
            vision_model_version, llm_version, config_version, vision_latency_ms,
            advisory_latency_ms, total_latency_ms
        ) VALUES (
            'legacy-uuid-7199', 'PCB-7199', '2026-11-03T17:33:19', 'img.jpg',
            'REJECT', 'REJECT', 'RULE_002', 'desc', 'consumer', 0, 0.98,
            '[]', 4, '{}', '{}', 'v1.0', 'Llama-3.2-1B-Instruct', '1.0', 92.31, 0.17, 92.48
        )
    """)
    conn.commit()
    conn.close()

    # Open with DatabaseManager which triggers schema migration & historical reconciliation
    db_mgr = DatabaseManager(db_path)
    rec = db_mgr.get_inspection("legacy-uuid-7199")

    # Verify reconciliation
    assert rec["board_id"] == "PCB-7199"
    assert rec["advisory_backend"] == "deterministic_fallback"
    assert rec["llm_version"] == "None (Deterministic Expert Fallback)"
    assert rec["advisory_model_id"] == "ipc_expert_rules_engine"
    assert rec["advisory_status"] == "active"
    assert rec["advisory_latency_ms"] == 0.17

    db_mgr.close()
