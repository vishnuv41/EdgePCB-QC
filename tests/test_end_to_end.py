"""
End-to-End System Integration Tests for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Tests complete multi-stage pipeline:
Vision Inspection -> Decision Engine -> Llama Advisory -> SQLite Audit Log -> Operator Override
"""

import pytest
from pathlib import Path
from inspection.cpu_inference import CPUInspector
from decision_engine.decision import DecisionEngine
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from database.models import DatabaseManager

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMG = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
SAMPLE_ANNO = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041_not" / "00041000.txt"


def test_full_offline_pipeline_flow(tmp_path):
    db_path = tmp_path / "integration_test.db"
    db_mgr = DatabaseManager(db_path)
    cpu_inspector = CPUInspector()
    decision_engine = DecisionEngine()
    advisory_runner = ExecuTorchLlamaRunner()

    assert SAMPLE_IMG.exists(), f"Sample image missing at {SAMPLE_IMG}"
    assert SAMPLE_ANNO.exists(), f"Sample annotation missing at {SAMPLE_ANNO}"

    # Stage 1: Vision Inference
    defects = cpu_inspector.inspect_with_annotations(SAMPLE_IMG, SAMPLE_ANNO)
    assert len(defects) > 0, "Vision stage should detect defect instances from sample"
    vision_ms = cpu_inspector.get_last_timing()

    # Stage 2: Deterministic Decision Engine
    board_id = "PCB-INTEGRATION-001"
    payload = {
        "board_id": board_id,
        "defects": defects,
        "coverage": 0.97
    }
    decision_res = decision_engine.evaluate(payload, preset="consumer")
    assert decision_res["decision"] in ["PASS", "INSPECT", "REJECT"]
    assert decision_res["rule_id"].startswith("RULE_")

    # Stage 3: LLM Advisory via ExecuTorch
    advisory_res = advisory_runner.generate_advisory(decision_res)
    assert "advisory" in advisory_res
    adv = advisory_res["advisory"]
    assert len(adv["what_detected"]) > 0
    assert len(adv["operator_action"]) > 0
    advisory_ms = advisory_res["metrics"]["latency_ms"]

    # Stage 4: Persistent SQLite Audit Logging
    insp_id = db_mgr.log_inspection(
        board_id=board_id,
        image_path=str(SAMPLE_IMG),
        decision_record=decision_res,
        advisory_record=adv,
        timings={
            "vision_ms": vision_ms,
            "advisory_ms": advisory_ms,
            "total_ms": vision_ms + advisory_ms
        }
    )
    assert insp_id is not None

    # Stage 5: Retrieve and Validate Record
    saved_rec = db_mgr.get_inspection(insp_id)
    assert saved_rec["board_id"] == board_id
    assert saved_rec["decision"] == decision_res["decision"]
    assert len(saved_rec["defects"]) == len(defects)
    assert saved_rec["llm_explanation"] is not None

    # Stage 6: Operator Override
    override_ok = db_mgr.record_override(
        inspection_id=insp_id,
        operator_id="LEAD-QC-99",
        new_decision="INSPECT",
        reason="Manual bench inspection requested to verify trace clearance."
    )
    assert override_ok is True

    updated_rec = db_mgr.get_inspection(insp_id)
    assert updated_rec["final_decision"] == "INSPECT"
    assert updated_rec["operator_override"]["operator_id"] == "LEAD-QC-99"

    # Stage 7: Analytics Validation
    analytics = db_mgr.get_defect_statistics()
    assert analytics["total_inspections"] == 1
    assert analytics["decisions"]["INSPECT"] == 1

    db_mgr.close()
