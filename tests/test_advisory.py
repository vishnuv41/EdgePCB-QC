"""
Unit Tests for Advisory Pipeline & Schema Contracts
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
- Pydantic schema validation for inspection payload and advisory responses
- Llama 3 prompt construction and token formatting
- Deterministic fallback generation for PASS and REJECT boards
- ExecuTorch runtime runner integration and metric recording
"""

import pytest
from pathlib import Path
from advisory.schema import InspectionPayload, AdvisoryExplanation
from advisory.prompt import format_llama_prompt
from advisory.fallback import generate_fallback_advisory
from advisory.executorch_runtime import ExecuTorchLlamaRunner


def test_schema_valid_payload():
    raw_payload = {
        "board_id": "PCB-TEST-001",
        "timestamp": "2026-11-03T19:00:00Z",
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Detected critical defect",
        "requires_electrical_test": True,
        "coverage": 0.98,
        "preset_applied": "consumer",
        "defects": [
            {
                "class": "open_circuit",
                "confidence": 0.92,
                "severity": "critical",
                "bbox": [10.0, 20.0, 50.0, 60.0],
                "location": "upper-right",
                "area_fraction": 0.015
            }
        ],
        "summary_counts": {"critical": 1, "major": 0, "minor": 0}
    }
    payload = InspectionPayload(**raw_payload)
    assert payload.board_id == "PCB-TEST-001"
    assert payload.decision == "REJECT"
    assert len(payload.defects) == 1
    assert payload.defects[0].defect_class == "open_circuit"


def test_prompt_formatting():
    payload = {
        "board_id": "PCB-001",
        "decision": "INSPECT",
        "rule_id": "RULE_003_SUSPECTED_CRITICAL_DEFECT",
        "defects": [{"class": "short_circuit", "confidence": 0.58}]
    }
    prompt = format_llama_prompt(payload)
    assert "<|begin_of_text|>" in prompt
    assert "<|start_header_id|>system<|end_header_id|>" in prompt
    assert "<|start_header_id|>user<|end_header_id|>" in prompt
    assert "<|start_header_id|>assistant<|end_header_id|>" in prompt
    assert "PCB-001" in prompt


def test_fallback_clean_board_pass():
    payload = {
        "board_id": "PCB-CLEAN",
        "decision": "PASS",
        "defects": [],
        "requires_electrical_test": False
    }
    advisory = generate_fallback_advisory(payload)
    assert isinstance(advisory, AdvisoryExplanation)
    assert "Defect-free" in advisory.what_detected
    assert advisory.electrical_confirmation_needed is False


def test_fallback_critical_reject():
    payload = {
        "board_id": "PCB-CRIT",
        "decision": "REJECT",
        "defects": [
            {"class": "open_circuit", "confidence": 0.95, "severity": "critical"}
        ],
        "requires_electrical_test": True
    }
    advisory = generate_fallback_advisory(payload)
    assert isinstance(advisory, AdvisoryExplanation)
    assert "Open Circuit" in advisory.what_detected
    assert advisory.electrical_confirmation_needed is True
    assert "[REJECT VERDICT]" in advisory.operator_action


def test_executorch_runner_fallback_graceful():
    # Test runner with missing artifact triggers graceful fallback
    fallback_runner = ExecuTorchLlamaRunner(pte_path=Path("non_existent_model.pte"))
    payload = {
        "board_id": "PCB-RUNNER-01",
        "decision": "PASS",
        "defects": []
    }
    result = fallback_runner.generate_advisory(payload)
    assert "advisory" in result
    assert "metrics" in result
    assert result["metrics"]["backend"] == "deterministic_fallback"
    assert result["metrics"]["status"] == "active"
    assert result["metrics"]["latency_ms"] >= 0.0


def test_executorch_runner_fast_mode():
    runner = ExecuTorchLlamaRunner()
    payload = {
        "board_id": "PCB-RUNNER-02",
        "decision": "REJECT",
        "defects": [
            {"class": "short_circuit", "confidence": 0.91, "severity": "critical"}
        ]
    }
    result = runner.generate_advisory(payload, mode="fast")
    assert "advisory" in result
    assert result["metrics"]["backend"] == "deterministic_fallback"
    assert result["advisory"]["electrical_confirmation_needed"] is True
    assert result["metrics"]["latency_ms"] < 100.0
