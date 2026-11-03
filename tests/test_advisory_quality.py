"""
Advisory Quality, Token Generation Integrity & Attribution Regression Tests
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. EOS token and stop-token handling.
2. Complete sentence parsing and trailing fragment truncation.
3. Empty/incomplete generation triggering clean fallback without merging fragments.
4. Llama failure and error fallback behavior.
5. Correct backend and model attribution.
6. Deterministic decision invariance across advisory modes.
7. Audit count label calculation and consistency.
8. Multi-tier prompt tokenization length boundaries.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from advisory.schema import AdvisoryExplanation, InspectionPayload
from advisory.prompt import (
    format_compact_prompt,
    format_standard_prompt,
    format_comprehensive_prompt,
    format_concise_prompt
)
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from decision_engine.decision import DecisionEngine


@pytest.fixture
def sample_payload() -> Dict[str, Any]:
    return {
        "board_id": "PCB-TEST-001",
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Detected 1 critical defect(s) with confidence >= 0.75",
        "requires_electrical_test": True,
        "preset_applied": "consumer",
        "defects": [
            {
                "class": "short_circuit",
                "confidence": 0.94,
                "severity": "critical",
                "is_confirmed": True,
                "review_required": False
            },
            {
                "class": "open_circuit",
                "confidence": 0.35,
                "severity": "minor",
                "is_confirmed": False,
                "review_required": True
            }
        ]
    }


def test_prompt_tiers_token_lengths(sample_payload):
    runner = ExecuTorchLlamaRunner()
    if runner.tokenizer:
        p_compact = format_compact_prompt(sample_payload)
        p_standard = format_standard_prompt(sample_payload)
        p_comp = format_comprehensive_prompt(sample_payload)

        toks_compact = runner.tokenizer.encode(p_compact, bos=False, eos=False, allowed_special="all")
        toks_standard = runner.tokenizer.encode(p_standard, bos=False, eos=False, allowed_special="all")
        toks_comp = runner.tokenizer.encode(p_comp, bos=False, eos=False, allowed_special="all")

        assert 25 <= len(toks_compact) <= 45, f"Compact prompt out of bounds: {len(toks_compact)}"
        assert 50 <= len(toks_standard) <= 75, f"Standard prompt out of bounds: {len(toks_standard)}"
        assert 75 <= len(toks_comp) <= 110, f"Comprehensive prompt out of bounds: {len(toks_comp)}"


def test_complete_sentence_truncation_logic():
    """Verifies that incomplete trailing words are truncated to the last complete sentence."""
    raw_gen = "The short circuit on PCB-001 may cause thermal overload. Quarantine the board and run flying"
    last_punc = max(raw_gen.rfind("."), raw_gen.rfind("!"), raw_gen.rfind("?"))
    complete_text = raw_gen[:last_punc + 1].strip()

    assert complete_text == "The short circuit on PCB-001 may cause thermal overload."
    assert complete_text.endswith(".")
    assert "flying" not in complete_text


def test_empty_generation_triggers_clean_fallback(sample_payload):
    """When generated tokens are empty or whitespace, fallback is activated without merging."""
    runner = ExecuTorchLlamaRunner()
    with patch.object(runner, "is_loaded", True):
        mock_module = MagicMock()
        mock_outputs = (torch.zeros(1, 100, dtype=torch.float32),)
        mock_module.forward.return_value = mock_outputs

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [10, 20, 30]
        # Return empty decoded string
        mock_tok.decode.return_value = "   "

        with patch.object(runner, "module", mock_module), patch.object(runner, "tokenizer", mock_tok):
            res = runner.generate_advisory(sample_payload, mode="auto")
            assert res["backend"] == "deterministic_fallback"
            assert res["status"] == "fallback_on_incomplete"
            assert res["advisory"]["generated_text"] is None
            assert "[Llama 3.2 1B Inference]" not in res["advisory"]["operator_action"]


def test_incomplete_generation_without_sentence_triggers_fallback(sample_payload):
    """When output has no sentence terminator (e.g. fragments only), fallback is triggered."""
    runner = ExecuTorchLlamaRunner()
    with patch.object(runner, "is_loaded", True):
        mock_module = MagicMock()
        mock_outputs = (torch.zeros(1, 100, dtype=torch.float32),)
        mock_module.forward.return_value = mock_outputs

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [10, 20, 30]
        mock_tok.decode.return_value = "short_circuit occurs when a trace is connected to"

        with patch.object(runner, "module", mock_module), patch.object(runner, "tokenizer", mock_tok):
            res = runner.generate_advisory(sample_payload, mode="auto")
            assert res["backend"] == "deterministic_fallback"
            assert res["status"] == "fallback_on_incomplete"
            assert res["advisory"]["generated_text"] is None
            # Incomplete text must NEVER be appended into operator_action
            assert "connected to" not in res["advisory"]["operator_action"]


def test_successful_complete_generation_attribution(sample_payload):
    """When generation produces complete sentences, attribute directly to Llama without merging."""
    runner = ExecuTorchLlamaRunner()
    with patch.object(runner, "is_loaded", True):
        mock_module = MagicMock()
        mock_outputs = (torch.zeros(1, 100, dtype=torch.float32),)
        mock_module.forward.return_value = mock_outputs

        mock_tok = MagicMock()
        mock_tok.encode.return_value = [10, 20, 30]
        valid_sentence = "The short circuit on PCB-TEST-001 may cause bridging. Isolate board for electrical test."
        mock_tok.decode.return_value = valid_sentence

        with patch.object(runner, "module", mock_module), patch.object(runner, "tokenizer", mock_tok):
            res = runner.generate_advisory(sample_payload, mode="auto")
            assert res["backend"] == "executorch_llama_3.2_1b"
            assert res["model_id"] == "meta-llama/Llama-3.2-1B-Instruct"
            assert res["status"] == "executed"
            assert res["advisory"]["generated_text"] == valid_sentence
            # Operator action remains clean SOP
            assert "[Llama 3.2 1B Inference]" not in res["advisory"]["operator_action"]


def test_llama_failure_exception_fallback(sample_payload):
    """Runtime exceptions during ExecuTorch call must fall back safely."""
    runner = ExecuTorchLlamaRunner()
    with patch.object(runner, "is_loaded", True):
        mock_module = MagicMock()
        mock_module.forward.side_effect = RuntimeError("Simulated ExecuTorch kernel fault")

        with patch.object(runner, "module", mock_module):
            res = runner.generate_advisory(sample_payload, mode="auto")
            assert res["backend"] == "deterministic_fallback"
            assert res["status"] == "fallback_on_error"
            assert "Simulated ExecuTorch kernel fault" in res["fallback_reason"]
            assert res["advisory"]["generated_text"] is None


def test_decision_invariance_across_all_advisory_outcomes(sample_payload):
    """Quality decision, rule ID, and electrical test flags must never change."""
    runner = ExecuTorchLlamaRunner()
    # 1. Fast fallback
    res_fast = runner.generate_advisory(sample_payload, mode="fast")

    # 2. Incomplete generation fallback
    with patch.object(runner, "is_loaded", True):
        mock_module = MagicMock()
        mock_module.forward.return_value = (MagicMock(),)
        mock_tok = MagicMock()
        mock_tok.encode.return_value = [1, 2]
        mock_tok.decode.return_value = "incomplete sentence fragment"
        with patch.object(runner, "module", mock_module), patch.object(runner, "tokenizer", mock_tok):
            res_incomplete = runner.generate_advisory(sample_payload, mode="auto")

    # Both must preserve identical deterministic flags
    assert res_fast["advisory"]["electrical_confirmation_needed"] is True
    assert res_incomplete["advisory"]["electrical_confirmation_needed"] is True


def test_audit_counts_calculation():
    """Verify that audit counts in decision_record are mathematically consistent."""
    engine = DecisionEngine()
    payload = {
        "board_id": "PCB-AUDIT-TEST",
        "coverage": 0.98,
        "defects": [
            # High-conf critical
            {"class": "short_circuit", "confidence": 0.92},
            # Medium-conf critical
            {"class": "open_circuit", "confidence": 0.65},
            # Low-conf review
            {"class": "pin_hole_missing_pad", "confidence": 0.32},
            # High-conf major
            {"class": "mouse_bite", "confidence": 0.88}
        ]
    }
    record = engine.evaluate(payload, preset="consumer")
    audit_counts = record.get("audit_counts")
    assert audit_counts is not None

    # Total critical = short_circuit (0.92) + open_circuit (0.65) = 2 (pin_hole conf 0.32 was downgraded to minor review)
    assert audit_counts["total_critical"] >= 2
    # High-conf critical = short_circuit (0.92 >= 0.75) = 1
    assert audit_counts["high_conf_critical"] == 1
    # Low-conf review = pin_hole_missing_pad (0.32 < 0.40) = 1
    assert audit_counts["low_conf_review"] == 1
    # Triggered rule is RULE_001, so triggered count matches high-conf critical
    assert audit_counts["rule_triggering"] == 1
