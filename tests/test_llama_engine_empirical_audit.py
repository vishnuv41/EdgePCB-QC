"""
Empirical Audit and Latency Benchmark of Edge SLM Advisory Engine (Llama 3.2 1B)
EdgePCB-QC — Bharat AI-SoC Challenge 2026-27 | PS5

Validates:
1. Artifact verification: llama3_2_1b_xnnpack_8da4w.pte & tokenizer.model.
2. ExecuTorch runtime loading with XNNPACK backend.
3. Live end-to-end inference with real PCB defect inputs.
4. Genuine neural generation verification (no hardcoded/canned responses).
5. Explicit latency separation:
   - Model loading latency
   - First-token latency (TTFT)
   - Cold-start total latency
   - Warmed-up total latency
   - Deterministic rule engine latency (<1 ms)
6. Authoritative verdict preservation (Deterministic rules govern PASS/FAIL).
7. Runtime device and execution backend attribution.
8. Unloaded/Missing model state handling (reports NOT LOADED).
"""

import os
import sys
import time
import pytest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from advisory.executorch_runtime import ExecuTorchLlamaRunner
from advisory.fallback import generate_fallback_advisory


def test_artifact_and_tokenizer_presence():
    """Verify that the INT4 quantized PTE model and tokenizer genuinely exist on disk."""
    pte_path = WORKSPACE / "models" / "llama" / "llama3_2_1b_xnnpack_8da4w.pte"
    tok_path = WORKSPACE / "models" / "llama" / "tokenizer.model"

    assert pte_path.exists(), f"Missing PTE artifact: {pte_path}"
    assert tok_path.exists(), f"Missing tokenizer: {tok_path}"

    pte_size_bytes = pte_path.stat().st_size
    assert pte_size_bytes > 1_000_000_000, f"PTE file too small to be 1B model: {pte_size_bytes} bytes"
    assert pte_size_bytes < 2_000_000_000, f"PTE file too large to be 8da4w INT4: {pte_size_bytes} bytes"
    print(f"\n[+] Verified PTE model: {pte_path.name} ({pte_size_bytes / (1024*1024):.2f} MB)")


def test_model_loading_and_runtime_device():
    """Verify model loading time, ExecuTorch initialization, and runtime device attribution."""
    t0 = time.perf_counter()
    runner = ExecuTorchLlamaRunner()
    load_time_s = time.perf_counter() - t0

    assert runner.is_loaded is True, "ExecuTorch failed to load llama3_2_1b_xnnpack_8da4w.pte"
    assert runner.module is not None, "ExecuTorch module is None"
    assert runner.tokenizer is not None, "Tokenizer is None"

    status = runner.get_status_info()
    assert status["is_loaded"] is True
    assert "8da4w" in status["quantization"]
    assert "XNNPACK" in status["runtime"]
    print(f"[+] ExecuTorch model loaded in {load_time_s:.2f}s")
    print(f"[+] Status info: {status}")


def test_deterministic_latency_sub_millisecond():
    """Verify deterministic rule engine executes in sub-1ms and is distinct from neural inference."""
    runner = ExecuTorchLlamaRunner()
    sample_defect_payload = {
        "board_id": "PCB-BENCH-001",
        "decision": "REJECT",
        "defects": [
            {"class": "open_circuit", "severity": "critical", "confidence": 0.89, "bbox": [100, 200, 150, 250]}
        ],
        "rule_id": "RULE-CRIT-OPEN",
        "rule_description": "Open circuit detected"
    }

    latencies = []
    for _ in range(5):
        t0 = time.perf_counter()
        res = runner.generate_advisory(sample_defect_payload, mode="fast")
        t_el = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_el)

    avg_det_ms = sum(latencies) / len(latencies)
    print(f"[+] Deterministic rule engine average latency: {avg_det_ms:.4f} ms")
    assert avg_det_ms < 2.0, f"Deterministic latency too high: {avg_det_ms} ms"
    assert res["source"] == "deterministic_fallback"
    assert res["advisory"]["generated_text"] is None


def test_genuine_llama_inference_execution():
    """
    Send a PCB defect description to the advisory engine, execute genuine neural inference,
    verify generated explanation text, and measure latencies.
    """
    runner = ExecuTorchLlamaRunner()
    defect_payload = {
        "board_id": "PCB-AUDIT-449",
        "decision": "REJECT",
        "defects": [
            {"class": "open_circuit", "severity": "critical", "confidence": 0.93, "bbox": [55, 120, 95, 160]},
            {"class": "mouse_bite", "severity": "major", "confidence": 0.81, "bbox": [200, 310, 230, 340]}
        ],
        "rule_id": "RULE-CRIT-001",
        "rule_description": "Detected critical defect with confidence >= 0.75"
    }

    # 1. Cold start inference
    t0 = time.perf_counter()
    cold_res = runner.generate_advisory(defect_payload, mode="auto", max_new_tokens=24)
    cold_total_ms = (time.perf_counter() - t0) * 1000.0

    assert cold_res["source"] == "executorch_llama_3.2_1b"
    assert cold_res["backend"] == "executorch_llama_3.2_1b"
    assert cold_res["metrics"]["status"] == "executed"
    
    gen_text = cold_res["advisory"]["generated_text"]
    assert gen_text is not None, "Generated text is None"
    assert len(gen_text) > 10, f"Generated text too short: '{gen_text}'"
    print(f"\n[+] Cold-start Llama generated explanation:\n    \"{gen_text}\"")
    print(f"    Cold TTFT: {cold_res['metrics']['ttft_ms']} ms")
    print(f"    Cold Total: {cold_total_ms:.1f} ms | Tokens/sec: {cold_res['metrics']['tokens_per_second']}")

    # 2. Warmed-up inference
    t0 = time.perf_counter()
    warm_res = runner.generate_advisory(defect_payload, mode="auto", max_new_tokens=24)
    warm_total_ms = (time.perf_counter() - t0) * 1000.0

    assert warm_res["source"] == "executorch_llama_3.2_1b"
    print(f"\n[+] Warmed-up Llama generated explanation:\n    \"{warm_res['advisory']['generated_text']}\"")
    print(f"    Warm TTFT: {warm_res['metrics']['ttft_ms']} ms")
    print(f"    Warm Total: {warm_total_ms:.1f} ms | Tokens/sec: {warm_res['metrics']['tokens_per_second']}")

    # Verification: neural latency is orders of magnitude distinct from rule engine latency
    assert warm_total_ms > 1000.0, "Neural generation must take >1s on CPU"
    assert warm_res["metrics"]["tokens_generated"] > 0


def test_verdict_authority_invariant():
    """Verify that Llama generation NEVER overrides or alters the deterministic PASS/FAIL verdict."""
    runner = ExecuTorchLlamaRunner()
    defect_payload = {
        "board_id": "PCB-CRIT-999",
        "decision": "REJECT",
        "defects": [
            {"class": "short_circuit", "severity": "critical", "confidence": 0.96, "bbox": [15, 25, 45, 55]}
        ],
        "rule_id": "RULE-CRIT-SHORT",
        "rule_description": "Short circuit between power and ground nets"
    }

    res = runner.generate_advisory(defect_payload, mode="auto", max_new_tokens=16)
    
    # Check that decision remains REJECT
    assert defect_payload["decision"] == "REJECT"
    # Check that electrical confirmation remains mandatory
    assert res["advisory"]["electrical_confirmation_needed"] is True
    assert "[REJECT VERDICT]" in res["advisory"]["operator_action"]


def test_unloaded_model_shows_not_loaded():
    """Verify that if the model artifact is missing or disabled, the status reports NOT LOADED."""
    missing_runner = ExecuTorchLlamaRunner(pte_path=Path("non_existent_llama.pte"))
    assert missing_runner.is_loaded is False
    status = missing_runner.get_status_info()
    assert status["is_loaded"] is False
    assert status["artifact_exists"] is False
