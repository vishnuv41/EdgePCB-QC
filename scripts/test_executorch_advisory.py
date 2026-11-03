"""
Test Real On-Device ExecuTorch Llama 3.2 Advisory Inference
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import sys
import os
import json
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from advisory.executorch_runtime import ExecuTorchLlamaRunner

def run_inference():
    print("=" * 65)
    print("PHASE 5 — REAL LOCAL INFERENCE VIA EXECUTORCH RUNTIME")
    print("=" * 65)

    runner = ExecuTorchLlamaRunner()
    print(f"[*] ExecuTorch Model Loaded: {runner.is_loaded}")
    print(f"[*] PTE Path:                {runner.pte_path}")

    # Competition test payload
    sample_payload = {
        "board_id": "PCB-001",
        "decision": "REJECT",
        "detected_defects": [
            {
                "class": "open_circuit",
                "severity": "CRITICAL",
                "confidence": 0.92
            }
        ]
    }

    print("\n[*] Input Inspection Payload:")
    print(json.dumps(sample_payload, indent=2))

    print("\n[*] Generating advisory explanation...")
    t0 = time.perf_counter()
    result = runner.generate_advisory(sample_payload)
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 65)
    print("ADVISORY EXPLANATION OUTPUT (SCHEMA VALIDATED)")
    print("=" * 65)
    print(json.dumps(result, indent=2))

    # Verify schema constraints
    adv = result["advisory"]
    assert "what_detected" in adv, "Missing what_detected"
    assert "why_it_matters" in adv, "Missing why_it_matters"
    assert "operator_action" in adv, "Missing operator_action"
    assert "confidence_caveat" in adv, "Missing confidence_caveat"
    assert adv["electrical_confirmation_needed"] is True, "Must require electrical confirmation"

    print("\n[+] Verification PASSED: Grounded advisory explanation successfully generated.")
    print(f"[+] Total execution time: {elapsed:.2f}s")
    print(f"[+] Backend verified:     {result['metrics']['backend']}")
    print(f"[+] TTFT:                 {result['metrics']['ttft_ms']} ms")
    print(f"[+] Peak RAM:             {result['metrics']['peak_ram_mb']} MB")

if __name__ == "__main__":
    run_inference()
