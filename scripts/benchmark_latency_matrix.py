"""
Llama 3.2 1B ExecuTorch Latency & Token Performance Matrix Benchmark
Bharat AI-SoC Challenge 2026-27 | PS5

Benchmarks the full parameter matrix specified in Requirement 4:
- Prompts: 32 tokens (compact), 64 tokens (standard), 96 tokens (comprehensive)
- Generations: 16 tokens, 24 tokens
- CPU Threads: 4 threads, 8 threads
Investigates:
- Prefill time (TTFT) scaling vs prompt length
- Decode time and KV-cache efficiency
- Memory footprint containment
- Complete sentence generation quality
"""

import sys
import os
import json
import time
from pathlib import Path
import psutil

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from advisory.executorch_runtime import ExecuTorchLlamaRunner
from advisory.schema import InspectionPayload


def run_latency_matrix():
    print("=" * 80)
    print("EXECUTORCH LLAMA 3.2 1B LATENCY & PERFORMANCE MATRIX BENCHMARK")
    print("=" * 80)

    # Initialize runner once (verifying single load in memory)
    t_load_start = time.perf_counter()
    runner = ExecuTorchLlamaRunner()
    load_time_ms = round((time.perf_counter() - t_load_start) * 1000, 2)
    assert runner.is_loaded, "ExecuTorch runner must be loaded"
    print(f"[+] Model resident in memory. Init load time: {load_time_ms} ms")

    test_payload = {
        "board_id": "PCB-PERF-001",
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Detected critical defect with confidence >= 0.75",
        "requires_electrical_test": True,
        "preset_applied": "consumer",
        "defects": [
            {
                "class": "short_circuit",
                "confidence": 0.94,
                "severity": "critical",
                "is_confirmed": True
            }
        ]
    }

    prompt_tiers = [
        ("compact", "32-token"),
        ("standard", "64-token"),
        ("comprehensive", "96-token")
    ]
    gen_lengths = [16, 24]
    thread_counts = [4, 8]

    results = []

    total_runs = len(prompt_tiers) * len(gen_lengths) * len(thread_counts)
    current_run = 0

    print(f"[*] Executing {total_runs} benchmark configurations...\n")

    for p_tier, p_label in prompt_tiers:
        for gen_len in gen_lengths:
            for threads in thread_counts:
                current_run += 1
                config_name = f"{p_label} prompt / {gen_len} gen / {threads} threads"
                print(f"[{current_run}/{total_runs}] Running: {config_name} ...", end=" ", flush=True)

                res = runner.generate_advisory(
                    test_payload,
                    mode="auto",
                    prompt_tier=p_tier,
                    max_new_tokens=gen_len,
                    num_threads=threads
                )

                metrics = res["metrics"]
                adv = res["advisory"]

                p_tokens = metrics.get("prompt_tokens", metrics.get("tokens_evaluated", 0))
                g_tokens = metrics.get("tokens_generated", 0)
                ttft_ms = metrics.get("ttft_ms", 0.0)
                tot_latency_ms = metrics.get("latency_ms", 0.0)
                decode_time_s = round((tot_latency_ms - ttft_ms) / 1000.0, 2)
                decode_tok_s = round(g_tokens / (decode_time_s or 1.0), 2)
                peak_ram = metrics.get("peak_ram_mb", 0.0)
                gen_text = adv.get("generated_text") or metrics.get("raw_incomplete_output", "")
                is_comp = metrics.get("is_complete", False)

                record = {
                    "config": config_name,
                    "prompt_tier": p_tier,
                    "prompt_tokens": p_tokens,
                    "target_gen_tokens": gen_len,
                    "actual_gen_tokens": g_tokens,
                    "threads": threads,
                    "ttft_ms": ttft_ms,
                    "decode_time_s": decode_time_s,
                    "total_time_s": round(tot_latency_ms / 1000.0, 2),
                    "decode_tok_per_sec": decode_tok_s,
                    "overall_tok_per_sec": metrics.get("tokens_per_second", 0.0),
                    "peak_ram_mb": peak_ram,
                    "is_complete": is_comp,
                    "status": res["status"],
                    "generated_text": gen_text
                }
                results.append(record)
                print(f"Done in {record['total_time_s']}s (TTFT: {ttft_ms:.1f}ms, Gen: {decode_tok_s} tok/s, Complete: {is_comp})")

    # Save to JSON
    output_dir = WORKSPACE_ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "advisory_latency_matrix.json"

    matrix_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "cpu": "AMD Ryzen 9 8940HX (16 cores, 32 threads)",
            "memory": "32 GB DDR5",
            "model": "meta-llama/Llama-3.2-1B-Instruct (FP16 .pte)",
            "runtime": "ExecuTorch 1.5.1 CPU"
        },
        "model_load_time_ms": load_time_ms,
        "configurations": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(matrix_report, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[+] Saved authoritative latency benchmark matrix to {output_file}")
    print("=" * 80)

    # Print markdown table
    print("\n| Configuration | Threads | Prompt Tokens | Gen Tokens | TTFT (ms) | Gen Time (s) | Gen (tok/s) | Peak RAM | Complete? |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['config']} | {r['threads']} | {r['prompt_tokens']} | {r['actual_gen_tokens']} | {r['ttft_ms']:.1f} | {r['decode_time_s']}s | {r['decode_tok_per_sec']} | {r['peak_ram_mb']:.1f} MB | {r['is_complete']} |")


if __name__ == "__main__":
    run_latency_matrix()
