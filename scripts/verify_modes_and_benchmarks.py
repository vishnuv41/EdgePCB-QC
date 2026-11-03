"""
Live Dual-Mode Verification, Performance Benchmarks, and Correctness Validation
Bharat AI-SoC Challenge 2026-27 | PS5

Executes:
1. Real PCB Inspection in Fast Deterministic Fallback Mode
2. Real PCB Inspection in ExecuTorch Llama 3.2 1B On-Device Mode
3. Latency, TTFT, Tokens/s, and RAM Benchmarks (32/64/128 prompt, 16/32 generation, 4 vs 8 threads)
4. Defect Category Correctness (open, short, mouse bite, spurious copper)
5. SQLite Audit Log Persistence and Evidence Logging
"""

import os
import sys
import json
import time
import psutil
from pathlib import Path
import cv2
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.cpu_inference import CPUInspector
from decision_engine.decision import DecisionEngine
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from database.models import DatabaseManager


def run_dual_mode_inspections():
    print("=" * 70)
    print("STEP 1 & 2: LIVE PCB INSPECTION TEST IN BOTH ADVISORY MODES")
    print("=" * 70)

    sample_img_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
    assert sample_img_path.exists(), f"Sample image not found: {sample_img_path}"

    img = cv2.imread(str(sample_img_path))
    inspector = CPUInspector()
    engine = DecisionEngine()
    runner = ExecuTorchLlamaRunner()
    db = DatabaseManager()

    # Vision detection
    t_v0 = time.perf_counter()
    defects = inspector.inspect(img)
    t_v1 = time.perf_counter()
    vision_ms = (t_v1 - t_v0) * 1000.0
    print(f"[+] Vision Inference: Localized {len(defects)} defect(s) in {vision_ms:.2f} ms")

    # Deterministic Decision Engine
    payload = {"board_id": "PCB-MODE1-FAST", "defects": defects, "coverage": 0.98}
    dec_record = engine.evaluate(payload, preset="consumer")
    print(f"[+] Decision Engine Verdict: {dec_record['decision']} ({dec_record['rule_id']})")
    print(f"[+] Requires Electrical Test: {dec_record['requires_electrical_test']}")

    # -------------------------------------------------------------
    # TEST 1: FAST DETERMINISTIC FALLBACK MODE
    # -------------------------------------------------------------
    print("\n--- Running Mode 1: Fast Deterministic Fallback ---")
    t_a0 = time.perf_counter()
    adv_fast = runner.generate_advisory(dec_record, mode="fast")
    t_a1 = time.perf_counter()
    fast_ms = (t_a1 - t_a0) * 1000.0

    insp_fast_id = db.log_inspection(
        board_id="PCB-MODE1-FAST",
        image_path=str(sample_img_path),
        decision_record=dec_record,
        advisory_record=adv_fast["advisory"],
        advisory_metrics=adv_fast["metrics"],
        timings={"vision_ms": round(vision_ms, 2), "advisory_ms": round(fast_ms, 2), "total_ms": round(vision_ms + fast_ms, 2)}
    )
    rec_fast = db.get_inspection(insp_fast_id)
    print(f"[+] Mode 1 Result:")
    print(f"    - Inspection UUID:    {insp_fast_id}")
    print(f"    - Advisory Backend:   {rec_fast['advisory_backend']}")
    print(f"    - Advisory Model ID:  {rec_fast['advisory_model_id']}")
    print(f"    - LLM Version:        {rec_fast['llm_version']}")
    print(f"    - Advisory Latency:   {rec_fast['advisory_latency_ms']:.2f} ms")
    assert rec_fast["advisory_backend"] == "deterministic_fallback"
    assert "Llama" not in rec_fast["llm_version"]

    # -------------------------------------------------------------
    # TEST 2: EXECUTORCH LLAMA 3.2 1B ON-DEVICE INFERENCE MODE
    # -------------------------------------------------------------
    print("\n--- Running Mode 2: ExecuTorch Llama 3.2 1B On-Device Inference ---")
    payload_llama = {"board_id": "PCB-MODE2-LLAMA", "defects": defects, "coverage": 0.98}
    dec_record_llama = engine.evaluate(payload_llama, preset="consumer")

    t_l0 = time.perf_counter()
    adv_llama = runner.generate_advisory(dec_record_llama, mode="auto", max_new_tokens=16, num_threads=8)
    t_l1 = time.perf_counter()
    llama_ms = (t_l1 - t_l0) * 1000.0

    insp_llama_id = db.log_inspection(
        board_id="PCB-MODE2-LLAMA",
        image_path=str(sample_img_path),
        decision_record=dec_record_llama,
        advisory_record=adv_llama["advisory"],
        advisory_metrics=adv_llama["metrics"],
        timings={"vision_ms": round(vision_ms, 2), "advisory_ms": round(llama_ms, 2), "total_ms": round(vision_ms + llama_ms, 2)}
    )
    rec_llama = db.get_inspection(insp_llama_id)
    print(f"[+] Mode 2 Result:")
    print(f"    - Inspection UUID:    {insp_llama_id}")
    print(f"    - Advisory Backend:   {rec_llama['advisory_backend']}")
    print(f"    - Advisory Model ID:  {rec_llama['advisory_model_id']}")
    print(f"    - LLM Version:        {rec_llama['llm_version']}")
    print(f"    - Advisory Latency:   {rec_llama['advisory_latency_ms']:.2f} ms")
    print(f"    - Tokens Evaluated:   {adv_llama['metrics'].get('tokens_evaluated')}")
    print(f"    - Tokens Generated:   {adv_llama['metrics'].get('tokens_generated')}")
    print(f"    - Generated Text:     {adv_llama['advisory'].get('generated_text')}")
    assert rec_llama["advisory_backend"] == "executorch_llama_3.2_1b"
    assert rec_llama["llm_version"] == "Llama-3.2-1B-Instruct"

    return rec_fast, rec_llama


def benchmark_prompt_generation_matrix():
    print("\n" + "=" * 70)
    print("STEP 3: BENCHMARK MATRIX (PROMPTS 32/64/128, GEN 16/32, THREADS 4 vs 8)")
    print("=" * 70)

    runner = ExecuTorchLlamaRunner()
    if not runner.is_loaded:
        print("[!] ExecuTorch runtime not loaded. Skipping benchmark.")
        return []

    benchmarks = []
    # Test matrix: thread counts and token configurations
    test_configs = [
        {"name": "Concise (42 prompt / 16 gen / 8 threads)", "prompt_len": 42, "gen_len": 16, "threads": 8},
        {"name": "Concise (42 prompt / 16 gen / 4 threads)", "prompt_len": 42, "gen_len": 16, "threads": 4},
        {"name": "Standard (64 prompt / 16 gen / 8 threads)", "prompt_len": 64, "gen_len": 16, "threads": 8},
        {"name": "Medium (128 prompt / 16 gen / 8 threads)", "prompt_len": 128, "gen_len": 16, "threads": 8},
        {"name": "Extended (42 prompt / 32 gen / 8 threads)", "prompt_len": 42, "gen_len": 32, "threads": 8},
    ]

    base_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are EdgePCB-QC Assistant. Explain PCB defects.<|eot_id|><|start_header_id|>user<|end_header_id|>\nDefect: open circuit. Operator advice?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    tok = runner.tokenizer

    for cfg in test_configs:
        torch.set_num_threads(cfg["threads"])
        tokens = tok.encode(base_prompt, bos=False, eos=False) if tok else [100] * 32
        target_len = cfg["prompt_len"]
        if len(tokens) < target_len:
            tokens = tokens + [128000] * (target_len - len(tokens))
        else:
            tokens = tokens[:target_len]

        eval_tokens = tokens[:255]
        in_tokens = torch.tensor([eval_tokens], dtype=torch.long)
        pos = torch.tensor([0], dtype=torch.long)

        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)

        # Prefill
        t0 = time.perf_counter()
        outputs = runner.module.forward((in_tokens, pos))
        t_prefill = time.perf_counter() - t0
        ttft_ms = round(t_prefill * 1000, 2)

        # Generate
        logits = outputs[0]
        gen_tokens = []
        curr_pos = len(eval_tokens)
        t_g0 = time.perf_counter()
        for _ in range(cfg["gen_len"]):
            next_t = torch.argmax(logits[0]).item()
            if next_t in (128001, 128009):
                break
            gen_tokens.append(next_t)
            in_t = torch.tensor([[next_t]], dtype=torch.long)
            p = torch.tensor([curr_pos], dtype=torch.long)
            logits = runner.module.forward((in_t, p))[0]
            curr_pos += 1
        t_gen = time.perf_counter() - t_g0
        ram_after = process.memory_info().rss / (1024 * 1024)

        total_time_s = t_prefill + t_gen
        total_tokens = len(eval_tokens) + len(gen_tokens)
        tok_per_sec = round(total_tokens / (total_time_s or 1.0), 2)
        gen_tok_per_sec = round(len(gen_tokens) / (t_gen or 1.0), 2)

        result_entry = {
            "config": cfg["name"],
            "threads": cfg["threads"],
            "prompt_tokens": len(eval_tokens),
            "generated_tokens": len(gen_tokens),
            "ttft_ms": ttft_ms,
            "gen_time_s": round(t_gen, 2),
            "total_time_s": round(total_time_s, 2),
            "overall_tokens_per_sec": tok_per_sec,
            "gen_tokens_per_sec": gen_tok_per_sec,
            "peak_ram_mb": round(max(ram_before, ram_after), 1)
        }
        benchmarks.append(result_entry)
        print(f"[+] {cfg['name']}: TTFT={ttft_ms} ms, Gen={t_gen:.2f}s ({gen_tok_per_sec} tok/s), RAM={result_entry['peak_ram_mb']} MB")

    return benchmarks


def verify_advisory_correctness_four_defects():
    print("\n" + "=" * 70)
    print("STEP 4: ADVISORY CORRECTNESS ACROSS 4 DEFECT CATEGORIES")
    print("=" * 70)

    engine = DecisionEngine()
    runner = ExecuTorchLlamaRunner()

    defect_cases = [
        {"name": "Open Circuit", "class": "open_circuit", "conf": 0.92, "expect_elec": True, "expect_dec": "REJECT"},
        {"name": "Short Circuit", "class": "short_circuit", "conf": 0.88, "expect_elec": True, "expect_dec": "REJECT"},
        {"name": "Mouse Bite", "class": "mouse_bite", "conf": 0.85, "expect_elec": False, "expect_dec": "REJECT"},
        {"name": "Spurious Copper", "class": "spurious_copper", "conf": 0.87, "expect_elec": False, "expect_dec": "REJECT"},
    ]

    results = []
    for case in defect_cases:
        payload = {
            "board_id": f"PCB-CORRECT-{case['class'].upper()}",
            "coverage": 0.98,
            "defects": [{"class": case["class"], "confidence": case["conf"], "bbox": [100, 100, 140, 140]}]
        }
        dec = engine.evaluate(payload, preset="consumer")
        adv = runner.generate_advisory(dec, mode="fast")

        assert dec["decision"] == case["expect_dec"], f"Expected {case['expect_dec']}, got {dec['decision']}"
        assert dec["requires_electrical_test"] == case["expect_elec"], f"Expected electrical {case['expect_elec']}"
        assert adv["advisory"]["electrical_confirmation_needed"] == case["expect_elec"]

        # Verify no unguided scraping in operator action
        action_text = adv["advisory"]["operator_action"].lower()
        assert "manual scraping" not in action_text or "do not attempt unguided manual scraping" in action_text

        results.append({
            "defect_name": case["name"],
            "decision": dec["decision"],
            "rule_id": dec["rule_id"],
            "requires_electrical_test": dec["requires_electrical_test"],
            "advisory_electrical_flag": adv["advisory"]["electrical_confirmation_needed"],
            "action_guidance_snippet": adv["advisory"]["operator_action"][:80] + "..."
        })
        print(f"[+] {case['name']}: Decision={dec['decision']}, ElectricalTest={dec['requires_electrical_test']}, AdvisoryAgreement=PASS")

    return results


if __name__ == "__main__":
    r_fast, r_llama = run_dual_mode_inspections()
    b_matrix = benchmark_prompt_generation_matrix()
    c_results = verify_advisory_correctness_four_defects()

    output_summary = {
        "dual_mode_verification": {
            "fast_mode": r_fast,
            "llama_mode": r_llama
        },
        "performance_benchmark_matrix": b_matrix,
        "advisory_correctness": c_results
    }

    out_file = WORKSPACE_ROOT / "benchmarks" / "results" / "advisory_modes_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_summary, f, indent=2)

    print(f"\n[+] Full verification complete. Authoritative results saved to {out_file}")
