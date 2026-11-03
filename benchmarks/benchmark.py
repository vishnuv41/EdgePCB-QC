"""
Independent Benchmark Suite for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Measures:
1. Vision Pipeline (Inference latency, FPS, CPU utilization, Peak RAM)
2. Advisory Pipeline (TTFT, response latency, tokens/sec, Peak RAM)
Outputs real, measured performance metrics to benchmarks/results/
"""

import os
import sys
import json
import time
import platform
import psutil
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.cpu_inference import CPUInspector
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from inspection.reference_compare import (
    load_reference, load_test_image, align_images, compare_images, highlight_differences
)

RESULTS_DIR = WORKSPACE_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_hardware_info():
    mem = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024**3), 2),
        "available_ram_gb": round(mem.available / (1024**3), 2),
        "target_deployment": "Raspberry Pi 5 (8GB) + Hailo-8/8L AI HAT+"
    }


def benchmark_vision_pipeline(num_runs: int = 30):
    print("\n" + "=" * 60)
    print("BENCHMARKING VISION PIPELINE (Pipeline A - CPU/Laptop)")
    print("=" * 60)

    test_img_dir = WORKSPACE_ROOT / "datasets" / "processed" / "deeppcb_yolo" / "images" / "test"
    test_images = list(test_img_dir.glob("*.jpg"))
    if not test_images:
        # Fallback to raw sample
        test_images = [
            WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
        ]

    inspector = CPUInspector()
    process = psutil.Process(os.getpid())

    latencies = []
    cpu_usages = []
    ram_usages = []

    # Warmup
    sample_img = cv2.imread(str(test_images[0]))
    for _ in range(3):
        inspector.inspect(sample_img)

    for i in range(num_runs):
        img_path = test_images[i % len(test_images)]
        img = cv2.imread(str(img_path))

        cpu_before = psutil.cpu_percent(interval=None)
        t0 = time.perf_counter()
        defects = inspector.inspect(img)
        t1 = time.perf_counter()
        cpu_after = psutil.cpu_percent(interval=None)

        dur_ms = (t1 - t0) * 1000.0
        latencies.append(dur_ms)
        cpu_usages.append(max(cpu_before, cpu_after))
        ram_usages.append(process.memory_info().rss / (1024 * 1024))

    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    metrics = {
        "num_runs": num_runs,
        "input_resolution": "640x640",
        "mean_latency_ms": round(mean_lat, 2),
        "min_latency_ms": round(float(np.min(latencies)), 2),
        "max_latency_ms": round(float(np.max(latencies)), 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 1),
        "mean_cpu_percent": round(float(np.mean(cpu_usages)), 1),
        "peak_ram_mb": round(float(np.max(ram_usages)), 1)
    }

    print(f"[+] Mean Latency:  {metrics['mean_latency_ms']} ms (P95: {metrics['p95_latency_ms']} ms)")
    print(f"[+] Throughput:    {metrics['throughput_fps']} FPS")
    print(f"[+] Peak Process RAM: {metrics['peak_ram_mb']} MB")
    return metrics


def benchmark_reference_comparison(num_runs: int = 15):
    print("\n" + "=" * 60)
    print("BENCHMARKING GOLDEN REFERENCE COMPARISON (OpenCV)")
    print("=" * 60)

    sample_test = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
    sample_temp = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_temp.jpg"

    if not sample_test.exists() or not sample_temp.exists():
        return None

    test_img = load_test_image(sample_test)
    temp_img = load_reference(sample_temp)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        aligned, _ = align_images(test_img, temp_img)
        diff, mask = compare_images(aligned, temp_img)
        _, anomalies = highlight_differences(aligned, mask)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    mean_lat = float(np.mean(latencies))
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0

    metrics = {
        "num_runs": num_runs,
        "mean_latency_ms": round(mean_lat, 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "throughput_fps": round(fps, 1)
    }
    print(f"[+] Reference Comparison Latency: {metrics['mean_latency_ms']} ms ({metrics['throughput_fps']} FPS)")
    return metrics


def benchmark_advisory_pipeline(num_runs: int = 25):
    print("\n" + "=" * 60)
    print("BENCHMARKING ADVISORY PIPELINE (Pipeline B - Reasoning)")
    print("=" * 60)

    runner = ExecuTorchLlamaRunner()
    payload = {
        "board_id": "PCB-BENCH-001",
        "decision": "REJECT",
        "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT",
        "rule_description": "Detected high-confidence open circuit",
        "defects": [
            {"class": "open_circuit", "confidence": 0.92, "severity": "critical", "location": "upper-right"},
            {"class": "spur", "confidence": 0.84, "severity": "major", "location": "lower-left"}
        ],
        "requires_electrical_test": True
    }

    latencies = []
    ttfts = []
    ram_usages = []
    process = psutil.Process(os.getpid())

    # Warmup
    for _ in range(2):
        runner.generate_advisory(payload)

    for _ in range(num_runs):
        res = runner.generate_advisory(payload)
        m = res.get("metrics", {})
        latencies.append(m.get("latency_ms", 0.0))
        ttfts.append(m.get("ttft_ms", 0.0))
        ram_usages.append(process.memory_info().rss / (1024 * 1024))

    mean_lat = float(np.mean(latencies))
    mean_ttft = float(np.mean(ttfts))

    metrics = {
        "num_runs": num_runs,
        "backend": runner.generate_advisory(payload)["metrics"]["backend"],
        "mean_latency_ms": round(mean_lat, 2),
        "mean_ttft_ms": round(mean_ttft, 2),
        "peak_ram_mb": round(float(np.max(ram_usages)), 1)
    }

    print(f"[+] Advisory Latency: {metrics['mean_latency_ms']} ms")
    print(f"[+] Time To First Token (TTFT): {metrics['mean_ttft_ms']} ms")
    print(f"[+] Peak Process RAM: {metrics['peak_ram_mb']} MB")
    return metrics


def run_full_benchmark():
    hw = get_hardware_info()
    vis = benchmark_vision_pipeline()
    ref = benchmark_reference_comparison()
    adv = benchmark_advisory_pipeline()

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_data = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_environment": hw,
        "vision_metrics": vis,
        "reference_comparison_metrics": ref,
        "advisory_metrics": adv
    }

    out_file = RESULTS_DIR / f"benchmark_{timestamp_str}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print(f"[+] Benchmark results saved to: {out_file}")
    print("=" * 60)
    return out_file


if __name__ == "__main__":
    run_full_benchmark()
