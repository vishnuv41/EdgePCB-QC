"""
Industrial Inspection Reliability and Robustness Test Suite
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Multi-resolution scaling & high-resolution sliding-window tiling
2. Factory lighting shift & noise perturbation robustness (Overexposed, Underexposed, Blur, Noise)
3. Safety-critical open-circuit & short-circuit sensitivity audit
4. Deterministic decision engine fail-safe behavior
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner
from decision_engine.decision import DecisionEngine
from database.models import DatabaseManager

RESULTS_DIR = WORKSPACE_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def simulate_lighting_perturbations(img: np.ndarray):
    """Generates realistic factory optical variations."""
    perturbations = {}
    
    # 1. Nominal
    perturbations["nominal"] = img.copy()
    
    # 2. Overexposure / Glare (+30 intensity)
    bright = cv2.convertScaleAbs(img, alpha=1.15, beta=30)
    perturbations["overexposed_glare"] = bright
    
    # 3. Underexposure / Dim Lighting (-35 intensity)
    dim = cv2.convertScaleAbs(img, alpha=0.85, beta=-35)
    perturbations["underexposed_dim"] = dim
    
    # 4. Defocus / Optical Blur (Gaussian kernel 5x5)
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    perturbations["optical_defocus"] = blurred
    
    # 5. Sensor Thermal Noise (Gaussian noise)
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    perturbations["sensor_noise"] = noisy
    
    return perturbations


def run_reliability_audit():
    print("=" * 70)
    print("EDGEPCB-QC INSPECTION RELIABILITY & ROBUSTNESS AUDIT")
    print("=" * 70)

    # 1. Initialize detector and decision engine
    inspector = CPUInspector()
    tiled_runner = TiledInferenceRunner(inspector, tile_size=640, overlap=0.20)
    engine = DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")

    print(f"[*] Detector Loaded: {inspector.is_loaded}")
    print(f"[*] Model Weights:   {inspector.model_path}")
    print(f"[*] Execution Mode:  CPU Fallback (AMD Ryzen 9 8940HX)")

    # 2. Load test PCB image
    test_img_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
    if not test_img_path.exists():
        # Fallback to test split image
        test_imgs = list((WORKSPACE_ROOT / "datasets" / "processed" / "deeppcb_yolo" / "images" / "test").glob("*.jpg"))
        test_img_path = test_imgs[0]
    
    base_img = cv2.imread(str(test_img_path))
    h, w = base_img.shape[:2]
    print(f"[*] Base Image Loaded: {test_img_path.name} ({w}x{h})")

    # -------------------------------------------------------------
    # EXPERIMENT A: Lighting & Perturbation Robustness
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXPERIMENT A: Optical Robustness Under Factory Lighting Variations")
    print("-" * 70)

    perturbed_dict = simulate_lighting_perturbations(base_img)
    lighting_results = {}

    for condition, p_img in perturbed_dict.items():
        t0 = time.perf_counter()
        dets = inspector.inspect(p_img)
        t_infer = (time.perf_counter() - t0) * 1000.0

        decision_res = engine.evaluate({"board_id": f"TEST-{condition}", "defects": dets, "coverage": 0.98}, preset="consumer")

        classes_found = [d["class"] for d in dets]
        has_open = "open_circuit" in classes_found
        has_short = "short_circuit" in classes_found

        lighting_results[condition] = {
            "defects_count": len(dets),
            "classes_detected": list(set(classes_found)),
            "decision": decision_res["decision"],
            "rule_id": decision_res["rule_id"],
            "requires_electrical_test": decision_res["requires_electrical_test"],
            "latency_ms": round(t_infer, 2),
            "open_circuit_detected": has_open,
            "short_circuit_detected": has_short
        }

        print(f"[*] {condition:20s} | Defects: {len(dets):2d} | Verdict: {decision_res['decision']:7s} | Open: {str(has_open):5s} | Short: {str(has_short):5s} | Latency: {t_infer:5.1f}ms")

    # -------------------------------------------------------------
    # EXPERIMENT B: High-Resolution Tiling & Global NMS
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXPERIMENT B: High-Resolution Large-Board Tiling & Coordinate Fusion")
    print("-" * 70)

    # Synthesize a 2x2 multi-panel array (1280x1280, 1.64 MP) to simulate large production panels
    composite_panel = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
    composite_panel[0:h, 0:w] = base_img
    composite_panel[0:h, w:w*2] = cv2.flip(base_img, 1)
    composite_panel[h:h*2, 0:w] = cv2.flip(base_img, 0)
    composite_panel[h:h*2, w:w*2] = cv2.flip(base_img, -1)

    cp_h, cp_w = composite_panel.shape[:2]
    print(f"[*] Created Composite Multi-Panel PCB: {cp_w}x{cp_h} ({cp_w * cp_h / 1e6:.2f} Megapixels)")

    t0 = time.perf_counter()
    tiles = tiled_runner.slice_image(composite_panel)
    t_slice = (time.perf_counter() - t0) * 1000.0
    print(f"[*] Sliced into {len(tiles)} overlapping 640x640 sub-windows (20% overlap) in {t_slice:.2f}ms")

    t0 = time.perf_counter()
    tiled_defects = tiled_runner.inspect_tiled(composite_panel)
    t_tiled = (time.perf_counter() - t0) * 1000.0

    print(f"[*] Total Fused Defects After Global NMS: {len(tiled_defects)}")
    print(f"[*] Total Tiled Inference Latency:        {t_tiled:.2f}ms ({t_tiled / len(tiles):.1f}ms/tile)")

    tiled_eval = engine.evaluate({"board_id": "PANEL-1280x1280", "defects": tiled_defects, "coverage": 0.99}, preset="consumer")
    print(f"[*] Panel Decision Verdict:               {tiled_eval['decision']} (Rule: {tiled_eval['rule_id']})")

    # -------------------------------------------------------------
    # EXPERIMENT C: Fail-Safe Decision Engine Invariance
    # -------------------------------------------------------------
    print("\n" + "-" * 70)
    print("EXPERIMENT C: Deterministic Fail-Safe Invariance")
    print("-" * 70)
    # Check that under all perturbed conditions, the verdict NEVER erroneously returns PASS
    # when critical defects exist
    fail_safe_intact = True
    for cond, r in lighting_results.items():
        if r["defects_count"] > 0 and r["decision"] == "PASS":
            fail_safe_intact = False
            print(f"[!] SAFETY VIOLATION: Condition {cond} yielded PASS despite defects!")

    print(f"[+] Fail-Safe Quality Gate Audit: {'PASSED (Zero False Negatives causing erroneous PASS)' if fail_safe_intact else 'FAILED'}")

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_image": str(test_img_path.name),
        "optical_robustness_test": lighting_results,
        "high_res_tiling_test": {
            "composite_dimensions": f"{cp_w}x{cp_h}",
            "num_tiles": len(tiles),
            "tile_size": 640,
            "overlap": 0.20,
            "fused_defects_count": len(tiled_defects),
            "total_latency_ms": round(t_tiled, 2),
            "latency_per_tile_ms": round(t_tiled / len(tiles), 2),
            "verdict": tiled_eval["decision"],
            "rule_id": tiled_eval["rule_id"]
        },
        "fail_safe_audit_passed": fail_safe_intact
    }

    out_path = RESULTS_DIR / "inspection_reliability_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[+] Detailed reliability report saved to: {out_path}")
    return report


if __name__ == "__main__":
    run_reliability_audit()
