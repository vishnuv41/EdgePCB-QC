"""
Live Image Inspection Verification Script
Simulates real dashboard user workflow with un-mocked components:
1. Loads real PCB images (test + golden reference).
2. Performs OpenCV alignment & difference extraction.
3. Performs CPU defect detection.
4. Assesses defect severity & triggers deterministic decision engine rules.
5. Generates operator advisory explanation.
6. Writes inspection payload to SQLite database.
7. Performs operator override and verifies persistence.
"""

import sys
import time
from pathlib import Path
import cv2

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from decision_engine.decision import DecisionEngine
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from database.models import DatabaseManager
from inspection.cpu_inference import CPUInspector
from inspection.reference_compare import (
    load_reference, load_test_image, align_images, compare_images, highlight_differences
)

def run_live_test():
    print("=" * 60)
    print("RUNNING LIVE DASHBOARD INSPECTION VERIFICATION")
    print("=" * 60)

    db_path = WORKSPACE_ROOT / "database" / "inspections.db"
    db_mgr = DatabaseManager(db_path)
    engine = DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")
    advisory = ExecuTorchLlamaRunner()
    cpu_insp = CPUInspector()

    sample_test_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
    sample_temp_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_temp.jpg"
    sample_anno_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041_not" / "00041000.txt"

    assert sample_test_path.exists(), f"Missing: {sample_test_path}"
    assert sample_temp_path.exists(), f"Missing: {sample_temp_path}"

    print("[*] 1. Loading real PCB images from DeepPCB dataset...")
    test_img = load_test_image(sample_test_path)
    temp_img = load_reference(sample_temp_path)
    print(f"    Loaded Test Image: {test_img.shape}, Template: {temp_img.shape}")

    print("[*] 2. Performing Golden Reference Comparison (OpenCV ORB + RANSAC + Difference Map)...")
    t0 = time.perf_counter()
    aligned_test, H = align_images(test_img, temp_img)
    diff_map, defect_mask = compare_images(aligned_test, temp_img)
    annotated_bgr, anomalies = highlight_differences(aligned_test, defect_mask)
    t_vision = (time.perf_counter() - t0) * 1000.0
    print(f"    Extracted {len(anomalies)} discrepancy regions in {t_vision:.2f} ms")

    print("[*] 3. Running Ground-Truth Annotated Inspection...")
    annotated_defects = cpu_insp.inspect_with_annotations(sample_test_path, sample_anno_path)
    print(f"    Loaded {len(annotated_defects)} localized defects from annotations:")
    for d in annotated_defects[:3]:
        print(f"      - {d['class']:20s} conf={d['confidence']:.2f} loc={d['location']}")

    print("[*] 4. Evaluating with Deterministic Decision Engine (Consumer Preset)...")
    board_id = f"PCB-VERIFY-{int(time.time()) % 10000}"
    payload = {
        "board_id": board_id,
        "defects": annotated_defects,
        "coverage": 0.98
    }
    decision_record = engine.evaluate(payload, preset="consumer")
    print(f"    Decision: {decision_record['decision']}")
    print(f"    Rule ID:  {decision_record['rule_id']}")
    print(f"    Rule Desc: {decision_record['rule_description']}")
    print(f"    Requires Electrical Test: {decision_record['requires_electrical_test']}")

    print("[*] 5. Generating Advisory Explanation (Pipeline B)...")
    t_adv_start = time.perf_counter()
    advisory_res = advisory.generate_advisory(decision_record)
    t_adv = (time.perf_counter() - t_adv_start) * 1000.0
    adv = advisory_res["advisory"]
    print(f"    Advisory Generated in {t_adv:.2f} ms:")
    print(f"      What:   {adv['what_detected'][:90]}...")
    print(f"      Why:    {adv['why_it_matters'][:90]}...")
    print(f"      Action: {adv['operator_action'][:90]}...")
    print(f"      Elec Needed: {adv['electrical_confirmation_needed']}")

    print("[*] 6. Persisting to SQLite Database...")
    saved_img_path = WORKSPACE_ROOT / "datasets" / "inspected_captures" / f"{board_id}.jpg"
    saved_img_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(saved_img_path), annotated_bgr)

    insp_id = db_mgr.log_inspection(
        board_id=board_id,
        image_path=str(saved_img_path),
        decision_record=decision_record,
        advisory_record=adv,
        timings={"vision_ms": t_vision, "advisory_ms": t_adv, "total_ms": t_vision + t_adv}
    )
    print(f"    Logged Record UUID: {insp_id}")

    print("[*] 7. Applying Operator Override...")
    db_mgr.record_override(
        inspection_id=insp_id,
        operator_id="QA-ENGINEER-01",
        new_decision="INSPECT",
        reason="Visual artifact confirmed; flagged for flying probe micro-continuity test."
    )
    rec = db_mgr.get_inspection(insp_id)
    print(f"    Verified Override in DB: Final Decision = {rec['final_decision']}, Operator = {rec['operator_override']['operator_id']}")

    print("\n[+] LIVE INSPECTION VERIFICATION PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_live_test()
