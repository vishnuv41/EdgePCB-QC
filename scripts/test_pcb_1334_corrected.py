import cv2
import json
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner
from inspection.quality_assessment import assess_inspection_image
from inspection.coverage import calculate_inspection_coverage
from decision_engine.decision import DecisionEngine
from advisory.fallback import generate_fallback_advisory

WORKSPACE_ROOT = Path(".").resolve()
img_path = WORKSPACE_ROOT / "datasets" / "inspected_captures" / "PCB-1334_1793781335.jpg"
img = cv2.imread(str(img_path))
print(f"Loaded PCB-1334 image: {img.shape}")

engine = DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")

baseline_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
multidomain_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"

print("\n" + "=" * 70)
print("TEST 1: PCB-1334 ON BASELINE MODEL (DeepPCB Smoke Monochrome)")
print("=" * 70)

insp_base = CPUInspector(model_path=baseline_path, device="cpu")
meta_base = insp_base.get_model_metadata()
print(f"Active model: {meta_base['model_name']}")
print(f"Supported domains: {meta_base['supported_domains']}")

cov_base = calculate_inspection_coverage(img)
print(f"Calculated coverage: {cov_base['coverage']*100:.2f}% ({cov_base['method']})")

qa_base = assess_inspection_image(
    img,
    coverage=cov_base["coverage"],
    min_coverage_required=0.90,
    supported_domains=meta_base["supported_domains"]
)
print(f"Quality status: {qa_base['image_quality_status']}")
print(f"Domain status: {qa_base['domain_status']}")
print(f"Reason for inspect: {qa_base['reason_for_inspect']}")

# Tiled inspection
tiler_base = TiledInferenceRunner(insp_base)
defects_base = tiler_base.inspect_tiled(img)
print(f"Detections count: {len(defects_base)}")

payload_base = {
    "board_id": "PCB-1334-BASELINE",
    "defects": defects_base,
    "coverage": cov_base["coverage"],
    "quality_assessment": qa_base
}
dec_base = engine.evaluate(payload_base, preset="consumer")
print(f"Decision: {dec_base['decision']}")
print(f"Triggered Rule: {dec_base['rule_id']}")
print(f"Rule Description: {dec_base['rule_description']}")
print(f"Requires Electrical Test: {dec_base['requires_electrical_test']}")

adv_base = generate_fallback_advisory(dec_base)
print(f"Advisory What: {adv_base.what_detected}")
print(f"Advisory Caveat: {adv_base.confidence_caveat}")

print("\n" + "=" * 70)
print("TEST 2: PCB-1334 ON MULTI-DOMAIN MODEL (Unified Color + Monochrome)")
print("=" * 70)

insp_multi = CPUInspector(model_path=multidomain_path, device="cpu")
meta_multi = insp_multi.get_model_metadata()
print(f"Active model: {meta_multi['model_name']}")
print(f"Supported domains: {meta_multi['supported_domains']}")

cov_multi = calculate_inspection_coverage(img)
print(f"Calculated coverage: {cov_multi['coverage']*100:.2f}% ({cov_multi['method']})")

qa_multi = assess_inspection_image(
    img,
    coverage=cov_multi["coverage"],
    min_coverage_required=0.90,
    supported_domains=meta_multi["supported_domains"]
)
print(f"Quality status: {qa_multi['image_quality_status']}")
print(f"Domain status: {qa_multi['domain_status']}")
print(f"Reason for inspect: {qa_multi['reason_for_inspect']}")

tiler_multi = TiledInferenceRunner(insp_multi)
defects_multi = tiler_multi.inspect_tiled(img)
print(f"Detections count: {len(defects_multi)}")
for idx, d in enumerate(defects_multi, 1):
    print(f"  Defect {idx}: class={d['class']} conf={d['confidence']:.3f} loc={d['location']}")

payload_multi = {
    "board_id": "PCB-1334-MULTIDOMAIN",
    "defects": defects_multi,
    "coverage": cov_multi["coverage"],
    "quality_assessment": qa_multi
}
dec_multi = engine.evaluate(payload_multi, preset="consumer")
print(f"Decision: {dec_multi['decision']}")
print(f"Triggered Rule: {dec_multi['rule_id']}")
print(f"Rule Description: {dec_multi['rule_description']}")
print(f"Requires Electrical Test: {dec_multi['requires_electrical_test']}")

adv_multi = generate_fallback_advisory(dec_multi)
print(f"Advisory What: {adv_multi.what_detected}")
print(f"Advisory Caveat: {adv_multi.confidence_caveat}")
