import cv2
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner
from inspection.quality_assessment import assess_inspection_image
from inspection.coverage import calculate_inspection_coverage
from decision_engine.decision import DecisionEngine
from advisory.fallback import generate_fallback_advisory
from database.models import DatabaseManager

db_mgr = DatabaseManager(WORKSPACE_ROOT / "database" / "inspections.db")
engine = DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")

multidomain_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
inspector = CPUInspector(model_path=multidomain_path, device="cpu")
tiler = TiledInferenceRunner(inspector)
meta = inspector.get_model_metadata()

img_path = WORKSPACE_ROOT / "datasets" / "inspected_captures" / "PCB-1334_1793781335.jpg"
img = cv2.imread(str(img_path))

# Compute coverage
cov_res = calculate_inspection_coverage(img)
coverage_val = cov_res["coverage"]

# Run detector
t0_v = time.perf_counter()
defects = tiler.inspect_tiled(img)
t1_v = time.perf_counter()
v_latency = (t1_v - t0_v) * 1000.0

# Assess quality & domain
quality_assessment = assess_inspection_image(
    img,
    coverage=coverage_val,
    min_coverage_required=0.90,
    supported_domains=meta["supported_domains"]
)

# Decision
payload = {
    "board_id": "PCB-1334-AUDIT-REVISED",
    "defects": defects,
    "coverage": coverage_val,
    "quality_assessment": quality_assessment
}
decision_record = engine.evaluate(payload, preset="consumer")

# Advisory
t0_a = time.perf_counter()
adv = generate_fallback_advisory(decision_record)
t1_a = time.perf_counter()
a_latency = (t1_a - t0_a) * 1000.0

# Log
rec_id = db_mgr.log_inspection(
    board_id="PCB-1334-AUDIT-REVISED",
    image_path=str(img_path),
    decision_record=decision_record,
    advisory_record=adv.model_dump(),
    advisory_metrics={
        "backend": "deterministic_fallback",
        "model_id": "ipc_expert_rules_engine",
        "status": "active",
        "latency_ms": round(a_latency, 2)
    },
    timings={
        "vision_ms": round(v_latency, 2),
        "advisory_ms": round(a_latency, 2),
        "total_ms": round(v_latency + a_latency, 2)
    },
    vision_model_version=meta["model_version"]
)

print(f"Logged revised audit inspection {rec_id} for PCB-1334-AUDIT-REVISED")
print(f"Decision: {decision_record['decision']}")
print(f"Rule: {decision_record['rule_id']}")
print(f"Coverage: {coverage_val*100:.1f}%")
print(f"Defect count: {len(defects)}")
print(f"Requires electrical test: {decision_record['requires_electrical_test']}")
print(f"Model version: {meta['model_version']}")
print(f"Advisory: {adv.what_detected}")
print(f"Caveat: {adv.confidence_caveat}")
