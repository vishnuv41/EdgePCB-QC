import cv2
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from inspection.cpu_inference import CPUInspector
from inspection.quality_assessment import assess_inspection_image
from inspection.coverage import calculate_inspection_coverage
from decision_engine.decision import DecisionEngine
from advisory.fallback import generate_fallback_advisory
from database.models import DatabaseManager

db_mgr = DatabaseManager(WORKSPACE_ROOT / "database" / "inspections.db")
engine = DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")

baseline_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
multidomain_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"

# Load sample image
img_path = WORKSPACE_ROOT / "datasets" / "inspected_captures" / "PCB-1334_1793781335.jpg"
img = cv2.imread(str(img_path))
assert img is not None, "Image failed to load"

# -------------------------------------------------------------
# STEP 1: RUN INFERENCE WITH BASELINE CHECKPOINT
# -------------------------------------------------------------
print("=" * 60)
print("1. RUNNING WITH BASELINE CHECKPOINT")
print("=" * 60)
inspector = CPUInspector(model_path=baseline_path, device="cpu")
meta_base = inspector.get_model_metadata()
print(f"Model Name: {meta_base['model_name']}")
print(f"Model Version: {meta_base['model_version']}")
print(f"Supported Domains: {meta_base['supported_domains']}")

cov_base = calculate_inspection_coverage(img)["coverage"]
qa_base = assess_inspection_image(img, coverage=cov_base, supported_domains=meta_base["supported_domains"])
defects_base = inspector.inspect(cv2.resize(img, (640, 640)))

payload_base = {
    "board_id": "TEST-SWITCH-BASELINE",
    "defects": defects_base,
    "coverage": cov_base,
    "quality_assessment": qa_base
}
dec_base = engine.evaluate(payload_base)
adv_base = generate_fallback_advisory(dec_base)

uuid_base = db_mgr.log_inspection(
    board_id="TEST-SWITCH-BASELINE",
    image_path=str(img_path),
    decision_record=dec_base,
    advisory_record=adv_base.model_dump(),
    advisory_metrics={"backend": "deterministic_fallback", "model_id": "ipc_expert_rules_engine", "status": "active"},
    timings={"vision_ms": 35.0, "advisory_ms": 0.5, "total_ms": 35.5},
    vision_model_version=meta_base["model_version"]
)
print(f"Logged Baseline Inspection ID: {uuid_base}")

# -------------------------------------------------------------
# STEP 2: SWITCH TO MULTI-DOMAIN CHECKPOINT DYNAMICALLY
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. SWITCHING TO MULTI-DOMAIN CHECKPOINT VIA load_checkpoint()")
print("=" * 60)
switch_success = inspector.load_checkpoint(multidomain_path)
assert switch_success is True
assert inspector.is_loaded is True

meta_multi = inspector.get_model_metadata()
print(f"Model Name: {meta_multi['model_name']}")
print(f"Model Version: {meta_multi['model_version']}")
print(f"Supported Domains: {meta_multi['supported_domains']}")
assert meta_multi["model_name"] == "Unified Multi-Domain YOLOv8n"
assert "multidomain" in meta_multi["model_version"].lower()

cov_multi = calculate_inspection_coverage(img)["coverage"]
qa_multi = assess_inspection_image(img, coverage=cov_multi, supported_domains=meta_multi["supported_domains"])
defects_multi = inspector.inspect(cv2.resize(img, (640, 640)))

payload_multi = {
    "board_id": "TEST-SWITCH-MULTIDOMAIN",
    "defects": defects_multi,
    "coverage": cov_multi,
    "quality_assessment": qa_multi
}
dec_multi = engine.evaluate(payload_multi)
adv_multi = generate_fallback_advisory(dec_multi)

uuid_multi = db_mgr.log_inspection(
    board_id="TEST-SWITCH-MULTIDOMAIN",
    image_path=str(img_path),
    decision_record=dec_multi,
    advisory_record=adv_multi.model_dump(),
    advisory_metrics={"backend": "deterministic_fallback", "model_id": "ipc_expert_rules_engine", "status": "active"},
    timings={"vision_ms": 35.0, "advisory_ms": 0.5, "total_ms": 35.5},
    vision_model_version=meta_multi["model_version"]
)
print(f"Logged Multi-Domain Inspection ID: {uuid_multi}")

# -------------------------------------------------------------
# STEP 3: VERIFY DATABASE PERSISTENCE OF BOTH MODEL VERSIONS
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. VERIFYING DATABASE RECORDS")
print("=" * 60)
rec_base = db_mgr.get_inspection(uuid_base)
rec_multi = db_mgr.get_inspection(uuid_multi)

print(f"Base Record -> board_id: {rec_base['board_id']}, vision_model_version: '{rec_base['vision_model_version']}', decision: {rec_base['decision']}")
print(f"Multi Record -> board_id: {rec_multi['board_id']}, vision_model_version: '{rec_multi['vision_model_version']}', decision: {rec_multi['decision']}")

assert "smoke" in rec_base['vision_model_version'].lower()
assert "multidomain" in rec_multi['vision_model_version'].lower()
assert rec_base['vision_model_version'] != rec_multi['vision_model_version']
print("\n[SUCCESS] Checkpoint switching verified with full database traceability!")
