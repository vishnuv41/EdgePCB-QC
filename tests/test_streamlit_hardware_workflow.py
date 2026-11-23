"""
End-to-End Tests for EdgePCB-QC Hardware Workflow
Bharat AI-SoC Challenge 2026-27 | PS5

Tests the complete physical pipeline simulation:
Mock Camera Capture -> Preprocessing -> InferenceManager -> DecisionEngine -> Llama Advisory -> SQLite Persistence
"""

import time
import pytest
import numpy as np
from pathlib import Path

from hardware.camera.camera_manager import CameraManager
from hardware.inference.inference_manager import InferenceManager
from decision_engine.decision import DecisionEngine
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from database.models import DatabaseManager

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_complete_hardware_inspection_flow():
    # 1. Initialize subsystems
    cam_mgr = CameraManager()
    inf_mgr = InferenceManager(backend_mode="auto")
    engine = DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")
    advisory = ExecuTorchLlamaRunner()
    db = DatabaseManager(WORKSPACE_ROOT / "database" / "inspections.db")

    # 2. Start mock camera and capture frame
    started = cam_mgr.start_camera("mock:synthetic", resolution=(640, 640), framerate=15)
    assert started
    time.sleep(0.2)

    captured_frame = cam_mgr.capture_still()
    assert captured_frame is not None
    assert captured_frame.shape == (640, 640, 3)

    # 3. Vision inference
    t0 = time.perf_counter()
    res = inf_mgr.inspect(captured_frame)
    latency_ms = res["latency_ms"]
    detections = res["detections"]
    assert "backend" in res

    # 4. Deterministic decision engine evaluation
    eval_payload = {
        "board_id": f"TEST-HW-{int(time.time())}",
        "defects": detections,
        "coverage": 1.0,
        "quality_flags": []
    }
    eval_res = engine.evaluate(
        eval_payload,
        preset="consumer"
    )
    verdict = eval_res["decision"]
    assert verdict in ["PASS", "INSPECT", "REJECT"]

    # 5. Llama Advisory generation (deterministic fallback or ExecuTorch)
    board_id = eval_payload["board_id"]
    adv_input = {
        "board_id": board_id,
        "decision": verdict,
        "rule_id": eval_res["rule_id"],
        "rule_description": eval_res["rule_description"],
        "quality_preset": "consumer",
        "coverage_score": 1.0,
        "defects": detections,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    adv_out = advisory.generate_advisory(adv_input, mode="fast")
    assert "advisory" in adv_out
    assert "what_detected" in adv_out["advisory"]
    assert "operator_action" in adv_out["advisory"]

    # 6. Database persistence
    rec_id = db.log_inspection(
        board_id=board_id,
        image_path=f"captures/{board_id}.jpg",
        decision_record=eval_res,
        advisory_record=adv_out.get("advisory"),
        advisory_metrics=adv_out.get("metrics"),
        timings={"total_ms": round((time.perf_counter() - t0) * 1000, 2)},
        inspection_source="test",
        vision_model_version="YOLOv8n-pcb-v1.0",
        image_sha256="0" * 64,
        model_sha256="66d8d225fe..."
    )
    assert rec_id is not None

    # Verify retrieval
    retrieved = db.get_inspection(rec_id)
    assert retrieved is not None
    assert retrieved["board_id"] == board_id
    assert retrieved["decision"] == verdict

    # Clean up camera
    cam_mgr.stop_camera()
    assert not cam_mgr.is_streaming
