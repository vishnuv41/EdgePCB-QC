"""
Tests for Camera Abstraction & Camera Manager
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import time
from pathlib import Path
import pytest
import numpy as np
from hardware.camera.camera_manager import CameraManager, CameraStatus, MockCamera, get_camera_manager
from hardware.dataset.dataset_browser import discover_dataset_catalog, load_dataset_image
from hardware.network.network_status import get_network_diagnostics
from database.models import DatabaseManager


def test_camera_manager_initial_state():
    cm = CameraManager()
    assert cm.status == CameraStatus.DISCONNECTED
    assert not cm.is_streaming
    assert cm.get_latest_frame() is None


def test_camera_discovery():
    cm = CameraManager()
    cameras = cm.discover_cameras()
    assert isinstance(cameras, list)
    assert len(cameras) >= 1  # Mock camera is always discovered

    mock_cams = [c for c in cameras if c["type"] == "mock"]
    assert len(mock_cams) == 1
    assert mock_cams[0]["id"] == "mock:synthetic"


def test_mock_camera_lifecycle():
    cam = MockCamera(resolution=(640, 640), framerate=15)
    assert not cam.is_connected

    connected = cam.connect()
    assert connected
    assert cam.is_connected

    frame = cam.get_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (640, 640, 3)

    info = cam.get_info()
    assert info["connected"] is True
    assert "640x640" in info["resolution"]

    cam.disconnect()
    assert not cam.is_connected
    assert cam.get_frame() is None


def test_camera_manager_mock_streaming():
    cm = CameraManager()
    started = cm.start_camera("mock:synthetic", resolution=(640, 640), framerate=15)
    assert started
    assert cm.status == CameraStatus.STREAMING
    assert cm.is_streaming

    time.sleep(0.3)
    frame = cm.get_latest_frame()
    assert frame is not None
    assert frame.shape == (640, 640, 3)

    still = cm.capture_still()
    assert still is not None
    assert still.shape == (640, 640, 3)

    status_info = cm.get_status_info()
    assert status_info["status"] == "STREAMING"
    assert status_info["camera_type"] == "mock"

    cm.stop_camera()
    assert cm.status == CameraStatus.DISCONNECTED
    assert not cm.is_streaming
    assert cm.get_latest_frame() is None


def test_camera_manager_invalid_device_handling():
    cm = CameraManager()
    # Attempting to start non-existent hardware should fail honestly
    started = cm.start_camera("nonexistent:device:99")
    assert not started
    assert cm.status == CameraStatus.DISCONNECTED


def test_camera_persistence_across_multiple_discover_calls():
    """Verify camera streaming persists without disconnection across repeated device discovery calls (Streamlit rerun simulation)."""
    cm = CameraManager()
    started = cm.start_camera("mock:synthetic", resolution=(640, 640), framerate=15)
    assert started
    assert cm.is_streaming

    # Repeatedly call discover_cameras to simulate Streamlit reruns
    for _ in range(5):
        cams = cm.discover_cameras()
        assert len(cams) >= 1
        assert cm.is_streaming, "Camera disconnected during discover_cameras rerun simulation!"
        assert cm.status == CameraStatus.STREAMING

    frame = cm.get_latest_frame()
    assert frame is not None
    assert frame.shape == (640, 640, 3)

    cm.stop_camera()
    assert not cm.is_streaming
    assert cm.status == CameraStatus.DISCONNECTED


def test_camera_manager_singleton_idempotency():
    """Verify get_camera_manager returns the persistent singleton and start_camera is idempotent."""
    mgr1 = get_camera_manager()
    mgr2 = get_camera_manager()
    assert mgr1 is mgr2

    # Starting camera on mgr1 should reflect in mgr2
    started = mgr1.start_camera("mock:synthetic", resolution=(640, 640), framerate=15)
    assert started
    assert mgr2.is_streaming

    # Re-calling start_camera with same parameters should return True immediately (idempotent)
    second_start = mgr2.start_camera("mock:synthetic", resolution=(640, 640), framerate=15)
    assert second_start is True
    assert mgr1.is_streaming

    mgr2.stop_camera()
    assert not mgr1.is_streaming


def test_dataset_browser_in_place_discovery_and_loading():
    """Verify Mode B dataset browser discovers real images in-place and loads them with zero file duplication."""
    catalog = discover_dataset_catalog(max_per_dataset=50)
    assert isinstance(catalog, dict)
    assert "DeepPCB (High-Contrast Monochrome Scans)" in catalog
    assert "DsPCBSD+ (Color Industrial Bare Boards)" in catalog

    total_images = sum(len(records) for records in catalog.values())
    assert total_images > 0, "No dataset images discovered in project datasets/ directory!"

    # Find the first valid image record and load it
    sample_record = None
    for ds_name, records in catalog.items():
        if records:
            sample_record = records[0]
            break

    assert sample_record is not None
    img_bgr, meta = load_dataset_image(sample_record["path"])
    assert img_bgr is not None
    assert isinstance(img_bgr, np.ndarray)
    assert img_bgr.ndim == 3
    assert meta["width"] > 0
    assert meta["height"] > 0
    assert meta["size_kb"] > 0
    assert Path(sample_record["path"]).exists()


def test_dataset_inspection_demo_provenance_logging(tmp_path):
    """Verify inspections from Mode B are logged with strictly 'demo' provenance tag, never 'production'."""
    db_path = tmp_path / "test_demo_provenance.db"
    db = DatabaseManager(db_path)

    decision_res = {
        "board_id": "DEMO-TEST-001",
        "decision": "PASS",
        "rule_id": "RULE_001_CLEAN_BOARD_AUTO_PASS",
        "rule_description": "Clean board with 0 localized defects",
        "quality_preset": "consumer"
    }

    db.log_inspection(
        board_id="DEMO-TEST-001",
        image_path="datasets/raw/deeppcb/test_001.jpg",
        decision_record=decision_res,
        advisory_record={"what_detected": "None", "operator_action": "Pass"},
        advisory_metrics={"latency_ms": 0.5},
        timings={"vision_ms": 12.5, "total_ms": 15.0},
        inspection_source="demo",
        vision_model_version="YOLOv8n-pcb-v1.0",
        image_sha256="abc123mockhash",
        model_sha256="66d8d225fe..."
    )

    inspections = db.list_inspections(limit=10)
    assert len(inspections) == 1
    assert inspections[0]["board_id"] == "DEMO-TEST-001"
    assert inspections[0]["inspection_source"] == "demo"
    assert inspections[0]["inspection_source"] != "production"
    assert inspections[0]["inspection_source"] != "real"


def test_network_diagnostics_edge_access_urls():
    """Verify local network diagnostics formats full accessible URLs without truncation."""
    diag = get_network_diagnostics(port=8501)
    assert diag["offline_mode"] is True
    assert ":8501" in diag["localhost_url"]
    assert ":8501" in diag["primary_url"]
    assert ":8501" in diag["hostname_url"]
    assert "interfaces" in diag
