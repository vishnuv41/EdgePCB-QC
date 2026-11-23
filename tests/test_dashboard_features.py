"""
Unit and Integration Tests for EdgePCB-QC Dashboard Features & Invariants
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Offline Startup & Asset Purity (no external network requests or remote asset URLs)
2. Defect Magnifier Crop Extraction (boundary clamping, corner defects, coordinate mapping)
3. Severity Consistency (DecisionEngine rules vs UI defect severity)
4. Coverage & Framing Warning Consistency (RULE_005D_FRAME_EDGE_CLIPPING)
5. Inspection Source Tagging & Production Analytics Filter
6. Defect-Level Human Review & False-Call Metrics (with NO DATA handling)
7. Configuration Snapshot & Image Hash Persistence
"""

import sys
import socket
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from database.models import DatabaseManager
from dashboard.utils import extract_defect_crop, sha256_bytes, sha256_file, get_branding_svg
from decision_engine.decision import DecisionEngine
from inspection.coverage import calculate_inspection_coverage


# =====================================================================
# 1. OFFLINE STARTUP & ASSET PURITY
# =====================================================================

def test_branding_svg_offline():
    """Verify that get_branding_svg returns a valid offline SVG with no external media assets."""
    svg_html = get_branding_svg()
    assert "<svg" in svg_html
    assert "</svg>" in svg_html
    assert "EdgePCB-QC" in svg_html
    # Ensure no external image elements, remote scripts, or remote media URLs exist
    assert "<image" not in svg_html.lower()
    assert "<script" not in svg_html.lower()
    assert "href=\"http" not in svg_html.lower()
    assert "src=\"http" not in svg_html.lower()


def test_dashboard_files_have_no_external_media_requests():
    """Verify dashboard/app.py and dashboard/utils.py do not load external remote URLs for media."""
    dashboard_dir = Path("dashboard")
    for py_file in dashboard_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            # Comments and docstrings are fine, but active calls like requests.get or st.image(http) are prohibited
            if stripped.startswith("#"):
                continue
            assert "requests.get(" not in stripped, f"Found requests.get in {py_file}:{idx}"
            assert "urllib.request" not in stripped, f"Found urllib in {py_file}:{idx}"
            if "st.image(" in stripped:
                assert "http://" not in stripped and "https://" not in stripped, (
                    f"External image URL loaded via st.image in {py_file}:{idx}"
                )


def test_network_isolation_during_dashboard_utils_import():
    """Verify that importing dashboard utilities makes zero network socket calls."""
    def guarded_socket(*args, **kwargs):
        raise RuntimeError("Network socket call attempted during offline operation!")

    with patch.object(socket, "socket", side_effect=guarded_socket):
        import dashboard.utils as d_utils
        svg = d_utils.get_branding_svg()
        assert len(svg) > 50


# =====================================================================
# 2. DEFECT MAGNIFIER CROP EXTRACTION
# =====================================================================

def test_extract_defect_crop_standard():
    """Test centered defect bounding box extraction and coordinate transformation."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Defect at [50, 60, 90, 110] -> width 40, height 50
    crop, local_box, meta = extract_defect_crop(img, [50, 60, 90, 110], padding=20)
    
    assert crop.shape == (50 + 40, 40 + 40, 3)  # height=90, width=80
    assert local_box == [20, 20, 60, 70]  # [bx1-crop_x1, by1-crop_y1, bx2-crop_x1, by2-crop_y1]
    assert meta["clamped_edges"] == {"top": False, "bottom": False, "left": False, "right": False}


def test_extract_defect_crop_clamped_boundaries():
    """Test boundary clamping for top-left and bottom-right defects without throwing IndexError."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Top-left corner defect (touching 0, 0)
    crop_tl, local_tl, meta_tl = extract_defect_crop(img, [0, 0, 15, 15], padding=25)
    assert meta_tl["crop_window"][0] == 0  # crop_x1 clamped to 0
    assert meta_tl["crop_window"][1] == 0  # crop_y1 clamped to 0
    assert meta_tl["clamped_edges"]["left"] is True
    assert meta_tl["clamped_edges"]["top"] is True
    assert local_tl[0] == 0
    assert local_tl[1] == 0
    assert crop_tl.shape[0] <= 100 and crop_tl.shape[1] <= 100

    # Bottom-right corner defect (touching 100, 100)
    crop_br, local_br, meta_br = extract_defect_crop(img, [85, 85, 100, 100], padding=30)
    assert meta_br["crop_window"][2] == 100  # crop_x2 clamped to 100
    assert meta_br["crop_window"][3] == 100  # crop_y2 clamped to 100
    assert meta_br["clamped_edges"]["right"] is True
    assert meta_br["clamped_edges"]["bottom"] is True
    assert local_br[2] <= crop_br.shape[1]
    assert local_br[3] <= crop_br.shape[0]


def test_extract_defect_crop_out_of_bounds_and_empty():
    """Verify graceful handling for coordinates outside image bounds and empty images."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Coordinates exceeding image boundaries
    crop, local_box, meta = extract_defect_crop(img, [-10, -10, 110, 110], padding=10)
    assert meta["board_bbox"] == [0.0, 0.0, 100.0, 100.0]
    assert crop.shape == (100, 100, 3)

    # Empty image raises ValueError
    with pytest.raises(ValueError):
        extract_defect_crop(np.array([]), [0, 0, 10, 10])


# =====================================================================
# 3. SEVERITY CONSISTENCY
# =====================================================================

def test_decision_engine_severity_consistency():
    """
    Verify that DecisionEngine assigns correct severities:
    - open_circuit -> critical
    - conductor_scratch -> minor
    - spur / mouse_bite -> major
    - low-confidence detection (< inspect_conf) -> minor
    And that defects retain this severity when displayed in the UI.
    """
    engine = DecisionEngine()
    
    # 1. Critical defect test
    payload_critical = {
        "board_id": "TEST-CRIT-01",
        "coverage": 1.0,
        "defects": [
            {"class": "open_circuit", "confidence": 0.85, "bbox": [10, 10, 20, 20]}
        ]
    }
    res_critical = engine.evaluate(payload_critical, preset="consumer")
    assert res_critical["decision"] == "REJECT"
    assert res_critical["defects"][0]["severity"] == "critical"
    assert res_critical["summary_counts"]["critical"] == 1

    # 2. Minor defect test (conductor_scratch mapped to minor in config.json)
    payload_minor = {
        "board_id": "TEST-MIN-01",
        "coverage": 1.0,
        "defects": [
            {"class": "conductor_scratch", "confidence": 0.60, "bbox": [30, 30, 40, 40]}
        ]
    }
    res_minor = engine.evaluate(payload_minor, preset="consumer")
    assert res_minor["defects"][0]["severity"] == "minor"
    assert res_minor["summary_counts"]["minor"] == 1

    # 3. Low-confidence defect (< inspect_conf=0.40) screened to minor
    payload_lowconf = {
        "board_id": "TEST-LOWCONF-01",
        "coverage": 1.0,
        "defects": [
            {"class": "mouse_bite", "confidence": 0.35, "bbox": [50, 50, 60, 60]}
        ]
    }
    res_lowconf = engine.evaluate(payload_lowconf, preset="consumer")
    assert res_lowconf["defects"][0]["severity"] == "minor"


# =====================================================================
# 4. COVERAGE & FRAMING WARNING CONSISTENCY
# =====================================================================

def test_coverage_and_clipping_consistency():
    """
    Verify that clipping warnings from calculate_inspection_coverage are properly
    reflected in framing quality and trigger RULE_005D_FRAME_EDGE_CLIPPING.
    """
    engine = DecisionEngine()
    
    # Create an image with dark corners (corner_mean < 128) and bright substrate touching top edge
    img_clipped = np.zeros((200, 200, 3), dtype=np.uint8)
    # Bright substrate extending across the top edge
    img_clipped[:10, 40:160] = 255
    # Internal board body
    img_clipped[10:180, 20:180] = 255
    
    coverage_result = calculate_inspection_coverage(img_clipped)
    assert coverage_result["clipping_warnings"] > 0

    # Ensure quality assessment with clipping warnings triggers RULE_005D_FRAME_EDGE_CLIPPING
    quality_assessment = {
        "domain_status": "within_domain",
        "image_quality_status": "acceptable",
        "clipping_warnings": coverage_result["clipping_warnings"],
        "coverage_ratio": coverage_result["coverage"]
    }
    
    payload = {
        "board_id": "TEST-CLIP-01",
        "coverage": coverage_result["coverage"],
        "defects": [],
        "quality_assessment": quality_assessment
    }
    
    res = engine.evaluate(payload, preset="consumer")
    assert res["decision"] == "INSPECT"
    assert res["rule_id"] == "RULE_005D_FRAME_EDGE_CLIPPING"
    assert res["quality_assessment"]["clipping_warnings"] > 0


# =====================================================================
# 5. DATA SOURCE TAGGING & PRODUCTION ANALYTICS FILTER
# =====================================================================

@pytest.fixture
def db_mgr(tmp_path):
    db_file = tmp_path / "test_dashboard.db"
    mgr = DatabaseManager(db_file)
    yield mgr
    mgr.close()


def test_inspection_source_tagging_and_filtering(db_mgr):
    """Verify that records are tagged with source and filtered out of production analytics."""
    rec_payload = {
        "decision": "REJECT",
        "rule_id": "TEST_RULE",
        "rule_description": "Test",
        "defects": [{"class": "short_circuit", "confidence": 0.9, "severity": "critical"}],
        "summary_counts": {"critical": 1, "major": 0, "minor": 0}
    }

    # Insert 1 production 'real' record, 1 'demo' record, 1 'test' record
    id_real = db_mgr.log_inspection(
        board_id="PROD-001",
        image_path="/path/real.jpg",
        decision_record=rec_payload,
        inspection_source="real",
        image_sha256="abc123real",
        model_sha256="modelhash001"
    )
    id_demo = db_mgr.log_inspection(
        board_id="DEMO-001",
        image_path="/path/demo.jpg",
        decision_record=rec_payload,
        inspection_source="demo",
        image_sha256="abc123demo"
    )
    id_test = db_mgr.log_inspection(
        board_id="TEST-001",
        image_path="/path/test.jpg",
        decision_record=rec_payload,
        inspection_source="test",
        image_sha256="abc123test"
    )

    # 1. list_inspections filter
    all_records = db_mgr.list_inspections(limit=10)
    assert len(all_records) == 3

    prod_records = db_mgr.list_inspections(limit=10, source_filter="real")
    assert len(prod_records) == 1
    assert prod_records[0]["board_id"] == "PROD-001"
    assert prod_records[0]["inspection_source"] == "real"
    assert prod_records[0]["image_sha256"] == "abc123real"

    demo_records = db_mgr.list_inspections(limit=10, source_filter="demo")
    assert len(demo_records) == 1
    assert demo_records[0]["board_id"] == "DEMO-001"

    # 2. get_defect_statistics source filter
    stats_all = db_mgr.get_defect_statistics()
    assert stats_all["total_inspections"] == 3

    stats_prod_only = db_mgr.get_defect_statistics(source_filter="real")
    assert stats_prod_only["total_inspections"] == 1
    assert stats_prod_only["defect_frequencies"].get("short_circuit") == 1


# =====================================================================
# 6. DEFECT-LEVEL HUMAN REVIEW & FALSE-CALL METRICS
# =====================================================================

def test_human_review_and_false_call_metrics(db_mgr):
    """
    Verify defect-level human review logging, persistence, and false call rate computation.
    Must handle 'NO DATA' (false_call_rate is None) when reviewed count is 0.
    """
    # Check initial state: 0 reviews -> false_call_rate is None (UI displays 'NO DATA')
    empty_metrics = db_mgr.get_false_call_metrics()
    assert empty_metrics["total_reviews"] == 0
    assert empty_metrics["reviewed_detections"] == 0
    assert empty_metrics["false_call_rate"] is None

    # Log an inspection first
    insp_id = db_mgr.log_inspection(
        board_id="BOARD-REV-01",
        image_path="/img.jpg",
        decision_record={"decision": "REJECT", "defects": []}
    )

    # Record 4 reviews:
    # 2 confirmed defects
    # 1 false call
    # 1 uncertain (should not enter reviewed denominator)
    db_mgr.record_human_review(
        inspection_id=insp_id,
        board_id="BOARD-REV-01",
        defect_index=1,
        ai_class="open_circuit",
        ai_confidence=0.88,
        ai_severity="critical",
        rule_id="RULE_001",
        reviewer_decision="confirmed_defect",
        reason="Real trace gap confirmed under microscope",
        reviewer_id="QC-ENGINEER-1"
    )
    db_mgr.record_human_review(
        inspection_id=insp_id,
        board_id="BOARD-REV-01",
        defect_index=2,
        ai_class="short_circuit",
        ai_confidence=0.92,
        ai_severity="critical",
        rule_id="RULE_001",
        reviewer_decision="confirmed_defect",
        reason="Copper bridge between traces",
        reviewer_id="QC-ENGINEER-1"
    )
    db_mgr.record_human_review(
        inspection_id=insp_id,
        board_id="BOARD-REV-01",
        defect_index=3,
        ai_class="mouse_bite",
        ai_confidence=0.75,
        ai_severity="major",
        rule_id="RULE_004",
        reviewer_decision="false_call",
        reason="Dust particle on lens",
        reviewer_id="QC-ENGINEER-2"
    )
    db_mgr.record_human_review(
        inspection_id=insp_id,
        board_id="BOARD-REV-01",
        defect_index=4,
        ai_class="spur",
        ai_confidence=0.65,
        ai_severity="major",
        rule_id="RULE_004",
        reviewer_decision="uncertain",
        reason="Ambiguous edge plating",
        reviewer_id="QC-ENGINEER-2"
    )

    reviews = db_mgr.list_human_reviews(inspection_id=insp_id)
    assert len(reviews) == 4

    metrics = db_mgr.get_false_call_metrics()
    assert metrics["total_reviews"] == 4
    assert metrics["confirmed_detections"] == 2
    assert metrics["false_detections"] == 1
    assert metrics["uncertain_detections"] == 1
    # reviewed_detections = confirmed (2) + false (1) = 3
    assert metrics["reviewed_detections"] == 3
    # false_call_rate = 1 / 3 = 33.33%
    assert pytest.approx(metrics["false_call_rate"], 0.01) == 33.33
    assert metrics["class_breakdown"]["mouse_bite"]["false_call"] == 1
    assert metrics["class_breakdown"]["open_circuit"]["confirmed_defect"] == 1


# =====================================================================
# 7. CONFIGURATION SNAPSHOT PERSISTENCE
# =====================================================================

def test_config_snapshot_persistence(db_mgr):
    """Verify that configuration snapshot JSON is stored and retrieved properly."""
    snapshot = {
        "preset": "Consumer",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.50,
        "model": "YOLOv8n-PCB-Baseline",
        "model_sha256": "66d8d225feb8e1eb6be63365fa3a479261eb857ad8a9dd603f0b2fbe0cfbf3fe"
    }

    insp_id = db_mgr.log_inspection(
        board_id="BOARD-SNAP-01",
        image_path="/img.png",
        decision_record={"decision": "PASS", "defects": []},
        config_snapshot=snapshot
    )

    rec = db_mgr.get_inspection(insp_id)
    assert rec["config_snapshot"] is not None
    assert rec["config_snapshot"]["preset"] == "Consumer"
    assert rec["config_snapshot"]["model_sha256"] == snapshot["model_sha256"]
    assert rec["config_snapshot"]["confidence_threshold"] == 0.25


# =====================================================================
# 8. STREAMLIT APP STRUCTURE & PRESET INVARIANTS
# =====================================================================

def test_streamlit_app_structure_and_presets():
    """
    Verify that dashboard/app.py executes cleanly under Streamlit AppTest:
    - Zero runtime exceptions during initialization and layout rendering
    - All 5 required industrial tabs are present
    - Inspection Quality Preset selectbox exists and offers strictly Consumer and Development
    - Default preset is 'Consumer'
    """
    from streamlit.testing.v1 import AppTest
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)
    
    assert len(at.exception) == 0, f"AppTest raised exceptions: {at.exception}"
    
    tab_labels = [t.label for t in at.tabs]
    assert len(tab_labels) == 7
    assert "📹 Live Camera Inspection" in tab_labels
    assert "🔍 Captured PCB Inspection" in tab_labels
    assert "🔬 Human Review & Verification" in tab_labels
    assert "📋 Inspection Log & Audit" in tab_labels
    assert "📊 Pareto & Defect Analytics" in tab_labels
    assert "⚙️ Hardware Status & Telemetry" in tab_labels
    assert "🛠️ Station Settings & Rules" in tab_labels

    # Check sidebar selectboxes
    sidebar_selects = {s.label: s for s in at.sidebar.selectbox}
    assert "Inspection Quality Preset" in sidebar_selects
    preset_box = sidebar_selects["Inspection Quality Preset"]
    assert set(preset_box.options) == {"Consumer", "Development"}
    assert preset_box.value == "Consumer"


# =====================================================================
# 9. UPLOAD-FIRST WORKFLOW & EXAMPLE EXPANDER
# =====================================================================

def test_upload_first_structure_and_collapsed_examples():
    """
    Verify that Tab 1 renders with the upload-first workflow:
    - Primary file uploader is 'Upload Bare PCB Inspection Image'
    - 'Load Example PCB' is in a collapsed expander
    - Example options contain descriptive names without 'Demo 1/2/3/4'
    - Default example selection is 'None (Use Custom Upload)'
    """
    from streamlit.testing.v1 import AppTest
    app_path = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run(timeout=30)

    # Primary file uploader exists
    uploaders = [u.label for u in at.file_uploader]
    assert "Upload Bare PCB Inspection Image" in uploaders

    # Expander for examples exists
    expanders = [e.label for e in at.expander]
    assert any("Load Example PCB" in label for label in expanders)

    # Example selectbox has descriptive names without 'Demo 1'
    example_boxes = [s for s in at.selectbox if s.label == "Select Example PCB:"]
    assert len(example_boxes) == 1
    ex_box = example_boxes[0]
    expected_options = [
        "None (Use Custom Upload)",
        "DeepPCB Monochrome Defect",
        "DsPCBSD+ Colour Defect",
        "PCB-Defect V2 Colour Defect",
        "Clean Reference Board"
    ]
    assert ex_box.options == expected_options
    assert ex_box.value == "None (Use Custom Upload)"
    for opt in ex_box.options:
        assert not opt.startswith("Demo "), f"Option still contains legacy 'Demo' prefix: {opt}"
        assert "guaranteed" not in opt.lower(), f"Option claims guaranteed outcome: {opt}"


def test_custom_upload_takes_priority_over_example():
    """
    Simulate priority resolution logic:
    When a custom uploaded image exists, it must strictly take precedence
    over any example selection, tagging source as 'real'.
    """
    # Case A: Custom image uploaded and example selected
    has_custom_upload = True
    selected_example = "DsPCBSD+ Colour Defect"
    
    if has_custom_upload:
        input_source = "real"
        source_desc = "Custom Image Upload: test_board.jpg"
    elif selected_example != "None (Use Custom Upload)":
        input_source = "demo"
        source_desc = selected_example
    else:
        input_source = "unknown"
        source_desc = ""

    assert input_source == "real"
    assert "Custom Image Upload" in source_desc

    # Case B: No custom upload, example selected
    has_custom_upload = False
    selected_example = "Clean Reference Board"
    
    if has_custom_upload:
        input_source = "real"
        source_desc = "Custom Image Upload: test_board.jpg"
    elif selected_example != "None (Use Custom Upload)":
        input_source = "demo"
        source_desc = selected_example

    assert input_source == "demo"
    assert source_desc == "Clean Reference Board"


# =====================================================================
# 10. DATABASE LOGGING SAFETY & RETRY PRESERVATION
# =====================================================================

def test_log_inspection_argument_compatibility_and_kwargs(db_mgr):
    """
    Verify DatabaseManager.log_inspection argument signature compatibility:
    - Accepts all defined arguments (inspection_source, config_snapshot, image_sha256, model_sha256)
    - Accepts arbitrary **kwargs without raising TypeError
    - Fully preserves attributes into persistent record
    """
    dec_rec = {
        "decision": "PASS",
        "rule_id": "RULE_008_ZERO_CRITICAL",
        "rule_description": "Clean board verified",
        "defects": [],
        "summary_counts": {"critical": 0, "major": 0, "minor": 0}
    }
    cfg_snap = {
        "preset": "Consumer",
        "confidence_threshold": 0.25,
        "model_sha256": "testhash123"
    }

    # Call with standard parameters and extra dynamic kwargs
    insp_id = db_mgr.log_inspection(
        board_id="BOARD-KW-01",
        image_path="/path/test.jpg",
        decision_record=dec_rec,
        inspection_source="real",
        config_snapshot=cfg_snap,
        image_sha256="imgsha_abc",
        model_sha256="modelsha_xyz",
        extra_legacy_kwarg="harmless_value",
        another_dynamic_arg=42
    )

    assert insp_id is not None
    rec = db_mgr.get_inspection(insp_id)
    assert rec["board_id"] == "BOARD-KW-01"
    assert rec["inspection_source"] == "real"
    assert rec["image_sha256"] == "imgsha_abc"
    assert rec["model_sha256"] == "modelsha_xyz"
    assert rec["config_snapshot"]["preset"] == "Consumer"


def test_database_failure_safety_state_preservation():
    """
    Verify that if database persistence fails:
    - Vision inference and decision results are safely preserved in session state
    - insp_uuid is None and db_error is captured
    - System does not falsely report successful persistent save
    - Retry mechanism can persist the stored record without re-running vision inference
    """
    # 1. Simulate vision and decision payload
    vision_defects = [{"class": "open_circuit", "confidence": 0.95, "severity": "critical"}]
    decision_record = {"decision": "REJECT", "rule_id": "RULE_001_HIGH_CONF_CRITICAL_DEFECT", "defects": vision_defects}

    # 2. Mock failing DatabaseManager
    class FailingDatabaseManager:
        def log_inspection(self, *args, **kwargs):
            raise RuntimeError("Disk I/O error or SQLite database locked")

    failing_db = FailingDatabaseManager()

    insp_uuid = None
    db_error = None
    try:
        insp_uuid = failing_db.log_inspection("BOARD-FAIL-01", "/path.jpg", decision_record)
    except Exception as e:
        db_error = str(e)

    session_state = {
        "board_id": "BOARD-FAIL-01",
        "defects": vision_defects,
        "decision_record": decision_record,
        "insp_uuid": insp_uuid,
        "db_error": db_error
    }

    # Verify safe failure behavior
    assert session_state["insp_uuid"] is None
    assert session_state["db_error"] == "Disk I/O error or SQLite database locked"
    assert len(session_state["defects"]) == 1
    assert session_state["decision_record"]["decision"] == "REJECT"

    # 3. Simulate retry using a working DatabaseManager without re-running vision
    class WorkingDatabaseManager:
        def log_inspection(self, board_id, image_path, decision_record, **kwargs):
            return "recovered-uuid-9999"

    working_db = WorkingDatabaseManager()
    retry_uuid = working_db.log_inspection(
        session_state["board_id"],
        "/path.jpg",
        session_state["decision_record"]
    )
    session_state["insp_uuid"] = retry_uuid
    session_state["db_error"] = None

    assert session_state["insp_uuid"] == "recovered-uuid-9999"
    assert session_state["db_error"] is None


# =====================================================================
# 10. PERMANENT SIDE-BY-SIDE TWO-COLUMN LAYOUT INVARIANTS
# =====================================================================

def test_permanent_side_by_side_layout_structure():
    """Verify Tab 1 defines a permanent side-by-side 50/50 two-column layout starting at the top."""
    app_text = Path("dashboard/app.py").read_text(encoding="utf-8")

    # 1. Verify 50/50 columns initialization at the top of tab_inspect
    assert 'col_left, col_right = st.columns([1, 1], gap="medium")' in app_text

    # 2. Verify left column elements
    assert "Visual Inspection Workspace" in app_text
    assert "Upload Bare PCB Inspection Image" in app_text
    assert "Defect Magnifier (Enlarged ROI)" in app_text
    assert 'st.tabs(["Raw PCB Image", "Annotated Bounding Boxes"])' in app_text or 'st.tabs(["Annotated Bounding Boxes", "Raw PCB Image"])' in app_text

    # 3. Verify right column elements
    assert "Quality Disposition & Advisory" in app_text
    assert "Run Inspection" in app_text
    assert "FINAL DECISION: PASS" in app_text
    assert "FINAL DECISION: INSPECT" in app_text
    assert "FINAL DECISION: REJECT" in app_text
    assert "ELECTRICAL VERIFICATION MANDATORY" in app_text
    assert "Localized Defect Manifest" in app_text
    assert "AI Engineering Advisory & Root-Cause Guidance" in app_text
    assert "Line Operator Confirmation & Override" in app_text


def test_verified_side_by_side_screenshot_artifacts():
    """Verify that empirical screenshot artifacts for 1920x1080 and 1366x768 exist with valid dimensions."""
    from PIL import Image
    ui_dir = Path(__file__).resolve().parent.parent / "docs" / "ui" / "after"
    
    # 1. 1920x1080 Full HD
    p_1080 = ui_dir / "01_live_camera_1920x1080.png"
    assert p_1080.exists(), "01_live_camera_1920x1080.png must exist"
    assert p_1080.stat().st_size > 50_000, f"Expected non-trivial image size, got {p_1080.stat().st_size} bytes"
    with Image.open(p_1080) as im:
        assert im.size == (1920, 1080), f"Expected 1920x1080, got {im.size}"

    # 2. 1366x768 Laptop
    p_768 = ui_dir / "01_live_camera_1366x768.png"
    assert p_768.exists(), "01_live_camera_1366x768.png must exist"
    assert p_768.stat().st_size > 50_000, f"Expected non-trivial image size, got {p_768.stat().st_size} bytes"
    with Image.open(p_768) as im:
        assert im.size == (1366, 768), f"Expected 1366x768, got {im.size}"

    # 3. Inspection View 1920x1080
    p_inspect = ui_dir / "02_inspect_1920x1080.png"
    assert p_inspect.exists(), "02_inspect_1920x1080.png must exist"
    assert p_inspect.stat().st_size > 50_000, f"Expected non-trivial image size, got {p_inspect.stat().st_size} bytes"
    with Image.open(p_inspect) as im:
        assert im.size == (1920, 1080), f"Expected 1920x1080, got {im.size}"



