"""
Unit & Regression Tests for CPUInspector Dynamic Checkpoint Switching
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Loading the baseline checkpoint.
2. Loading the unified multi-domain checkpoint.
3. Switching baseline -> multi-domain.
4. Switching multi-domain -> baseline.
5. Missing checkpoint file raises FileNotFoundError.
6. Invalid / corrupted checkpoint raises RuntimeError and sets is_loaded=False.
7. Correct class names and model version after switching.
8. Correct supported-domain metadata after switching.
9. Failed switching does not silently use stale model weights.
"""

import pytest
from pathlib import Path
from inspection.cpu_inference import CPUInspector

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
MULTIDOMAIN_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"


def test_1_loading_baseline_checkpoint():
    """Verify loading the DeepPCB baseline checkpoint."""
    inspector = CPUInspector(model_path=BASELINE_PATH, device="cpu")
    assert inspector.is_loaded is True
    meta = inspector.get_model_metadata()
    assert meta["model_name"] == "DeepPCB Baseline YOLOv8n"
    assert "smoke" in meta["model_version"].lower()
    assert meta["supported_domains"] == ["monochrome_high_contrast"]
    assert meta["is_loaded"] is True


def test_2_loading_multidomain_checkpoint():
    """Verify loading the unified multi-domain checkpoint."""
    inspector = CPUInspector(model_path=MULTIDOMAIN_PATH, device="cpu")
    assert inspector.is_loaded is True
    meta = inspector.get_model_metadata()
    assert meta["model_name"] == "Unified Multi-Domain YOLOv8n"
    assert "multidomain" in meta["model_version"].lower()
    assert meta["supported_domains"] == ["monochrome_high_contrast", "color_pcb"]
    assert meta["is_loaded"] is True


def test_3_switching_baseline_to_multidomain():
    """Verify switching from baseline -> multi-domain updates weights and metadata."""
    inspector = CPUInspector(model_path=BASELINE_PATH, device="cpu")
    assert inspector.get_model_metadata()["model_name"] == "DeepPCB Baseline YOLOv8n"

    success = inspector.load_checkpoint(MULTIDOMAIN_PATH)
    assert success is True
    assert inspector.is_loaded is True

    meta = inspector.get_model_metadata()
    assert meta["model_name"] == "Unified Multi-Domain YOLOv8n"
    assert "pcb_multidomain_experiment" in meta["model_path"]
    assert meta["supported_domains"] == ["monochrome_high_contrast", "color_pcb"]


def test_4_switching_multidomain_to_baseline():
    """Verify switching from multi-domain -> baseline updates weights and metadata."""
    inspector = CPUInspector(model_path=MULTIDOMAIN_PATH, device="cpu")
    assert inspector.get_model_metadata()["model_name"] == "Unified Multi-Domain YOLOv8n"

    success = inspector.load_checkpoint(BASELINE_PATH)
    assert success is True
    assert inspector.is_loaded is True

    meta = inspector.get_model_metadata()
    assert meta["model_name"] == "DeepPCB Baseline YOLOv8n"
    assert "pcb_smoke_experiment" in meta["model_path"]
    assert meta["supported_domains"] == ["monochrome_high_contrast"]


def test_5_missing_checkpoint_file():
    """Verify that attempting to load a non-existent checkpoint raises FileNotFoundError and unloads."""
    inspector = CPUInspector(model_path=MULTIDOMAIN_PATH, device="cpu")
    assert inspector.is_loaded is True

    missing_path = WORKSPACE_ROOT / "models" / "checkpoints" / "non_existent_weights.pt"
    with pytest.raises(FileNotFoundError):
        inspector.load_checkpoint(missing_path)

    # Must be unloaded and have no stale model
    assert inspector.is_loaded is False
    assert inspector.yolo_model is None
    meta = inspector.get_model_metadata()
    assert meta["is_loaded"] is False
    assert meta["model_name"] == "Unloaded"


def test_6_invalid_checkpoint_file(tmp_path):
    """Verify that attempting to load a corrupt/invalid file raises RuntimeError and resets state."""
    inspector = CPUInspector(model_path=MULTIDOMAIN_PATH, device="cpu")
    assert inspector.is_loaded is True

    # Create a corrupted dummy file
    corrupt_file = tmp_path / "corrupt_model.pt"
    corrupt_file.write_text("CORRUPTED_NOT_A_VALID_WEIGHT_FILE", encoding="utf-8")

    with pytest.raises(RuntimeError):
        inspector.load_checkpoint(corrupt_file)

    assert inspector.is_loaded is False
    assert inspector.yolo_model is None
    meta = inspector.get_model_metadata()
    assert meta["is_loaded"] is False


def test_7_correct_class_names_and_version_after_switching():
    """Verify class names and architecture are properly set after switching."""
    inspector = CPUInspector(model_path=BASELINE_PATH, device="cpu")
    meta_base = inspector.get_model_metadata()
    assert meta_base["num_classes"] == 6
    assert "open_circuit" in meta_base["classes"]
    assert "short_circuit" in meta_base["classes"]

    inspector.load_checkpoint(MULTIDOMAIN_PATH)
    meta_multi = inspector.get_model_metadata()
    assert meta_multi["num_classes"] == 6
    assert meta_multi["classes"] == meta_base["classes"]
    assert "multidomain" in meta_multi["model_version"].lower()


def test_8_correct_supported_domain_metadata_after_switching():
    """Verify supported domains change strictly according to active checkpoint."""
    inspector = CPUInspector(model_path=MULTIDOMAIN_PATH, device="cpu")
    assert "color_pcb" in inspector.get_model_metadata()["supported_domains"]

    inspector.load_checkpoint(BASELINE_PATH)
    assert inspector.get_model_metadata()["supported_domains"] == ["monochrome_high_contrast"]
    assert "color_pcb" not in inspector.get_model_metadata()["supported_domains"]


def test_9_failed_switching_does_not_silently_use_stale_weights(tmp_path):
    """Verify that when a checkpoint switch fails, subsequent inspections fail fast rather than using old weights."""
    inspector = CPUInspector(model_path=MULTIDOMAIN_PATH, device="cpu")
    assert inspector.is_loaded is True

    # Attempt to load a missing file
    bogus_path = tmp_path / "ghost_checkpoint.pt"
    with pytest.raises(FileNotFoundError):
        inspector.load_checkpoint(bogus_path)

    # Crucial: Must NOT retain the multidomain model in the background
    assert inspector.yolo_model is None
    assert inspector.is_loaded is False

    # Attempting to inspect with unloaded model MUST raise RuntimeError
    import numpy as np
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError) as exc_info:
        inspector.inspect(dummy_img)

    assert "Cannot run inspection: No valid model checkpoint is loaded" in str(exc_info.value)
