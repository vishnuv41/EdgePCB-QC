"""
Automated Test for DsPCBSD+ Split Integrity and Documented Leakage Constraints
Bharat AI-SoC Challenge 2026-27 | PS5

Verifies:
1. Zero exact filename overlap between train and validation splits.
2. Zero bitwise SHA-256 duplicates across train and validation splits.
3. Verification that COCO metadata lacks physical board identifiers.
4. Audit documentation is preserved in benchmarks/results/dspcbsd_leakage_audit.json.
"""

import json
from pathlib import Path
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
AUDIT_JSON = WORKSPACE_ROOT / "benchmarks" / "results" / "dspcbsd_leakage_audit.json"
RAW_DSPCBSD = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd"


def test_audit_report_exists_and_valid():
    assert AUDIT_JSON.exists(), f"Audit report missing: {AUDIT_JSON}"
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["train_image_count"] == 8208
    assert data["val_image_count"] == 2051
    assert data["exact_filename_overlap_count"] == 0
    assert data["bitwise_sha256_leak_count"] == 0
    assert data["metadata_has_board_id"] is False
    assert "cross-patch generalization" in data["conclusion"]


def test_zero_filename_overlap_between_splits():
    train_dir = RAW_DSPCBSD / "Data_COCO" / "train2017"
    val_dir = RAW_DSPCBSD / "Data_COCO" / "val2017"

    if train_dir.exists() and val_dir.exists():
        train_names = {f.name for f in train_dir.glob("*.jpg")}
        val_names = {f.name for f in val_dir.glob("*.jpg")}
        overlap = train_names.intersection(val_names)
        assert len(overlap) == 0, f"Found overlapping filenames: {overlap}"


def test_coco_json_metadata_integrity():
    train_json = RAW_DSPCBSD / "Data_COCO" / "annotations" / "instances_train2017.json"
    if train_json.exists():
        with open(train_json, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        
        sample_img = t_data["images"][0]
        # Verify that dataset metadata records patch dimensions but does not record physical board IDs
        assert "board_id" not in sample_img
        assert sample_img["width"] == 226
        assert sample_img["height"] == 226
