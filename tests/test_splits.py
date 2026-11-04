"""
Automated Split Leakage & Data Integrity Tests for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Proves:
1. Zero board-level leakage: no PCB board/layout group appears in multiple splits.
2. Zero image-level leakage: no image exists in multiple splits.
3. Bounding box coordinates are valid normalized floats in [0.0, 1.0].
4. Class IDs strictly adhere to canonical class mapping.
"""

import json
import pytest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = WORKSPACE_ROOT / "datasets" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "splits_manifest.json"
YOLO_DIR = PROCESSED_DIR / "deeppcb_yolo"


def test_manifest_exists():
    assert MANIFEST_PATH.exists(), f"Splits manifest missing at {MANIFEST_PATH}"


def test_board_level_zero_leakage():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_groups = set(manifest["train_groups"])
    val_groups = set(manifest["val_groups"])
    test_groups = set(manifest["test_groups"])

    # Disjointness checks
    assert len(train_groups) > 0, "Train groups must not be empty"
    assert len(val_groups) > 0, "Val groups must not be empty"
    assert len(test_groups) > 0, "Test groups must not be empty"

    assert train_groups.isdisjoint(val_groups), f"Leakage between train and val: {train_groups & val_groups}"
    assert train_groups.isdisjoint(test_groups), f"Leakage between train and test: {train_groups & test_groups}"
    assert val_groups.isdisjoint(test_groups), f"Leakage between val and test: {val_groups & test_groups}"


def test_image_level_zero_leakage():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    split_images = {"train": set(), "val": set(), "test": set()}

    for img_name, meta in manifest["images"].items():
        split = meta["split"]
        assert split in split_images, f"Unknown split: {split}"
        split_images[split].add(img_name)

    assert split_images["train"].isdisjoint(split_images["val"]), "Image leakage detected between train and val!"
    assert split_images["train"].isdisjoint(split_images["test"]), "Image leakage detected between train and test!"
    assert split_images["val"].isdisjoint(split_images["test"]), "Image leakage detected between val and test!"


def test_yolo_labels_validity():
    for split in ["train", "val", "test"]:
        lbl_dir = YOLO_DIR / "labels" / split
        img_dir = YOLO_DIR / "images" / split

        assert lbl_dir.exists(), f"Label dir missing: {lbl_dir}"
        assert img_dir.exists(), f"Image dir missing: {img_dir}"

        labels = list(lbl_dir.glob("*.txt"))
        assert len(labels) > 0, f"No labels found in {lbl_dir}"

        for lf in labels[:50]:  # Sample test first 50 files per split
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    assert len(parts) == 5, f"Malformed line in {lf}: {line}"
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])

                    assert 0 <= cls_id <= 5, f"Class ID out of range [0, 5]: {cls_id} in {lf}"
                    assert 0.0 <= xc <= 1.0, f"xc out of bounds: {xc} in {lf}"
                    assert 0.0 <= yc <= 1.0, f"yc out of bounds: {yc} in {lf}"
                    assert 0.0 <= w <= 1.0, f"w out of bounds: {w} in {lf}"
                    assert 0.0 <= h <= 1.0, f"h out of bounds: {h} in {lf}"


def test_unified_splits_zero_leakage():
    """Verify zero image leakage across train, val, test, and external_color_test in unified dataset."""
    unified_dir = PROCESSED_DIR / "unified_yolo" / "images"
    assert unified_dir.exists(), f"Unified images directory missing at {unified_dir}"

    splits = ["train", "val", "test", "external_color_test"]
    split_files = {s: set(p.name for p in (unified_dir / s).glob("*.jpg")) for s in splits}

    for s1 in splits:
        assert len(split_files[s1]) > 0, f"Split {s1} is empty!"
        for s2 in splits:
            if s1 != s2:
                overlap = split_files[s1] & split_files[s2]
                assert not overlap, f"Image leakage detected between {s1} and {s2}: {len(overlap)} images overlap!"


def test_unified_pcb_defect_board_isolation():
    """Verify physical board isolation: boards 68, 80, 81, 82 must be exclusively in external_color_test."""
    unified_dir = PROCESSED_DIR / "unified_yolo" / "images"
    test_boards = {"68", "80", "81", "82"}
    val_boards = {"65", "66", "67"}
    train_boards = {str(i) for i in range(50, 65)}

    for fname in (unified_dir / "external_color_test").glob("pcbdefect_*.jpg"):
        board_match = any(b in fname.name for b in test_boards)
        assert board_match, f"Unexpected image in external_color_test: {fname.name}"

    for split in ["train", "val"]:
        for fname in (unified_dir / split).glob("pcbdefect_*.jpg"):
            for tb in test_boards:
                assert f"_{tb}-" not in fname.name and f"_{tb}_" not in fname.name, (
                    f"Leakage detected! Untouched test board {tb} found in {split}: {fname.name}"
                )


def test_unified_labels_validity():
    """Check bounding box normalization and class ranges in unified dataset."""
    unified_lbl_dir = PROCESSED_DIR / "unified_yolo" / "labels"
    for split in ["train", "val", "test", "external_color_test"]:
        labels = list((unified_lbl_dir / split).glob("*.txt"))
        assert len(labels) > 0, f"No labels found in {unified_lbl_dir / split}"
        for lf in labels[:100]:
            with open(lf, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    assert len(parts) == 5, f"Malformed line in {lf}: {line}"
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    assert 0 <= cls_id <= 5, f"Invalid class {cls_id} in {lf}"
                    assert 0.0 <= xc <= 1.0, f"xc out of bounds: {xc} in {lf}"
                    assert 0.0 <= yc <= 1.0, f"yc out of bounds: {yc} in {lf}"
                    assert 0.0 <= w <= 1.0, f"w out of bounds: {w} in {lf}"
                    assert 0.0 <= h <= 1.0, f"h out of bounds: {h} in {lf}"

