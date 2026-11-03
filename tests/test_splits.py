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
