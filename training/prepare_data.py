"""
Dataset Preparation & Leakage-Free Splitting for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Rules enforced:
1. Board-level splitting for DeepPCB: whole board groups are assigned exclusively
   to Train, Val, or Test. Never split images within the same board group across splits!
2. Split verification manifest generated at datasets/processed/splits_manifest.json.
3. Normalized YOLO format created for detector training with canonical class IDs.
"""

import os
import sys
import json
import yaml
import shutil
from pathlib import Path
from collections import defaultdict, Counter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = WORKSPACE_ROOT / "datasets" / "raw"
PROCESSED_DIR = WORKSPACE_ROOT / "datasets" / "processed"
CLASS_MAP_PATH = WORKSPACE_ROOT / "training" / "class_map.yaml"

DEEPPCB_RAW = RAW_DIR / "deeppcb" / "PCBData"
DSPCBSD_RAW = RAW_DIR / "dspcbsd"


def load_class_map():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def prepare_deeppcb_splits(class_map):
    print("\n" + "=" * 70)
    print("PREPARING DEEPPCB WITH BOARD-LEVEL LEAKAGE-FREE SPLITTING")
    print("=" * 70)

    if not DEEPPCB_RAW.exists():
        print(f"[!] DeepPCB raw directory not found at: {DEEPPCB_RAW}")
        return None

    # Class translation from DeepPCB ID (1..6) to Canonical Class ID (0..5)
    deeppcb_to_canonical = {
        1: 0,  # open -> open_circuit
        2: 1,  # short -> short_circuit
        3: 2,  # mousebite -> mouse_bite
        4: 3,  # spur -> spur
        5: 4,  # copper -> spurious_copper
        6: 5,  # pin-hole -> pin_hole_missing_pad
    }

    groups = sorted([d.name for d in DEEPPCB_RAW.iterdir() if d.is_dir()])
    print(f"[*] Found {len(groups)} distinct PCB board groups: {groups}")

    # Deterministic partition of groups:
    # 11 groups: 7 Train, 2 Val, 2 Test (completely unseen boards)
    train_groups = set(groups[:7])
    val_groups = set(groups[7:9])
    test_groups = set(groups[9:])

    print(f"[*] Train groups ({len(train_groups)}): {sorted(list(train_groups))}")
    print(f"[*] Val groups   ({len(val_groups)}): {sorted(list(val_groups))}")
    print(f"[*] Test groups  ({len(test_groups)}): {sorted(list(test_groups))}")

    # Prepare output directories
    target_dir = PROCESSED_DIR / "deeppcb_yolo"
    for split in ["train", "val", "test"]:
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "DeepPCB",
        "split_policy": "board_group_level_zero_leakage",
        "train_groups": list(train_groups),
        "val_groups": list(val_groups),
        "test_groups": list(test_groups),
        "images": {}
    }

    stats = {
        "train": {"images": 0, "boxes": 0, "classes": Counter()},
        "val": {"images": 0, "boxes": 0, "classes": Counter()},
        "test": {"images": 0, "boxes": 0, "classes": Counter()}
    }

    for group_name in groups:
        if group_name in train_groups:
            split = "train"
        elif group_name in val_groups:
            split = "val"
        else:
            split = "test"

        group_path = DEEPPCB_RAW / group_name
        img_dirs = [d for d in group_path.iterdir() if d.is_dir() and not d.name.endswith("_not")]
        anno_dirs = [d for d in group_path.iterdir() if d.is_dir() and d.name.endswith("_not")]

        if not img_dirs or not anno_dirs:
            continue

        img_dir = img_dirs[0]
        anno_dir = anno_dirs[0]

        for test_img_path in sorted(img_dir.glob("*_test.jpg")):
            base_id = test_img_path.name.replace("_test.jpg", "")
            temp_img_path = img_dir / f"{base_id}_temp.jpg"
            anno_path = anno_dir / f"{base_id}.txt"

            target_img_name = f"{group_name}_{base_id}.jpg"
            target_img_dest = target_dir / "images" / split / target_img_name
            target_lbl_dest = target_dir / "labels" / split / f"{group_name}_{base_id}.txt"

            # Copy or hardlink image
            if not target_img_dest.exists():
                shutil.copy2(test_img_path, target_img_dest)

            # Convert annotations: x1 y1 x2 y2 type -> class_id xc yc w h (normalized to 640x640)
            img_w, img_h = 640.0, 640.0
            yolo_lines = []

            if anno_path.exists():
                with open(anno_path, "r", encoding="utf-8") as af:
                    for line in af:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            x1, y1, x2, y2, raw_type = map(int, parts[:5])
                            canon_id = deeppcb_to_canonical.get(raw_type)
                            if canon_id is None:
                                continue

                            # Convert to YOLO format
                            w = (x2 - x1) / img_w
                            h = (y2 - y1) / img_h
                            xc = (x1 + x2) / (2.0 * img_w)
                            yc = (y1 + y2) / (2.0 * img_h)

                            # Clamp within [0, 1]
                            xc = max(0.0, min(1.0, xc))
                            yc = max(0.0, min(1.0, yc))
                            w = max(0.0, min(1.0, w))
                            h = max(0.0, min(1.0, h))

                            yolo_lines.append(f"{canon_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            stats[split]["classes"][canon_id] += 1
                            stats[split]["boxes"] += 1

            with open(target_lbl_dest, "w", encoding="utf-8") as lf:
                lf.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            stats[split]["images"] += 1
            manifest["images"][target_img_name] = {
                "group": group_name,
                "split": split,
                "template_path": str(temp_img_path.relative_to(WORKSPACE_ROOT)),
                "num_defects": len(yolo_lines)
            }

    # Write manifest
    manifest_path = PROCESSED_DIR / "splits_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[+] Splits manifest saved to: {manifest_path}")

    # Write YOLO data.yaml
    names = {
        0: "open_circuit",
        1: "short_circuit",
        2: "mouse_bite",
        3: "spur",
        4: "spurious_copper",
        5: "pin_hole_missing_pad"
    }
    yolo_yaml = {
        "path": str(target_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names
    }

    yaml_path = target_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(yolo_yaml, yf, default_flow_style=False)
    print(f"[+] YOLO data configuration written to: {yaml_path}")

    print("\n[*] Split Summary:")
    for split in ["train", "val", "test"]:
        print(f"    - {split.upper():5s}: {stats[split]['images']:5d} images, {stats[split]['boxes']:5d} bboxes")

    return target_dir


if __name__ == "__main__":
    c_map = load_class_map()
    prepare_deeppcb_splits(c_map)
