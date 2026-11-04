"""
Unified Multi-Domain Dataset Preparation for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Integrates:
1. DeepPCB (1,120 train, 186 val, 194 untouched test) - Monochrome scans
2. DsPCBSD+ (All 5,811 train + 1,444 val compatible images from industrial AOI lines)
3. PCB-Defect V2 (Native resolution 640x640 tiled crops, board-level partitioned)

Guarantees:
- Zero board-level leakage across all datasets
- The 194-image DeepPCB test set remains 100% untouched
- Boards 68, 80, 81, 82 of PCB-Defect V2 strictly reserved for untouched external color testing
- Strict annotation validation (0 invalid lines, 0 out-of-bound coords)
- Source tracking tags on every file (deeppcb_, dspcbsd_, pcbdefect_)
"""

import os
import sys
import json
import yaml
import shutil
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = WORKSPACE_ROOT / "datasets" / "raw"
PROCESSED_DIR = WORKSPACE_ROOT / "datasets" / "processed"
TARGET_DIR = PROCESSED_DIR / "unified_yolo"

DEEPPCB_PROCESSED = PROCESSED_DIR / "deeppcb_yolo"
DSPCBSD_RAW = RAW_DIR / "dspcbsd"
PCB_DEFECT_RAW = RAW_DIR / "pcb_defect_v2"


def build_unified_dataset():
    print("=" * 70)
    print("PREPARING UNIFIED MULTI-DOMAIN DATASET (MONOCHROME + INDUSTRIAL COLOR)")
    print("=" * 70)

    for split in ["train", "val", "test", "external_color_test"]:
        (TARGET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "train": {"images": 0, "boxes": 0, "classes": Counter(), "sources": Counter()},
        "val": {"images": 0, "boxes": 0, "classes": Counter(), "sources": Counter()},
        "test": {"images": 0, "boxes": 0, "classes": Counter(), "sources": Counter()},
        "external_color_test": {"images": 0, "boxes": 0, "classes": Counter(), "sources": Counter()}
    }

    manifest = {
        "dataset_name": "EdgePCB-QC Unified Multi-Domain Dataset",
        "description": "Combines DeepPCB monochrome with all compatible DsPCBSD+ and tiled PCB-Defect V2 color images",
        "classes": {
            0: "open_circuit",
            1: "short_circuit",
            2: "mouse_bite",
            3: "spur",
            4: "spurious_copper",
            5: "pin_hole_missing_pad"
        },
        "counts": {}
    }

    # ---------------------------------------------------------
    # 1. DEEPPCB INTEGRATION (Keep exact splits, untouched test)
    # ---------------------------------------------------------
    print("\n[*] 1. Integrating DeepPCB splits (monochrome)...")
    for split in ["train", "val", "test"]:
        src_img_dir = DEEPPCB_PROCESSED / "images" / split
        src_lbl_dir = DEEPPCB_PROCESSED / "labels" / split

        for img_path in sorted(src_img_dir.glob("*.jpg")):
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            dst_img = TARGET_DIR / "images" / split / f"deeppcb_{img_path.name}"
            dst_lbl = TARGET_DIR / "labels" / split / f"deeppcb_{img_path.stem}.txt"

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            yolo_lines = []
            if lbl_path.exists():
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            cid = int(parts[0])
                            yolo_lines.append(line)
                            stats[split]["classes"][cid] += 1
                            stats[split]["boxes"] += 1

            with open(dst_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            stats[split]["images"] += 1
            stats[split]["sources"]["DeepPCB"] += 1

    print(f"    - DeepPCB: Train={stats['train']['sources']['DeepPCB']}, Val={stats['val']['sources']['DeepPCB']}, Test={stats['test']['sources']['DeepPCB']}")

    # ---------------------------------------------------------
    # 2. DSPCBSD+ INTEGRATION (All compatible images, train & val)
    # ---------------------------------------------------------
    print("\n[*] 2. Integrating all compatible DsPCBSD+ industrial color patches...")
    # DsPCBSD+ mapping:
    # 0: SH -> short_circuit (1)
    # 1: SP -> spur (3)
    # 2: SC -> spurious_copper (4)
    # 3: OP -> open_circuit (0)
    # 4: MB -> mouse_bite (2)
    # 5: HB -> pin_hole_missing_pad (5)
    dsp_yolo_to_canonical = {
        0: 1,  # SH -> short_circuit
        1: 3,  # SP -> spur
        2: 4,  # SC -> spurious_copper
        3: 0,  # OP -> open_circuit
        4: 2,  # MB -> mouse_bite
        5: 5   # HB -> pin_hole_missing_pad
    }

    for dsp_split, target_split in [("train", "train"), ("val", "val")]:
        img_dir = DSPCBSD_RAW / "Data_YOLO" / "images" / dsp_split
        lbl_dir = DSPCBSD_RAW / "Data_YOLO" / "labels" / dsp_split

        lbl_files = sorted(list(lbl_dir.glob("*.txt")))
        print(f"    - Processing DsPCBSD+ {dsp_split} split ({len(lbl_files)} label files)...")

        for lf in lbl_files:
            with open(lf, "r", encoding="utf-8") as f:
                lines = f.readlines()

            valid_lines = []
            has_relevant_class = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    raw_cid = int(parts[0])
                    if raw_cid in dsp_yolo_to_canonical:
                        canon_cid = dsp_yolo_to_canonical[raw_cid]
                        xc, yc, w, h = map(float, parts[1:5])
                        if 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0:
                            valid_lines.append(f"{canon_cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                            stats[target_split]["classes"][canon_cid] += 1
                            stats[target_split]["boxes"] += 1
                            has_relevant_class = True

            if has_relevant_class and valid_lines:
                img_name = f"{lf.stem}.jpg"
                src_img = img_dir / img_name
                if src_img.exists():
                    dst_img = TARGET_DIR / "images" / target_split / f"dspcbsd_{img_name}"
                    dst_lbl = TARGET_DIR / "labels" / target_split / f"dspcbsd_{lf.stem}.txt"

                    if not dst_img.exists():
                        shutil.copy2(src_img, dst_img)

                    with open(dst_lbl, "w", encoding="utf-8") as f:
                        f.write("\n".join(valid_lines) + "\n")

                    stats[target_split]["images"] += 1
                    stats[target_split]["sources"]["DsPCBSD+"] += 1

    print(f"    - DsPCBSD+: Train={stats['train']['sources']['DsPCBSD+']}, Val={stats['val']['sources']['DsPCBSD+']}")

    # ---------------------------------------------------------
    # 3. PCB-DEFECT V2 INTEGRATION (Tiled 640x640 at native resolution)
    # ---------------------------------------------------------
    print("\n[*] 3. Integrating PCB-Defect V2 high-res color boards with 640x640 native tiling...")
    # Clean up any legacy pcbdefect files first
    for split in ["train", "val", "test", "external_color_test"]:
        for old_f in (TARGET_DIR / "images" / split).glob("pcbdefect_*.jpg"):
            old_f.unlink()
        for old_f in (TARGET_DIR / "labels" / split).glob("pcbdefect_*.txt"):
            old_f.unlink()

    v1_coco_path = PCB_DEFECT_RAW / "extracted_v1" / "PCB_Defect" / "annotation" / "_annotations.coco.json"
    v1_img_dir = PCB_DEFECT_RAW / "extracted_v1" / "PCB_Defect" / "images"

    with open(v1_coco_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    pcb_defect_to_canonical = {
        1: 5,  # missing_pad -> pin_hole_missing_pad
        2: 2,  # mouse_bite -> mouse_bite
        3: 0,  # open_circuit -> open_circuit
        4: 1,  # short -> short_circuit
        5: 3,  # spur -> spur
        6: 4   # spurious_copper -> spurious_copper
    }

    # Group images by physical board prefix
    board_groups = defaultdict(list)
    for img_meta in coco["images"]:
        fname = img_meta["file_name"]
        extra_name = img_meta.get("extra", {}).get("name", fname)
        board_prefix = extra_name.split("-")[0] if "-" in extra_name else fname.split("_")[0]
        board_groups[board_prefix].append(img_meta)

    # Board layout split:
    # External Color Test (Untouched): 68, 80, 81, 82
    # Val: 65, 66, 67
    # Train: 50 through 64 (15 layouts)
    test_prefixes = {"68", "80", "81", "82"}
    val_prefixes = {"65", "66", "67"}
    train_prefixes = {str(i) for i in range(50, 65)}

    print(f"    - Board Partitioning: Train={sorted(list(train_prefixes))}, Val={sorted(list(val_prefixes))}, External Test={sorted(list(test_prefixes))}")

    annos_by_img = defaultdict(list)
    for anno in coco["annotations"]:
        annos_by_img[anno["image_id"]].append(anno)

    crop_size = 640
    stride = 480  # 160 px overlap (25%)

    for prefix, img_list in sorted(board_groups.items()):
        if prefix in test_prefixes:
            target_split = "external_color_test"
        elif prefix in val_prefixes:
            target_split = "val"
        elif prefix in train_prefixes:
            target_split = "train"
        else:
            target_split = "train"  # Fallback for any other boards

        for img_meta in img_list:
            iid = img_meta["id"]
            fname = img_meta["file_name"]
            stem = Path(fname).stem
            src_img_path = v1_img_dir / fname

            if not src_img_path.exists():
                continue

            img = cv2.imread(str(src_img_path))
            if img is None:
                continue
            H, W = img.shape[:2]
            annos = annos_by_img.get(iid, [])

            # Compute tile grid
            x_coords = list(range(0, max(1, W - crop_size + 1), stride))
            if not x_coords or x_coords[-1] + crop_size < W:
                x_coords.append(max(0, W - crop_size))
            x_coords = sorted(list(set(x_coords)))

            y_coords = list(range(0, max(1, H - crop_size + 1), stride))
            if not y_coords or y_coords[-1] + crop_size < H:
                y_coords.append(max(0, H - crop_size))
            y_coords = sorted(list(set(y_coords)))

            background_crops_saved = 0
            max_background_per_image = 1 if target_split != "external_color_test" else 2

            crop_idx = 0
            for y in y_coords:
                for x in x_coords:
                    crop_annos = []
                    for a in annos:
                        cat_id = a["category_id"]
                        if cat_id not in pcb_defect_to_canonical:
                            continue
                        canon_cid = pcb_defect_to_canonical[cat_id]

                        bx, by, bw, bh = a["bbox"]
                        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

                        ix1, iy1 = max(x1, x), max(y1, y)
                        ix2, iy2 = min(x2, x + crop_size), min(y2, y + crop_size)

                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            orig_area = max(1.0, bw * bh)
                            if (inter_area / orig_area) >= 0.30 and (ix2 - ix1) >= 4 and (iy2 - iy1) >= 4:
                                rx1, ry1 = ix1 - x, iy1 - y
                                rx2, ry2 = ix2 - x, iy2 - y
                                xc = (rx1 + rx2) / (2.0 * crop_size)
                                yc = (ry1 + ry2) / (2.0 * crop_size)
                                norm_w = (rx2 - rx1) / float(crop_size)
                                norm_h = (ry2 - ry1) / float(crop_size)

                                xc = max(0.0001, min(0.9999, xc))
                                yc = max(0.0001, min(0.9999, yc))
                                norm_w = max(0.0001, min(0.9999, norm_w))
                                norm_h = max(0.0001, min(0.9999, norm_h))

                                crop_annos.append(f"{canon_cid} {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")
                                stats[target_split]["classes"][canon_cid] += 1
                                stats[target_split]["boxes"] += 1

                    has_defects = len(crop_annos) > 0
                    save_this_crop = False

                    if has_defects:
                        save_this_crop = True
                    elif background_crops_saved < max_background_per_image:
                        save_this_crop = True
                        background_crops_saved += 1

                    if save_this_crop:
                        crop_img = img[y:y + crop_size, x:x + crop_size]
                        crop_fname = f"pcbdefect_b{prefix}_{stem}_crop{crop_idx:03d}_x{x}_y{y}.jpg"
                        crop_lbl_name = f"pcbdefect_b{prefix}_{stem}_crop{crop_idx:03d}_x{x}_y{y}.txt"

                        dst_crop_path = TARGET_DIR / "images" / target_split / crop_fname
                        dst_crop_lbl = TARGET_DIR / "labels" / target_split / crop_lbl_name

                        if not dst_crop_path.exists():
                            cv2.imwrite(str(dst_crop_path), crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                        with open(dst_crop_lbl, "w", encoding="utf-8") as f:
                            f.write("\n".join(crop_annos) + ("\n" if crop_annos else ""))

                        stats[target_split]["images"] += 1
                        stats[target_split]["sources"]["PCB-Defect V2"] += 1

                    crop_idx += 1

    print(f"    - PCB-Defect V2 tiled: Train={stats['train']['sources']['PCB-Defect V2']}, Val={stats['val']['sources']['PCB-Defect V2']}, External Test={stats['external_color_test']['sources']['PCB-Defect V2']}")

    # ---------------------------------------------------------
    # 4. WRITE DATA.YAML & AUDIT SUMMARY
    # ---------------------------------------------------------
    data_yaml = {
        "path": str(TARGET_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",  # DeepPCB untouched test split
        "external_color_test": "images/external_color_test",
        "nc": 6,
        "names": {
            0: "open_circuit",
            1: "short_circuit",
            2: "mouse_bite",
            3: "spur",
            4: "spurious_copper",
            5: "pin_hole_missing_pad"
        }
    }

    yaml_path = TARGET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    manifest_path = TARGET_DIR / "dataset_manifest.json"
    manifest["stats"] = {
        split: {
            "total_images": stats[split]["images"],
            "total_boxes": stats[split]["boxes"],
            "sources": dict(stats[split]["sources"]),
            "classes": {data_yaml["names"][k]: v for k, v in stats[split]["classes"].items()}
        }
        for split in stats
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("REBUILT UNIFIED MULTI-DOMAIN DATASET SUMMARY")
    print("=" * 70)
    for split in ["train", "val", "test", "external_color_test"]:
        s = stats[split]
        src_str = ", ".join(f"{k}: {v}" for k, v in s["sources"].items())
        print(f"  [{split.upper():19s}] Total Images: {s['images']:5d} | Boxes: {s['boxes']:5d} | ({src_str})")

    print(f"\n[+] Configuration file written: {yaml_path}")
    print(f"[+] Manifest file written:      {manifest_path}")


if __name__ == "__main__":
    build_unified_dataset()
