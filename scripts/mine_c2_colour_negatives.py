"""
Mining Candidate Colour Hard Negatives for Experiment C2
EdgePCB-QC — Bharat AI-SoC Challenge 2026-27 | PS5

Rules enforced:
1. Mines ONLY from PCB-Defect V2 TRAIN boards (50 to 64).
2. NEVER touches validation boards (65 to 67) or test boards (68, 80, 81, 82).
3. Evaluates predictions from exp_c_hard_negatives.
4. Identifies false positives (predictions with 0.0 IoU against ground-truth defects).
5. Ensures candidate 640x640 tiles contain ZERO ground-truth defect annotations (strict clean safety margin).
6. Outputs candidate tiles, metadata manifest, and visual contact sheets with IDs.
7. STOPS for user review and approval before any tile is labeled as a negative.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO
from inspection.tiling import TiledInferenceRunner, compute_iou

CANONICAL_6_CLASSES = {
    0: "open_circuit",
    1: "short_circuit",
    2: "mouse_bite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole_missing_pad"
}

V2_CLASS_REMAP = {0: 5, 1: 2, 2: 0, 3: 1, 4: 3, 5: 4}

TRAIN_BOARDS = ["50", "51", "52", "53", "54", "55", "56", "57", "58", "59", "60", "61", "62", "63", "64"]
RAW_V2_DIR = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train"
OUTPUT_DIR = WORKSPACE_ROOT / "benchmarks" / "results" / "c2_mining_candidates"
CHECKPOINT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "exp_c_hard_negatives" / "weights" / "best.pt"


def load_v2_ground_truth(lbl_p: Path, img_w: int, img_h: int):
    """Loads oriented 4-corner bounding boxes from PCB-Defect V2 and converts to axis-aligned xyxy."""
    gt_boxes = []
    if not lbl_p.exists():
        return gt_boxes
    with open(lbl_p, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                raw_cid = int(parts[0])
                canonical_cid = V2_CLASS_REMAP.get(raw_cid, raw_cid)
                xs = [float(parts[1]) * img_w, float(parts[3]) * img_w, float(parts[5]) * img_w, float(parts[7]) * img_w]
                ys = [float(parts[2]) * img_h, float(parts[4]) * img_h, float(parts[6]) * img_h, float(parts[8]) * img_h]
                gt_boxes.append({
                    "class_id": canonical_cid,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "class_name": CANONICAL_6_CLASSES.get(canonical_cid, "defect")
                })
    return gt_boxes


def box_overlaps_any(box, gt_boxes, min_iou=0.01):
    """Checks if box overlaps any GT box."""
    for gt in gt_boxes:
        if compute_iou(box, gt["bbox"]) >= min_iou:
            return True
    return False


def box_intersects_box(b1, b2, margin=0):
    """Checks if two rectangles intersect with optional margin."""
    return not (b1[2] + margin < b2[0] or
                b1[0] - margin > b2[2] or
                b1[3] + margin < b2[1] or
                b1[1] - margin > b2[3])


def mine_colour_negatives():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles_dir = OUTPUT_DIR / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("MINING CANDIDATE COLOUR HARD NEGATIVES FOR EXPERIMENT C2")
    print("=" * 75)
    print(f"[*] Checkpoint: {CHECKPOINT_PATH}")
    print(f"[*] Allowed source boards: {TRAIN_BOARDS}")
    print(f"[*] Output directory: {OUTPUT_DIR}")

    model = YOLO(str(CHECKPOINT_PATH))

    # Adapter for TiledInferenceRunner
    class InspectorAdapter:
        def __init__(self, m):
            self.m = m
        def inspect(self, img_arr):
            res = self.m.predict(source=img_arr, conf=0.25, imgsz=640, device=0, verbose=False)[0]
            dets = []
            for b in res.boxes:
                cid = int(b.cls[0].cpu().numpy())
                cf = float(b.conf[0].cpu().numpy())
                box = b.xyxy[0].cpu().numpy().tolist()
                dets.append({
                    "class_id": cid,
                    "confidence": cf,
                    "bbox": box,
                    "class": CANONICAL_6_CLASSES.get(cid, "defect")
                })
            return dets

    adapter = InspectorAdapter(model)
    tiler = TiledInferenceRunner(
        inspector=adapter,
        tile_size=640,
        overlap=0.40,
        iou_threshold=0.30,
        edge_padding=True,
        class_agnostic_nms=True
    )

    # Gather training board images
    train_imgs = []
    for b in TRAIN_BOARDS:
        found = list((RAW_V2_DIR / "images").glob(f"{b}-*.jpg"))
        train_imgs.extend(found)
    train_imgs.sort()
    print(f"[*] Total training board images to scan: {len(train_imgs)}")

    candidate_records = []
    tile_size = 640
    candidate_id_counter = 1

    for img_p in train_imgs:
        board_id = img_p.name.split("-")[0]
        lbl_p = RAW_V2_DIR / "labels" / (img_p.stem + ".txt")

        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = load_v2_ground_truth(lbl_p, w, h)

        # Run tiled inference
        predictions = tiler.inspect_tiled(img)

        # Filter for False Positives
        for pred in predictions:
            p_box = pred["bbox"]
            p_cid = pred["class_id"]
            p_conf = pred["confidence"]
            p_cname = pred["class"]

            # If it overlaps any real defect, it's a true positive or near-defect detection -> SKIP
            if box_overlaps_any(p_box, gt_boxes, min_iou=0.01):
                continue

            # It is a candidate false positive!
            # Compute a 640x640 tile centered on the detection
            cx = (p_box[0] + p_box[2]) / 2.0
            cy = (p_box[1] + p_box[3]) / 2.0

            x1 = int(round(cx - tile_size / 2.0))
            y1 = int(round(cy - tile_size / 2.0))

            # Clamp to image boundaries
            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            if x1 + tile_size > w:
                x1 = max(0, w - tile_size)
            if y1 + tile_size > h:
                y1 = max(0, h - tile_size)

            x2 = min(w, x1 + tile_size)
            y2 = min(h, y1 + tile_size)

            tile_bbox = [x1, y1, x2, y2]

            # SAFETY INVARIANT: Check if ANY ground truth defect touches this 640x640 tile!
            has_gt_in_tile = False
            for gt in gt_boxes:
                if box_intersects_box(tile_bbox, gt["bbox"], margin=20):
                    has_gt_in_tile = True
                    break

            if has_gt_in_tile:
                # Reject this tile: real defect present in background margin
                continue

            # Extract the clean tile
            tile_img = img[y1:y2, x1:x2]
            if tile_img.shape[0] != tile_size or tile_img.shape[1] != tile_size:
                tile_img = cv2.resize(tile_img, (tile_size, tile_size))

            # Compute relative coordinates of the FP prediction inside the tile
            local_fp_box = [
                max(0.0, p_box[0] - x1),
                max(0.0, p_box[1] - y1),
                min(float(tile_size), p_box[2] - x1),
                min(float(tile_size), p_box[3] - y1)
            ]

            cand_id = f"C2_CAND_{candidate_id_counter:04d}"
            candidate_id_counter += 1

            tile_fname = f"{cand_id}_{board_id}_{img_p.stem}_fp_{p_cname}.jpg"
            tile_path = tiles_dir / tile_fname
            cv2.imwrite(str(tile_path), tile_img)

            # Compute sha256
            sha256 = hashlib.sha256(tile_img.tobytes()).hexdigest()

            candidate_records.append({
                "candidate_id": cand_id,
                "board_id": board_id,
                "source_image": img_p.name,
                "tile_filename": tile_fname,
                "crop_global_xyxy": [x1, y1, x2, y2],
                "pred_class_id": p_cid,
                "pred_class_name": p_cname,
                "pred_confidence": round(p_conf, 4),
                "local_fp_bbox": [round(coord, 1) for coord in local_fp_box],
                "sha256": sha256
            })

    print(f"\n[+] Total candidate false-positive tiles mined: {len(candidate_records)}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "candidates_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(candidate_records, f, indent=2)
    print(f"[+] Saved manifest to: {manifest_path}")

    # Generate Contact Sheets for Human Review (20 tiles per sheet: 4 cols x 5 rows)
    generate_contact_sheets(candidate_records, tiles_dir, OUTPUT_DIR)

    return candidate_records


def generate_contact_sheets(records, tiles_dir, output_dir):
    """Renders visual contact sheets with candidate IDs, board names, and predicted FP boxes for operator review."""
    cols = 4
    rows = 5
    per_page = cols * rows
    thumb_w, thumb_h = 320, 320
    header_h = 60

    num_pages = (len(records) + per_page - 1) // per_page
    print(f"[*] Generating {num_pages} contact sheet pages for visual review...")

    for page_idx in range(num_pages):
        page_records = records[page_idx * per_page : (page_idx + 1) * per_page]
        sheet_w = cols * thumb_w
        sheet_h = rows * (thumb_h + header_h)

        sheet = Image.new("RGB", (sheet_w, sheet_h), color=(240, 240, 245))
        draw = ImageDraw.Draw(sheet)

        for idx, rec in enumerate(page_records):
            c = idx % cols
            r = idx // cols
            x_offset = c * thumb_w
            y_offset = r * (thumb_h + header_h)

            tile_p = tiles_dir / rec["tile_filename"]
            if not tile_p.exists():
                continue

            im = Image.open(tile_p).convert("RGB")
            # Scale factor for drawn box
            orig_w, orig_h = im.size
            scale_x = thumb_w / orig_w
            scale_y = thumb_h / orig_h

            im_thumb = im.resize((thumb_w, thumb_h), Image.Resampling.BILINEAR)
            thumb_draw = ImageDraw.Draw(im_thumb)

            # Draw local predicted box
            box = rec["local_fp_bbox"]
            bx1 = box[0] * scale_x
            by1 = box[1] * scale_y
            bx2 = box[2] * scale_x
            by2 = box[3] * scale_y
            thumb_draw.rectangle([bx1, by1, bx2, by2], outline="red", width=3)

            # Paste image onto sheet
            sheet.paste(im_thumb, (x_offset, y_offset + header_h))

            # Draw header banner
            draw.rectangle([x_offset, y_offset, x_offset + thumb_w, y_offset + header_h], fill=(30, 41, 59))
            text_line1 = f"{rec['candidate_id']} | Bd: {rec['board_id']}"
            text_line2 = f"Pred: {rec['pred_class_name']} ({rec['pred_confidence']:.2f})"
            draw.text((x_offset + 8, y_offset + 8), text_line1, fill=(255, 255, 255))
            draw.text((x_offset + 8, y_offset + 32), text_line2, fill=(250, 204, 21))

            # Cell border
            draw.rectangle([x_offset, y_offset, x_offset + thumb_w, y_offset + header_h + thumb_h], outline=(200, 200, 200), width=1)

        sheet_path = output_dir / f"contact_sheet_page_{page_idx + 1:02d}.png"
        sheet.save(sheet_path)
        print(f"    * Saved: {sheet_path}")

    print("\n[!] MINING COMPLETE. REVIEW CONTACT SHEETS AND MANIFEST.")


if __name__ == "__main__":
    mine_colour_negatives()
