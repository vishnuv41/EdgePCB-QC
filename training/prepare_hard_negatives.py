"""
Prepare Clean-Board Hard-Negative Dataset for EdgePCB-QC (Experiment C)
Bharat AI-SoC Challenge 2026-27 | PS5

Integrates 1,120 verified clean template images from DeepPCB train groups
into the 6-class training dataset as empty-label hard negatives.
Strictly preserves test groups (group90100, group92000) as held-out test clean boards.
"""

import os
import shutil
import _winapi
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SRC_DATASET = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo"
DST_DATASET = WORKSPACE_ROOT / "datasets" / "processed" / "hard_negatives_yolo"
RAW_DEEPPCB = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData"

TRAIN_GROUPS = [
    "group00041", "group12000", "group12100", "group12300",
    "group13000", "group20085", "group44000", "group50600", "group77000"
]
TEST_GROUPS = ["group90100", "group92000"]

def prepare_hard_negatives():
    print("=" * 70)
    print("PREPARING HARD-NEGATIVE DATASET (EXPERIMENT C)")
    print("=" * 70)

    DST_DATASET.mkdir(parents=True, exist_ok=True)
    images_train = DST_DATASET / "images" / "train"
    labels_train = DST_DATASET / "labels" / "train"
    images_val = DST_DATASET / "images" / "val"
    labels_val = DST_DATASET / "labels" / "val"

    for d in [images_train, labels_train, images_val, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Symlink / copy existing 9,307 train images and labels
    print("[*] Linking baseline unified train & val datasets...")
    src_img_train = SRC_DATASET / "images" / "train"
    src_lbl_train = SRC_DATASET / "labels" / "train"

    # Link/copy labels
    train_existing = 0
    for p in src_lbl_train.glob("*.txt"):
        dst_p = labels_train / p.name
        if not dst_p.exists():
            shutil.copy2(p, dst_p)
        train_existing += 1

    # Link images using hard links or copy
    for p in src_img_train.glob("*.jpg"):
        dst_p = images_train / p.name
        if not dst_p.exists():
            # Create hardlink to save space
            try:
                os.link(str(p), str(dst_p))
            except Exception:
                shutil.copy2(p, dst_p)

    # Validation split junction/link
    for p in (SRC_DATASET / "labels" / "val").glob("*.txt"):
        dst_p = labels_val / p.name
        if not dst_p.exists():
            shutil.copy2(p, dst_p)
    for p in (SRC_DATASET / "images" / "val").glob("*.jpg"):
        dst_p = images_val / p.name
        if not dst_p.exists():
            try:
                os.link(str(p), str(dst_p))
            except Exception:
                shutil.copy2(p, dst_p)

    print(f"[+] Linked {train_existing} existing training images & labels.")

    # 2. Extract verified clean template images from train groups
    clean_added = 0
    for g in TRAIN_GROUPS:
        g_dir = RAW_DEEPPCB / g
        if not g_dir.exists():
            continue
        temp_imgs = list(g_dir.rglob("*_temp.jpg"))
        for t in temp_imgs:
            # Name: clean_deeppcb_<group>_<id>_temp.jpg
            board_id = t.stem
            out_img_name = f"clean_deeppcb_{g}_{board_id}.jpg"
            out_lbl_name = f"clean_deeppcb_{g}_{board_id}.txt"

            out_img_p = images_train / out_img_name
            out_lbl_p = labels_train / out_lbl_name

            if not out_img_p.exists():
                shutil.copy2(t, out_img_p)

            # Create empty label file (0 bytes) for hard-negative background learning
            if not out_lbl_p.exists():
                out_lbl_p.touch()

            clean_added += 1

    print(f"[+] Ingested {clean_added} verified clean template background images (empty labels).")
    print(f"[+] Total training set size: {train_existing + clean_added} images.")

    # 3. Create data.yaml
    yaml_content = f"""path: {DST_DATASET.resolve()}
train: images/train
val: images/val
test: {SRC_DATASET.resolve() / "images" / "test"}
external_color_test: {SRC_DATASET.resolve() / "images" / "external_color_test"}
nc: 6
names:
  0: open_circuit
  1: short_circuit
  2: mouse_bite
  3: spur
  4: spurious_copper
  5: pin_hole_missing_pad
"""
    with open(DST_DATASET / "data.yaml", "w") as f:
        f.write(yaml_content)

    print(f"[+] Saved {DST_DATASET / 'data.yaml'}")
    print("[+] Clean-Board Hard-Negative Dataset Ready!\n")

if __name__ == "__main__":
    prepare_hard_negatives()
