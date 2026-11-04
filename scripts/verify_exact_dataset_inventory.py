"""
Exact Dataset Accounting and Model Inventory Verification Script
Bharat AI-SoC Challenge 2026-27 | PS5

Directly inspects raw and processed datasets, labels, and training states.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def audit_all():
    print("=" * 80)
    print("EXACT DATASET AND MODEL INVENTORY AUDIT")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. DeepPCB Audit
    # ---------------------------------------------------------
    deeppcb_raw = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData"
    # Groups
    deeppcb_groups = sorted([d.name for d in deeppcb_raw.iterdir() if d.is_dir() and d.name.startswith("group")])
    train_groups = ["group00041", "group12000", "group12100", "group12300", "group13000", "group20085", "group44000", "group50600", "group77000"]
    val_groups = ["group12000", "group12100"] # Wait, check exact splits from prepare_data.py / prepare_unified_dataset.py
    test_groups = ["group90100", "group92000"]

    # Let's count test.txt and trainval.txt in PCBData
    test_txt = deeppcb_raw / "test.txt"
    trainval_txt = deeppcb_raw / "trainval.txt"
    test_txt_lines = [l.strip() for l in open(test_txt) if l.strip()] if test_txt.exists() else []
    trainval_txt_lines = [l.strip() for l in open(trainval_txt) if l.strip()] if trainval_txt.exists() else []

    all_test_imgs = list(deeppcb_raw.rglob("*_test.jpg"))
    all_temp_imgs = list(deeppcb_raw.rglob("*_temp.jpg"))
    all_labels = list(deeppcb_raw.rglob("*_not/*.txt"))

    # Count raw DeepPCB bboxes
    deeppcb_class_map = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}
    raw_deeppcb_bboxes = Counter()
    for lp in all_labels:
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[4])
                    raw_deeppcb_bboxes[deeppcb_class_map.get(cid, str(cid))] += 1

    # ---------------------------------------------------------
    # 2. DsPCBSD+ Audit
    # ---------------------------------------------------------
    dspcbsd_raw_train_img = list((WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_COCO" / "train2017").glob("*.jpg"))
    dspcbsd_raw_val_img = list((WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_COCO" / "val2017").glob("*.jpg"))
    
    train_coco = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_COCO" / "annotations" / "instances_train2017.json"
    val_coco = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_COCO" / "annotations" / "instances_val2017.json"
    
    dspcbsd_train_cats = Counter()
    dspcbsd_val_cats = Counter()
    dspcbsd_train_img_ids_with_ann = set()
    dspcbsd_val_img_ids_with_ann = set()

    if train_coco.exists():
        with open(train_coco) as f:
            d = json.load(f)
        c_map = {c["id"]: c["name"] for c in d["categories"]}
        for a in d["annotations"]:
            cname = c_map.get(a["category_id"])
            dspcbsd_train_cats[cname] += 1
            dspcbsd_train_img_ids_with_ann.add(a["image_id"])

    if val_coco.exists():
        with open(val_coco) as f:
            d = json.load(f)
        c_map = {c["id"]: c["name"] for c in d["categories"]}
        for a in d["annotations"]:
            cname = c_map.get(a["category_id"])
            dspcbsd_val_cats[cname] += 1
            dspcbsd_val_img_ids_with_ann.add(a["image_id"])

    # ---------------------------------------------------------
    # 3. PCB-Defect V2 Audit
    # ---------------------------------------------------------
    v2_dir = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated"
    v2_raw_imgs = list((v2_dir / "train" / "images").glob("*.jpg"))
    v2_raw_labels = list((v2_dir / "train" / "labels").glob("*.txt"))

    # Boards
    board_ids = set()
    for p in v2_raw_imgs:
        # e.g. 50-1-1.jpg -> board 50
        b_id = p.stem.split("-")[0]
        board_ids.add(b_id)

    v2_class_names = {0: "missing_pad", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"}
    v2_raw_bboxes = Counter()
    for lp in v2_raw_labels:
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    cid = int(parts[0])
                    v2_raw_bboxes[v2_class_names.get(cid, str(cid))] += 1

    # ---------------------------------------------------------
    # 4. Processed Unified Dataset Counts
    # ---------------------------------------------------------
    unified_dir = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo"
    unified_train_imgs = list((unified_dir / "images" / "train").glob("*.jpg"))
    unified_val_imgs = list((unified_dir / "images" / "val").glob("*.jpg"))
    unified_test_imgs = list((unified_dir / "images" / "test").glob("*.jpg"))
    unified_ext_imgs = list((unified_dir / "images" / "external_color_test").glob("*.jpg"))

    # Breakdown by dataset prefix in train/val/test/ext
    def get_source_breakdown(img_list):
        counts = Counter()
        for p in img_list:
            if p.name.startswith("deeppcb_") or p.name.startswith("dp_"):
                counts["deeppcb"] += 1
            elif p.name.startswith("dspcbsd_") or p.name.startswith("ds_"):
                counts["dspcbsd"] += 1
            elif p.name.startswith("pcbdefect_") or p.name.startswith("v2_"):
                counts["pcb_defect_v2"] += 1
            else:
                counts["other"] += 1
        return counts

    # Processed bboxes per class across unified splits
    canonical_names = {0: "open_circuit", 1: "short_circuit", 2: "mouse_bite", 3: "spur", 4: "spurious_copper", 5: "pin_hole_missing_pad"}
    unified_bboxes_per_split = {}
    for split in ["train", "val", "test", "external_color_test"]:
        s_lbl_dir = unified_dir / "labels" / split
        s_counts = Counter()
        for p in s_lbl_dir.glob("*.txt"):
            with open(p) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        s_counts[canonical_names.get(cid, str(cid))] += 1
        unified_bboxes_per_split[split] = s_counts

    out = {
        "deeppcb": {
            "total_test_imgs": len(all_test_imgs),
            "total_temp_imgs": len(all_temp_imgs),
            "total_pairs": len(all_test_imgs),
            "trainval_txt_count": len(trainval_txt_lines),
            "test_txt_count": len(test_txt_lines),
            "raw_bboxes": dict(raw_deeppcb_bboxes),
            "train_used": get_source_breakdown(unified_train_imgs)["deeppcb"],
            "val_used": get_source_breakdown(unified_val_imgs)["deeppcb"],
            "test_used": get_source_breakdown(unified_test_imgs)["deeppcb"],
            "ext_used": get_source_breakdown(unified_ext_imgs)["deeppcb"],
        },
        "dspcbsd": {
            "raw_train_imgs": len(dspcbsd_raw_train_img),
            "raw_val_imgs": len(dspcbsd_raw_val_img),
            "total_raw_imgs": len(dspcbsd_raw_train_img) + len(dspcbsd_raw_val_img),
            "raw_train_bboxes": dict(dspcbsd_train_cats),
            "raw_val_bboxes": dict(dspcbsd_val_cats),
            "train_used": get_source_breakdown(unified_train_imgs)["dspcbsd"],
            "val_used": get_source_breakdown(unified_val_imgs)["dspcbsd"],
            "test_used": get_source_breakdown(unified_test_imgs)["dspcbsd"],
            "ext_used": get_source_breakdown(unified_ext_imgs)["dspcbsd"],
        },
        "pcb_defect_v2": {
            "raw_whole_boards": len(v2_raw_imgs),
            "raw_board_ids": sorted(list(board_ids)),
            "raw_bboxes": dict(v2_raw_bboxes),
            "train_used_crops": get_source_breakdown(unified_train_imgs)["pcb_defect_v2"],
            "val_used_crops": get_source_breakdown(unified_val_imgs)["pcb_defect_v2"],
            "test_used_crops": get_source_breakdown(unified_test_imgs)["pcb_defect_v2"],
            "ext_used_crops": get_source_breakdown(unified_ext_imgs)["pcb_defect_v2"],
        },
        "unified_splits_total": {
            "train": len(unified_train_imgs),
            "val": len(unified_val_imgs),
            "test": len(unified_test_imgs),
            "external_color_test": len(unified_ext_imgs),
            "total_images": len(unified_train_imgs) + len(unified_val_imgs) + len(unified_test_imgs) + len(unified_ext_imgs),
            "bboxes_per_split": {k: dict(v) for k, v in unified_bboxes_per_split.items()}
        }
    }

    print(json.dumps(out, indent=2))
    with open(WORKSPACE_ROOT / "benchmarks" / "results" / "exact_inventory_counts.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    audit_all()
