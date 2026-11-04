"""
Audit Class Definitions and Taxonomies for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Analyzes:
1. DeepPCB: class 6 ('pin-hole')
2. DsPCBSD+: category 'HB' ('hole_breakout')
3. PCB-Defect V2: class 0 ('missing_pad')

Outputs counts, physical dimensions, aspect ratios, and domain context.
"""

import json
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

def audit_taxonomy():
    results = {}

    # 1. DeepPCB
    deeppcb_root = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData"
    deeppcb_files = list(deeppcb_root.rglob("*_not/*.txt"))
    deeppcb_classes = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}
    deeppcb_counts = {v: 0 for v in deeppcb_classes.values()}
    pin_hole_boxes = []

    for p in deeppcb_files:
        with open(p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    x1, y1, x2, y2, c = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                    cname = deeppcb_classes.get(c, "unknown")
                    deeppcb_counts[cname] = deeppcb_counts.get(cname, 0) + 1
                    if cname == "pin-hole":
                        w = x2 - x1
                        h = y2 - y1
                        pin_hole_boxes.append((w, h))

    # 2. DsPCBSD+
    dspcbsd_train = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_COCO" / "annotations" / "instances_train2017.json"
    dspcbsd_val = WORKSPACE_ROOT / "datasets" / "raw" / "dspcbsd" / "Data_COCO" / "annotations" / "instances_val2017.json"
    dspcbsd_counts = {}
    hb_boxes = []

    for name, p in [("train", dspcbsd_train), ("val", dspcbsd_val)]:
        if p.exists():
            with open(p, "r") as f:
                data = json.load(f)
            cats = {c["id"]: c["name"] for c in data["categories"]}
            for a in data["annotations"]:
                cname = cats.get(a["category_id"], "unknown")
                dspcbsd_counts[cname] = dspcbsd_counts.get(cname, 0) + 1
                if cname == "HB":
                    # bbox: [x, y, w, h]
                    w, h = a["bbox"][2], a["bbox"][3]
                    hb_boxes.append((w, h))

    # 3. PCB-Defect V2
    v2_label_dir = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train" / "labels"
    v2_classes = {0: "missing_pad", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: "spur", 5: "spurious_copper"}
    v2_counts = {v: 0 for v in v2_classes.values()}
    missing_pad_boxes = []

    if v2_label_dir.exists():
        for p in v2_label_dir.glob("*.txt"):
            with open(p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        cid = int(parts[0])
                        cname = v2_classes.get(cid, "unknown")
                        v2_counts[cname] = v2_counts.get(cname, 0) + 1
                        if cname == "missing_pad":
                            xs = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
                            ys = [float(parts[2]), float(parts[4]), float(parts[6]), float(parts[8])]
                            w = max(xs) - min(xs)
                            h = max(ys) - min(ys)
                            missing_pad_boxes.append((w, h))

    print("=" * 70)
    print("CLASS TAXONOMY PHYSICAL AUDIT REPORT")
    print("=" * 70)
    print("\n1. DeepPCB (Synthetic Scanned Monochrome Pairs):")
    for k, v in deeppcb_counts.items():
        print(f"   - {k:15s}: {v}")
    if pin_hole_boxes:
        ws = [b[0] for b in pin_hole_boxes]
        hs = [b[1] for b in pin_hole_boxes]
        ars = [max(w, h)/max(1, min(w, h)) for w, h in pin_hole_boxes]
        print(f"   * 'pin-hole' instances: {len(pin_hole_boxes)}")
        print(f"     Median size: {np.median(ws):.1f} x {np.median(hs):.1f} px, Median Aspect Ratio: {np.median(ars):.2f}")

    print("\n2. DsPCBSD+ (Industrial Optical Scanner Crops, 640x640):")
    for k, v in sorted(dspcbsd_counts.items()):
        print(f"   - {k:15s}: {v}")
    if hb_boxes:
        ws = [b[0] for b in hb_boxes]
        hs = [b[1] for b in hb_boxes]
        ars = [max(w, h)/max(1, min(w, h)) for w, h in hb_boxes]
        print(f"   * 'HB' (hole_breakout) instances: {len(hb_boxes)}")
        print(f"     Median size: {np.median(ws):.1f} x {np.median(hs):.1f} px, Median Aspect Ratio: {np.median(ars):.2f}")

    print("\n3. PCB-Defect V2 (High-Resolution Whole-Board RGB Photos):")
    for k, v in v2_counts.items():
        print(f"   - {k:15s}: {v}")
    print(f"   * 'missing_pad' instances (normalized across multi-MP boards): {len(missing_pad_boxes)}")

    # Summary of findings
    summary = {
        "deeppcb_pin_hole_count": len(pin_hole_boxes),
        "dspcbsd_hb_count": len(hb_boxes),
        "v2_missing_pad_count": len(missing_pad_boxes),
        "deeppcb_has_missing_pad": False,
        "deeppcb_has_hole_breakout": False,
        "dspcbsd_has_pin_hole": False,
        "dspcbsd_has_missing_pad": False,
        "v2_has_pin_hole": False,
        "v2_has_hole_breakout": False
    }

    out_file = WORKSPACE_ROOT / "benchmarks" / "results" / "class_taxonomy_audit.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[+] Saved audit results to {out_file}")

if __name__ == "__main__":
    audit_taxonomy()
