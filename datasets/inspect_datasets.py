"""
Comprehensive Dataset Inspector for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Inspects:
- DsPCBSD+ (COCO & YOLO)
- DeepPCB (Image pairs & bbox annotations)
- PCB-Defect V2 (if available)

Outputs:
- datasets/reports/dataset_summary.csv
- datasets/reports/class_distribution.png
- Console summary
"""

import os
import sys
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = WORKSPACE_ROOT / "datasets" / "raw"
REPORTS_DIR = WORKSPACE_ROOT / "datasets" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def inspect_dspcbsd():
    print("\n" + "=" * 70)
    print("INSPECTING DATASET 1: DsPCBSD+")
    print("=" * 70)
    dspcbsd_dir = RAW_DIR / "dspcbsd"
    coco_anno_dir = dspcbsd_dir / "Data_COCO" / "annotations"
    
    if not coco_anno_dir.exists():
        print("[!] DsPCBSD+ COCO directory not found at:", coco_anno_dir)
        return None

    results = {
        "dataset": "DsPCBSD+",
        "total_images": 0,
        "total_annotations": 0,
        "classes": {},
        "splits": {},
        "image_sizes": set(),
        "categories": {}
    }

    for split_file in ["instances_train2017.json", "instances_val2017.json"]:
        path = coco_anno_dir / split_file
        if not path.exists():
            continue
        print(f"[*] Reading {split_file}...")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        split_name = split_file.replace("instances_", "").replace(".json", "")
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])

        cat_map = {c["id"]: c["name"] for c in categories}
        results["categories"] = cat_map

        for img in images:
            results["image_sizes"].add((img.get("width"), img.get("height")))

        class_counts = Counter()
        for ann in annotations:
            cat_id = ann.get("category_id")
            cat_name = cat_map.get(cat_id, f"unknown_{cat_id}")
            class_counts[cat_name] += 1

        results["splits"][split_name] = {
            "num_images": len(images),
            "num_annotations": len(annotations),
            "class_counts": dict(class_counts)
        }
        results["total_images"] += len(images)
        results["total_annotations"] += len(annotations)

        for c, count in class_counts.items():
            results["classes"][c] = results["classes"].get(c, 0) + count

    print(f"[+] Total Images in DsPCBSD+: {results['total_images']}")
    print(f"[+] Total Annotations:        {results['total_annotations']}")
    print(f"[+] Unique Image Resolutions: {list(results['image_sizes'])[:5]}")
    print(f"[+] Categories ({len(results['classes'])} classes):")
    for cat_name, cnt in sorted(results["classes"].items(), key=lambda x: -x[1]):
        print(f"    - {cat_name:25s}: {cnt:6d} instances")

    return results


def inspect_deeppcb():
    print("\n" + "=" * 70)
    print("INSPECTING DATASET 3: DeepPCB")
    print("=" * 70)
    deeppcb_dir = RAW_DIR / "deeppcb" / "PCBData"
    if not deeppcb_dir.exists():
        print("[!] DeepPCB PCBData directory not found at:", deeppcb_dir)
        return None

    class_id_map = {
        1: "open",
        2: "short",
        3: "mousebite",
        4: "spur",
        5: "copper",
        6: "pin-hole"
    }

    results = {
        "dataset": "DeepPCB",
        "num_pairs": 0,
        "num_groups": 0,
        "total_annotations": 0,
        "classes": Counter(),
        "groups": defaultdict(lambda: {"pairs": 0, "annotations": 0})
    }

    groups = [g for g in deeppcb_dir.iterdir() if g.is_dir()]
    results["num_groups"] = len(groups)

    for group in groups:
        group_name = group.name
        # Subdirectories typically like group00041/00041 and group00041/00041_not
        img_dirs = [d for d in group.iterdir() if d.is_dir() and not d.name.endswith("_not")]
        anno_dirs = [d for d in group.iterdir() if d.is_dir() and d.name.endswith("_not")]

        if not img_dirs or not anno_dirs:
            continue

        img_dir = img_dirs[0]
        anno_dir = anno_dirs[0]

        test_images = list(img_dir.glob("*_test.jpg"))
        temp_images = list(img_dir.glob("*_temp.jpg"))
        results["num_pairs"] += len(test_images)
        results["groups"][group_name]["pairs"] = len(test_images)

        for anno_file in anno_dir.glob("*.txt"):
            try:
                with open(anno_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            cls_id = int(parts[4])
                            cls_name = class_id_map.get(cls_id, f"unknown_{cls_id}")
                            results["classes"][cls_name] += 1
                            results["total_annotations"] += 1
                            results["groups"][group_name]["annotations"] += 1
            except Exception as e:
                pass

    print(f"[+] Total Distinct Board Groups: {results['num_groups']}")
    print(f"[+] Total Image Pairs:          {results['num_pairs']}")
    print(f"[+] Total Annotations:          {results['total_annotations']}")
    print(f"[+] Classes ({len(results['classes'])} classes):")
    for cat_name, cnt in sorted(results["classes"].items(), key=lambda x: -x[1]):
        print(f"    - {cat_name:25s}: {cnt:6d} instances")

    return results


def inspect_pcb_defect_v2():
    print("\n" + "=" * 70)
    print("INSPECTING DATASET 2: PCB-Defect V2")
    print("=" * 70)
    v2_dir = RAW_DIR / "pcb_defect_v2"
    if not v2_dir.exists():
        print("[!] PCB-Defect V2 directory not present.")
        return None

    json_files = list(v2_dir.glob("**/*.json"))
    img_files = list(v2_dir.glob("**/*.jpg")) + list(v2_dir.glob("**/*.png"))

    if not json_files and not img_files:
        print("[i] Status: PENDING DOWNLOAD (Manual Mendeley download required if blocked).")
        return {
            "dataset": "PCB-Defect V2",
            "total_images": 0,
            "total_annotations": 0,
            "classes": {},
            "status": "pending_download"
        }

    results = {
        "dataset": "PCB-Defect V2",
        "total_images": len(img_files),
        "total_annotations": 0,
        "classes": Counter(),
        "status": "downloaded"
    }

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            cat_map = {c["id"]: c["name"] for c in data.get("categories", [])}
            for ann in data.get("annotations", []):
                cid = ann.get("category_id")
                cname = cat_map.get(cid, f"class_{cid}")
                results["classes"][cname] += 1
                results["total_annotations"] += 1
        except Exception:
            pass

    print(f"[+] Images Found:      {results['total_images']}")
    print(f"[+] Annotations Found: {results['total_annotations']}")
    return results


def export_summary(dspcbsd_res, deeppcb_res, v2_res):
    csv_path = REPORTS_DIR / "dataset_summary.csv"
    print(f"\n[*] Writing dataset summary to {csv_path}...")
    
    rows = []
    
    if dspcbsd_res:
        for cname, count in dspcbsd_res["classes"].items():
            rows.append({
                "dataset": "DsPCBSD+",
                "source": "Figshare DOI: 10.6084/m9.figshare.24970329.v1",
                "licence": "CC BY 4.0",
                "class_name": cname,
                "count": count,
                "notes": "9 industrial classes, cropped patches"
            })
            
    if deeppcb_res:
        for cname, count in deeppcb_res["classes"].items():
            rows.append({
                "dataset": "DeepPCB",
                "source": "GitHub tangsanli5201/DeepPCB",
                "licence": "MIT License",
                "class_name": cname,
                "count": count,
                "notes": "Paired template/test 640x640 images across board groups"
            })

    if v2_res and v2_res["total_annotations"] > 0:
        for cname, count in v2_res["classes"].items():
            rows.append({
                "dataset": "PCB-Defect V2",
                "source": "Mendeley Data DOI: 10.17632/vdj74sngvn.2",
                "licence": "CC BY 4.0",
                "class_name": cname,
                "count": count,
                "notes": "Whole-board high-resolution images"
            })
    elif v2_res:
        rows.append({
            "dataset": "PCB-Defect V2",
            "source": "Mendeley Data DOI: 10.17632/vdj74sngvn.2",
            "licence": "CC BY 4.0",
            "class_name": "N/A (Pending)",
            "count": 0,
            "notes": "Awaiting local placement or manual download from Mendeley"
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "source", "licence", "class_name", "count", "notes"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Dataset summary exported to {csv_path}")

    # Plot class distribution if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if dspcbsd_res and dspcbsd_res["classes"]:
            d_classes = sorted(dspcbsd_res["classes"].items(), key=lambda x: -x[1])
            axes[0].barh([c[0] for c in d_classes], [c[1] for c in d_classes], color="teal")
            axes[0].set_title(f"DsPCBSD+ Defect Distribution ({dspcbsd_res['total_annotations']} total)")
            axes[0].set_xlabel("Instance Count")
            axes[0].invert_yaxis()

        if deeppcb_res and deeppcb_res["classes"]:
            p_classes = sorted(deeppcb_res["classes"].items(), key=lambda x: -x[1])
            axes[1].barh([c[0] for c in p_classes], [c[1] for c in p_classes], color="darkorange")
            axes[1].set_title(f"DeepPCB Defect Distribution ({deeppcb_res['total_annotations']} total)")
            axes[1].set_xlabel("Instance Count")
            axes[1].invert_yaxis()

        plt.tight_layout()
        plot_path = REPORTS_DIR / "class_distribution.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[+] Class distribution plot saved to {plot_path}")
    except Exception as e:
        print(f"[i] Matplotlib plot generation deferred: {e}")


def main():
    dspcbsd_res = inspect_dspcbsd()
    deeppcb_res = inspect_deeppcb()
    v2_res = inspect_pcb_defect_v2()
    export_summary(dspcbsd_res, deeppcb_res, v2_res)


if __name__ == "__main__":
    main()
