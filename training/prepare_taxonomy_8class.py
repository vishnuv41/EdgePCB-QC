"""
Prepare 8-Class Split Taxonomy Dataset for EdgePCB-QC (Experiment B)
Bharat AI-SoC Challenge 2026-27 | PS5

Splits canonical Class 5 (pin_hole_missing_pad) into 3 distinct classes:
- Class 5: pin_hole (DeepPCB)
- Class 6: hole_breakout (DsPCBSD+)
- Class 7: missing_pad (PCB-Defect V2)
Uses NTFS directory junction to reuse image files with zero disk duplication.
"""

import os
import _winapi
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SRC_DATASET = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo"
DST_DATASET = WORKSPACE_ROOT / "datasets" / "processed" / "taxonomy_8class_yolo"

def prepare_taxonomy_8class():
    print("=" * 70)
    print("PREPARING 8-CLASS SPLIT TAXONOMY DATASET (EXPERIMENT B)")
    print("=" * 70)

    DST_DATASET.mkdir(parents=True, exist_ok=True)

    # 1. Directory junction for images
    dst_images = DST_DATASET / "images"
    if not dst_images.exists():
        src_images = (SRC_DATASET / "images").resolve()
        _winapi.CreateJunction(str(src_images), str(dst_images.resolve()))
        print(f"[+] Created NTFS image junction: {dst_images} -> {src_images}")
    else:
        print(f"[*] Image directory already linked: {dst_images}")

    # 2. Process and remap labels for train and val splits
    for split in ["train", "val", "test", "external_color_test"]:
        src_split_dir = SRC_DATASET / "labels" / split
        if not src_split_dir.exists():
            continue
        dst_split_dir = DST_DATASET / "labels" / split
        dst_split_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        remapped_counts = {5: 0, 6: 0, 7: 0}

        for p in src_split_dir.glob("*.txt"):
            count += 1
            fname = p.name
            out_lines = []
            with open(p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        coords = parts[1:]
                        if cid == 5:
                            if fname.startswith("deeppcb_"):
                                new_cid = 5  # pin_hole
                                remapped_counts[5] += 1
                            elif fname.startswith("dspcbsd_"):
                                new_cid = 6  # hole_breakout
                                remapped_counts[6] += 1
                            elif fname.startswith("pcbdefect_"):
                                new_cid = 7  # missing_pad
                                remapped_counts[7] += 1
                            else:
                                new_cid = 5
                        else:
                            new_cid = cid
                        out_lines.append(f"{new_cid} {' '.join(coords)}")
                    else:
                        out_lines.append(line.strip())

            dst_p = dst_split_dir / fname
            with open(dst_p, "w") as f:
                f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

        print(f"[+] Processed {split}: {count} label files.")
        print(f"    - Class 5 (pin_hole):     {remapped_counts[5]}")
        print(f"    - Class 6 (hole_breakout): {remapped_counts[6]}")
        print(f"    - Class 7 (missing_pad):   {remapped_counts[7]}")

    # 3. Write data.yaml
    yaml_content = f"""path: {DST_DATASET.resolve()}
train: images/train
val: images/val
test: images/test
external_color_test: images/external_color_test
nc: 8
names:
  0: open_circuit
  1: short_circuit
  2: mouse_bite
  3: spur
  4: spurious_copper
  5: pin_hole
  6: hole_breakout
  7: missing_pad
"""
    data_yaml_path = DST_DATASET / "data.yaml"
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[+] Generated {data_yaml_path}")
    print("[+] 8-Class Split Taxonomy Dataset Ready!\n")

if __name__ == "__main__":
    prepare_taxonomy_8class()
