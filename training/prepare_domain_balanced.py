"""
Prepare Domain-Balanced Dataset for EdgePCB-QC (Experiment D)
Bharat AI-SoC Challenge 2026-27 | PS5

Enforces equal representation across the three data modalities (1:1:1 domain parity):
- DeepPCB (monochrome scanner): 1,500 images (all 1,120 unique + 380 balanced samples)
- DsPCBSD+ (industrial optical scanner): 1,500 images (sampled from 5,811)
- PCB-Defect V2 (high-res colour photos): 1,500 images (sampled from 2,376)
Total training size = 4,500 domain-balanced images (33.3% per modality).
"""

import os
import random
import shutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SRC_DATASET = WORKSPACE_ROOT / "datasets" / "processed" / "unified_yolo"
DST_DATASET = WORKSPACE_ROOT / "datasets" / "processed" / "domain_balanced_yolo"

def prepare_domain_balanced(seed: int = 42):
    print("=" * 70)
    print("PREPARING DOMAIN-BALANCED DATASET (EXPERIMENT D)")
    print("=" * 70)

    random.seed(seed)
    DST_DATASET.mkdir(parents=True, exist_ok=True)
    images_train = DST_DATASET / "images" / "train"
    labels_train = DST_DATASET / "labels" / "train"
    images_val = DST_DATASET / "images" / "val"
    labels_val = DST_DATASET / "labels" / "val"

    for d in [images_train, labels_train, images_val, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    src_img_train = SRC_DATASET / "images" / "train"
    src_lbl_train = SRC_DATASET / "labels" / "train"

    # Group existing images by domain prefix
    all_train_files = list(src_lbl_train.glob("*.txt"))
    deeppcb_files = [f for f in all_train_files if f.name.startswith("deeppcb_")]
    dspcbsd_files = [f for f in all_train_files if f.name.startswith("dspcbsd_")]
    pcbdefect_files = [f for f in all_train_files if f.name.startswith("pcbdefect_")]

    print(f"[*] Available training files:")
    print(f"    - DeepPCB:     {len(deeppcb_files)}")
    print(f"    - DsPCBSD+:    {len(dspcbsd_files)}")
    print(f"    - PCB-Defect:  {len(pcbdefect_files)}")

    target_per_domain = 1500

    # 1. DeepPCB: 1120 unique + 380 oversampled
    sampled_deeppcb = list(deeppcb_files)
    extra_deeppcb = random.choices(deeppcb_files, k=target_per_domain - len(deeppcb_files))
    
    # 2. DsPCBSD+: sample 1500 without replacement
    sampled_dspcbsd = random.sample(dspcbsd_files, target_per_domain)

    # 3. PCB-Defect V2: sample 1500 without replacement
    sampled_pcbdefect = random.sample(pcbdefect_files, target_per_domain)

    selected_files = sampled_deeppcb + extra_deeppcb + sampled_dspcbsd + sampled_pcbdefect
    print(f"[+] Total selected balanced training samples: {len(selected_files)} (1500 per domain)")

    for idx, lbl_path in enumerate(selected_files):
        img_name = lbl_path.stem + ".jpg"
        src_img_p = src_img_train / img_name
        
        # In case of oversampled duplicates, give a distinct name
        out_stem = f"bal_{idx:05d}_{lbl_path.stem}"
        dst_lbl_p = labels_train / f"{out_stem}.txt"
        dst_img_p = images_train / f"{out_stem}.jpg"

        if not dst_lbl_p.exists():
            shutil.copy2(lbl_path, dst_lbl_p)

        if not dst_img_p.exists() and src_img_p.exists():
            try:
                os.link(str(src_img_p), str(dst_img_p))
            except Exception:
                shutil.copy2(src_img_p, dst_img_p)

    # Validation set link
    src_img_val = SRC_DATASET / "images" / "val"
    src_lbl_val = SRC_DATASET / "labels" / "val"
    for p in src_lbl_val.glob("*.txt"):
        dst_p = labels_val / p.name
        if not dst_p.exists():
            shutil.copy2(p, dst_p)
    for p in src_img_val.glob("*.jpg"):
        dst_p = images_val / p.name
        if not dst_p.exists():
            try:
                os.link(str(p), str(dst_p))
            except Exception:
                shutil.copy2(p, dst_p)

    # Write data.yaml
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
    print("[+] Domain-Balanced Dataset Ready!\n")

if __name__ == "__main__":
    prepare_domain_balanced()
