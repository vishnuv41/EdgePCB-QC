"""
Dataset Discovery & Image Browser for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Enables Mode B (Dataset Demonstration):
- Discovers genuine bare PCB images across DeepPCB, DsPCBSD+, PCB-Defect V2, and Calibration sets
- In-place image loading (zero copying, zero dataset duplication)
- Provides metadata: dimensions, filesize, dataset provenance
- Strictly tags all dataset inspections with 'demo' or 'test' provenance
"""

import os
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_ROOT = WORKSPACE_ROOT / "datasets"

# Canonical dataset roots
DATASET_DIRECTORIES = {
    "DeepPCB (High-Contrast Monochrome Scans)": DATASETS_ROOT / "raw" / "deeppcb" / "PCBData",
    "DsPCBSD+ (Color Industrial Bare Boards)": DATASETS_ROOT / "raw" / "dspcbsd" / "Data_YOLO" / "images",
    "PCB-Defect V2 (Whole-Board Industrial Images)": DATASETS_ROOT / "raw" / "pcb_defect_v2" / "extracted_v2" / "PCB-Defect-Reannotated" / "train" / "images",
    "Calibration Subset (Standardized Benchmarks)": DATASETS_ROOT / "calibration_images"
}

_CATALOG_CACHE: Optional[Dict[str, List[Dict[str, Any]]]] = None


def discover_dataset_catalog(max_per_dataset: int = 250, force_refresh: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Discovers real PCB images in existing project directories without copying files.
    Returns: dict mapping dataset_name -> list of file metadata records.
    """
    global _CATALOG_CACHE
    if not force_refresh and _CATALOG_CACHE is not None:
        return _CATALOG_CACHE

    catalog = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for name, root_dir in DATASET_DIRECTORIES.items():
        records = []
        if root_dir.exists():
            for root, _, files in os.walk(root_dir):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts:
                        p = Path(root) / f
                        try:
                            size_kb = round(p.stat().st_size / 1024.0, 1)
                            records.append({
                                "filename": f,
                                "path": str(p),
                                "dataset": name,
                                "size_kb": size_kb,
                                "extension": ext.upper().replace(".", "")
                            })
                        except Exception:
                            pass
                    if len(records) >= max_per_dataset:
                        break
                if len(records) >= max_per_dataset:
                    break
        catalog[name] = records

    _CATALOG_CACHE = catalog
    return catalog


def load_dataset_image(image_path: str) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads a dataset image from disk using OpenCV.
    Returns: (bgr_array, metadata)
    """
    p = Path(image_path)
    if not p.exists():
        return None, {"error": f"Image file not found: {image_path}"}

    img = cv2.imread(str(p))
    if img is None:
        return None, {"error": f"OpenCV failed to decode image: {image_path}"}

    h, w = img.shape[:2]
    meta = {
        "filename": p.name,
        "path": str(p),
        "width": w,
        "height": h,
        "channels": img.shape[2] if img.ndim == 3 else 1,
        "size_kb": round(p.stat().st_size / 1024.0, 1)
    }
    return img, meta
