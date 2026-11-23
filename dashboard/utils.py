"""
Utility functions for EdgePCB-QC Dashboard
Bharat AI-SoC Challenge 2026-27 | PS5

- Defect crop extraction with safe boundary clamping
- Tile-to-original board coordinate mapping
- File / image SHA-256 calculation
- Offline SVG branding assets
"""

import hashlib
from typing import Tuple, Dict, Any, List, Optional
from pathlib import Path
import cv2
import numpy as np


def sha256_bytes(data: bytes) -> str:
    """Computes SHA-256 hash of a byte string."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(filepath: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_defect_crop(
    image: np.ndarray,
    bbox: List[float],
    padding: int = 40
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """
    Safely extracts a magnified crop around a defect bounding box [x1, y1, x2, y2].
    
    Guarantees:
    - Never throws an IndexError on edge/corner defects
    - Correctly maps full-board bounding box to crop-local coordinates
    - Returns annotated crop with local bounding box, local bbox coordinates, and metadata
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot extract crop from empty image")

    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Ensure proper ordering and rounding
    bx1 = max(0.0, min(float(w_img), min(float(x1), float(x2))))
    by1 = max(0.0, min(float(h_img), min(float(y1), float(y2))))
    bx2 = max(0.0, min(float(w_img), max(float(x1), float(x2))))
    by2 = max(0.0, min(float(h_img), max(float(y1), float(y2))))

    # Compute crop window with padding, clamped strictly to image boundaries
    crop_x1 = int(max(0, bx1 - padding))
    crop_y1 = int(max(0, by1 - padding))
    crop_x2 = int(min(w_img, bx2 + padding))
    crop_y2 = int(min(h_img, by2 + padding))

    # Fallback to full image if bounding box was degenerated
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        crop_x1, crop_y1, crop_x2, crop_y2 = 0, 0, w_img, h_img

    crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    crop_h, crop_w = crop.shape[:2]

    # Map full-board coordinates to crop-local frame
    local_x1 = int(max(0, bx1 - crop_x1))
    local_y1 = int(max(0, by1 - crop_y1))
    local_x2 = int(min(crop_w, bx2 - crop_x1))
    local_y2 = int(min(crop_h, by2 - crop_y1))

    meta = {
        "board_bbox": [bx1, by1, bx2, by2],
        "crop_window": [crop_x1, crop_y1, crop_x2, crop_y2],
        "local_bbox": [local_x1, local_y1, local_x2, local_y2],
        "crop_width": crop_w,
        "crop_height": crop_h,
        "defect_width": max(1, int(bx2 - bx1)),
        "defect_height": max(1, int(by2 - by1)),
        "clamped_edges": {
            "top": crop_y1 == 0,
            "bottom": crop_y2 == h_img,
            "left": crop_x1 == 0,
            "right": crop_x2 == w_img
        }
    }

    return crop, [local_x1, local_y1, local_x2, local_y2], meta


def get_branding_svg() -> str:
    """Returns pure inline SVG string for EdgePCB-QC branding (100% offline)."""
    return """
    <div style="text-align: center; margin-bottom: 14px; padding: 2px 0;">
        <svg width="240" height="54" viewBox="0 0 240 54" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="cardBg" x1="0" y1="0" x2="240" y2="54" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#0f172a"/>
                    <stop offset="100%" stop-color="#1e293b"/>
                </linearGradient>
                <linearGradient id="chipGrad" x1="14" y1="13" x2="42" y2="41" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#38bdf8"/>
                    <stop offset="100%" stop-color="#0284c7"/>
                </linearGradient>
                <radialGradient id="lensGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.4"/>
                    <stop offset="100%" stop-color="#0f172a" stop-opacity="0"/>
                </radialGradient>
            </defs>
            <!-- Card Shell -->
            <rect width="240" height="54" rx="10" fill="url(#cardBg)" stroke="#334155" stroke-width="1.2"/>
            
            <!-- Circuit Board Micro-Pins -->
            <path d="M10 20H14M8 27H14M10 34H14 M42 20H46M42 27H48M42 34H46" stroke="#0284c7" stroke-width="1.4" stroke-linecap="round"/>
            <path d="M21 9V13M28 7V13M35 9V13 M21 41V45M28 41V47M35 41V45" stroke="#0284c7" stroke-width="1.4" stroke-linecap="round"/>
            
            <!-- Silicon Package / Die -->
            <rect x="14" y="13" width="28" height="28" rx="6" fill="#0b1120" stroke="url(#chipGrad)" stroke-width="1.8"/>
            
            <!-- Alignment / Fiducial Corner Markers -->
            <path d="M17 20V17H20 M36 17H39V20 M39 34V37H36 M20 37H17V34" stroke="#38bdf8" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/>
            
            <!-- Optical Lens Reticle -->
            <circle cx="28" cy="27" r="7.5" fill="url(#lensGlow)" stroke="#38bdf8" stroke-width="1.4"/>
            <circle cx="28" cy="27" r="2.8" fill="#38bdf8"/>
            
            <!-- AOI Inspection Laser Crosshairs -->
            <path d="M28 16.5V20M28 34V37.5 M17.5 27H21M35 27H38.5" stroke="#38bdf8" stroke-width="1.4" stroke-linecap="round"/>
            
            <!-- Typography -->
            <text x="56" y="25" fill="#f8fafc" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="800" font-size="15.5" letter-spacing="0.3">EdgePCB-QC</text>
            <text x="56" y="39" fill="#94a3b8" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="600" font-size="8.8" letter-spacing="1.4">INDUSTRIAL AOI SYSTEM</text>
            
            <!-- Status LED -->
            <circle cx="224" cy="27" r="3.2" fill="#10b981"/>
            <circle cx="224" cy="27" r="1.6" fill="#d1fae5"/>
        </svg>
    </div>
    """
