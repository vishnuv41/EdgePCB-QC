"""
Golden PCB Reference Comparison Module for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Performs reference-based automated optical inspection (AOI):
- Image registration & homography alignment (ORB + RANSAC)
- Illumination normalization & difference mapping
- Morphological noise suppression
- Extraction of suspicious discrepancy contours as INSPECT records
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any, Union
import cv2
import numpy as np


def load_reference(ref_path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(ref_path))
    if img is None:
        raise FileNotFoundError(f"Reference image could not be loaded: {ref_path}")
    return img


def load_test_image(test_path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(test_path))
    if img is None:
        raise FileNotFoundError(f"Test image could not be loaded: {test_path}")
    return img


def align_images(
    test_img: np.ndarray,
    ref_img: np.ndarray,
    max_features: int = 1500
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aligns test_img to ref_img using ORB feature matching and Homography warp.
    Returns (aligned_test_img, homography_matrix).
    """
    gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) if len(test_img.shape) == 3 else test_img
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

    orb = cv2.ORB_create(max_features)
    kp1, des1 = orb.detectAndCompute(gray_test, None)
    kp2, des2 = orb.detectAndCompute(gray_ref, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        # Fallback: return unwarped test image if insufficient features
        return test_img, np.eye(3, dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use top 20% matches
    num_good = max(10, int(len(matches) * 0.25))
    good_matches = matches[:num_good]

    if len(good_matches) < 4:
        return test_img, np.eye(3, dtype=np.float32)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return test_img, np.eye(3, dtype=np.float32)

    h, w = ref_img.shape[:2]
    aligned_test = cv2.warpPerspective(test_img, H, (w, h))
    return aligned_test, H


def compare_images(
    aligned_test: np.ndarray,
    ref_img: np.ndarray,
    threshold_val: int = 40
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes absolute difference, applies lighting normalization and morphological filtering.
    Returns (difference_map, binary_defect_mask).
    """
    g_test = cv2.cvtColor(aligned_test, cv2.COLOR_BGR2GRAY) if len(aligned_test.shape) == 3 else aligned_test
    g_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

    # Subtle Gaussian blur to minimize sub-pixel edge alignment jitter
    b_test = cv2.GaussianBlur(g_test, (5, 5), 0)
    b_ref = cv2.GaussianBlur(g_ref, (5, 5), 0)

    # Absolute difference
    diff = cv2.absdiff(b_test, b_ref)

    # Thresholding
    _, thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

    # Morphological cleaning (remove single-pixel noise, consolidate defect clusters)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return diff, clean_mask


def highlight_differences(
    aligned_test: np.ndarray,
    defect_mask: np.ndarray,
    min_area: int = 30
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extracts bounding boxes of discrepancy regions and draws them on an inspection canvas.
    Returns (annotated_image, suspicious_defect_records).
    """
    annotated = aligned_test.copy()
    h, w = aligned_test.shape[:2]
    total_area = float(h * w)

    contours, _ = cv2.findContours(defect_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    suspicious_defects = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1, x2, y2 = float(x), float(y), float(x + bw), float(y + bh)
        area_frac = float(area) / total_area if total_area > 0 else 0.0

        # Draw red bounding box
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            "DIFF_ANOMALY",
            (x, max(12, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

        suspicious_defects.append({
            "class": "mouse_bite" if bw < 10 or bh < 10 else "spurious_copper",
            "confidence": 0.65,  # Moderate visual discrepancy confidence
            "bbox": [x1, y1, x2, y2],
            "area_fraction": round(area_frac, 5),
            "location": "center region"
        })

    return annotated, suspicious_defects
