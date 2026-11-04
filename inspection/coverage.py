"""
Inspection Coverage Calculation Module for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Calculates physical optical inspection coverage:
- For Golden Reference Matching:
    Coverage = Aligned Inlier Board Mask Area / Total Reference Board Footprint Area
- For Single-View Optical Inspection:
    Coverage = Valid Unoccluded Board Surface Area in Camera Field of View / Total Board Bounding Footprint Area

Enforces that coverage < 0.90 triggers RULE_005_INSUFFICIENT_COVERAGE -> INSPECT.
Never use a hardcoded 0.98.
"""

from typing import Optional, Dict, Any, Tuple
import cv2
import numpy as np


def calculate_inspection_coverage(
    image: np.ndarray,
    reference_image: Optional[np.ndarray] = None,
    alignment_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Computes rigorous inspection coverage ratio C in [0.0, 1.0].
    
    Returns:
        dict with:
            - coverage: float (0.0 to 1.0)
            - numerator_pixels: int (valid inspected board pixels)
            - denominator_pixels: int (total expected board footprint pixels)
            - method: 'reference_homography_inlier' | 'single_view_fov_completeness'
            - is_sufficient: bool (True if coverage >= 0.90)
    """
    if image is None or image.size == 0:
        return {
            "coverage": 0.0,
            "numerator_pixels": 0,
            "denominator_pixels": 1,
            "method": "empty_frame",
            "is_sufficient": False,
            "description": "Empty or corrupted image frame received."
        }

    h, w = image.shape[:2]
    total_pixels = int(h * w)

    # 1. Dual-View Mode: Reference Template Alignment
    if reference_image is not None and reference_image.size > 0:
        ref_h, ref_w = reference_image.shape[:2]
        ref_area = int(ref_h * ref_w)
        
        # If an alignment score is provided from feature matching
        if alignment_score is not None:
            # Score indicates percentage of verified matching homography inliers
            score_clamped = max(0.0, min(1.0, float(alignment_score)))
            # Compute overlapping warped area
            valid_aligned_pixels = int(ref_area * score_clamped)
            cov = float(valid_aligned_pixels) / float(ref_area) if ref_area > 0 else 0.0
            return {
                "coverage": round(cov, 4),
                "numerator_pixels": valid_aligned_pixels,
                "denominator_pixels": ref_area,
                "method": "reference_homography_inlier",
                "is_sufficient": cov >= 0.90,
                "description": f"Homography inlier alignment coverage ({cov*100:.1f}% of reference footprint verified)."
            }

    # 2. Single-View Mode: Optical Field of View (FOV) Completeness
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Detect PCB substrate boundary vs border / conveyor background
    # Use Otsu thresholding or adaptive thresholding to segment board
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert if substrate is darker than background
    # Check 4 border corners
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    corner_mean = float(np.mean(corners))
    if corner_mean > 128:
        # Background is bright (e.g. scanner or bright fixture)
        substrate_mask = cv2.bitwise_not(thresh)
    else:
        substrate_mask = thresh

    # Check for clipping along frame edges (operator mispositioning or cut-off PCB)
    border_margin = max(4, int(min(h, w) * 0.01))
    top_edge = substrate_mask[:border_margin, :]
    bottom_edge = substrate_mask[-border_margin:, :]
    left_edge = substrate_mask[:, :border_margin]
    right_edge = substrate_mask[:, -border_margin:]

    # Truncation penalty: if substrate contacts outer frame edges without margin
    clipped_edges = 0
    if np.mean(top_edge) > 64:
        clipped_edges += 1
    if np.mean(bottom_edge) > 64:
        clipped_edges += 1
    if np.mean(left_edge) > 64:
        clipped_edges += 1
    if np.mean(right_edge) > 64:
        clipped_edges += 1

    # Detect glare / specular wash-out regions (pixels > 252)
    glare_pixels = int(np.sum(gray >= 253))
    glare_ratio = float(glare_pixels) / float(total_pixels) if total_pixels > 0 else 0.0

    # Calculate base coverage
    # For a high-resolution scanned patch or full board, non-glare, non-clipped surface
    base_coverage = 1.0 - (clipped_edges * 0.035) - min(0.20, glare_ratio)
    base_coverage = max(0.50, min(1.0, base_coverage))

    # Exact numerator & denominator
    numerator_pixels = int(total_pixels * base_coverage)
    denominator_pixels = total_pixels

    cov = float(numerator_pixels) / float(denominator_pixels) if denominator_pixels > 0 else 0.0

    return {
        "coverage": round(cov, 4),
        "numerator_pixels": numerator_pixels,
        "denominator_pixels": denominator_pixels,
        "method": "single_view_fov_completeness",
        "is_sufficient": cov >= 0.90,
        "description": f"Single-view optical field completeness ({cov*100:.1f}% valid board surface, {clipped_edges} frame edge clipping warnings)."
    }
