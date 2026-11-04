"""
Image Quality and Domain Shift Assessment for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Enforces conservative pre-inference and post-inference inspection gates:
1. Blur / Defocus Detection (Laplacian variance)
2. Illumination & Glare Analysis (Mean luminance, saturation clipping ratio)
3. Dynamic Range & Contrast Verification (Luminance standard deviation)
4. Domain Shift & Color Profile Assessment (HSV saturation & color variance)
5. Alignment & Reference Validation

Distinguishes:
- PASS: Acceptable quality + validated domain + sufficient coverage + 0 critical defects.
- INSPECT: Degraded quality, unfamiliar domain, low confidence, uncertain alignment, or suspected defects.
- REJECT: High-confidence critical defects or excess major defects.
"""

from typing import Dict, Any, Optional, List
import cv2
import numpy as np


class QualityAssessmentConfig:
    # Blur detection threshold (Laplacian variance)
    BLUR_THRESHOLD: float = 80.0
    
    # Illumination thresholds (0-255 luminance)
    UNDEREXPOSURE_THRESHOLD: float = 30.0
    OVEREXPOSURE_MEAN_THRESHOLD: float = 248.0
    MAX_GLARE_RATIO_COLOR: float = 0.35  # Glare pixel ratio for natural/color images
    
    # Contrast threshold (standard deviation of luminance)
    MIN_CONTRAST_THRESHOLD: float = 20.0
    
    # Color Saturation thresholds
    MAX_MONOCHROME_SATURATION: float = 8.0     # Mean HSV saturation for monochrome domain
    COLOR_DOMAIN_MIN_SATURATION: float = 14.0   # Solder-mask or copper color presence


def assess_inspection_image(
    image: np.ndarray,
    coverage: float = 1.0,
    min_coverage_required: float = 0.90,
    has_valid_reference: bool = True,
    reference_alignment_score: Optional[float] = None,
    supported_domains: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates raw inspection image quality and domain familiarity before decision commitment.
    
    Returns structured audit metrics:
    - image_quality_status: 'acceptable' | 'blurred' | 'underexposed' | 'overexposed' | 'low_contrast'
    - domain_status: 'within_domain' | 'unfamiliar_domain' | 'out_of_domain'
    - inspection_coverage: float
    - review_required: bool
    - reason_for_inspect: explanation string or None
    """
    if supported_domains is None:
        # Default baseline model domain: DeepPCB monochrome high-contrast
        supported_domains = ["monochrome_high_contrast"]

    if image is None or image.size == 0:
        return {
            "image_quality_status": "corrupt_empty",
            "domain_status": "out_of_domain",
            "inspection_coverage": 0.0,
            "is_quality_acceptable": False,
            "is_domain_validated": False,
            "review_required": True,
            "reason_for_inspect": "Corrupt or empty image received by inspection pipeline.",
            "blur_score": 0.0,
            "mean_brightness": 0.0,
            "contrast_score": 0.0,
            "saturation_score": 0.0
        }

    # Convert to grayscale for structural analysis
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        mean_saturation = float(np.mean(sat_channel))
        max_saturation = float(np.max(sat_channel))
    else:
        gray = image.copy()
        mean_saturation = 0.0
        max_saturation = 0.0

    # 1. Blur / Sharpness check
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurred = laplacian_var < QualityAssessmentConfig.BLUR_THRESHOLD

    # 2. Illumination checks
    mean_brightness = float(np.mean(gray))
    glare_ratio = float(np.sum(gray >= 252)) / float(gray.size)
    is_underexposed = mean_brightness < QualityAssessmentConfig.UNDEREXPOSURE_THRESHOLD
    
    # In monochrome high-contrast datasets (like DeepPCB), the white substrate is ~255 with large contrast
    # Genuine specular glare washes out the whole image or saturates color PCBs
    if mean_saturation < QualityAssessmentConfig.MAX_MONOCHROME_SATURATION:
        is_overexposed = mean_brightness > QualityAssessmentConfig.OVEREXPOSURE_MEAN_THRESHOLD and contrast_std < 30.0
    else:
        is_overexposed = (mean_brightness > QualityAssessmentConfig.OVEREXPOSURE_MEAN_THRESHOLD) or (glare_ratio > QualityAssessmentConfig.MAX_GLARE_RATIO_COLOR)

    # 3. Dynamic range / Contrast check
    contrast_std = float(np.std(gray))
    is_low_contrast = contrast_std < QualityAssessmentConfig.MIN_CONTRAST_THRESHOLD

    # Determine image_quality_status
    quality_issues = []
    if is_blurred:
        quality_issues.append(f"Optical blur detected (Laplacian variance {laplacian_var:.1f} < {QualityAssessmentConfig.BLUR_THRESHOLD})")
        image_quality_status = "blurred"
    elif is_underexposed:
        quality_issues.append(f"Severe underexposure (Mean brightness {mean_brightness:.1f} < {QualityAssessmentConfig.UNDEREXPOSURE_THRESHOLD})")
        image_quality_status = "underexposed"
    elif is_overexposed:
        quality_issues.append(f"Severe overexposure / glare (Brightness {mean_brightness:.1f}, glare ratio {glare_ratio*100:.1f}%)")
        image_quality_status = "overexposed"
    elif is_low_contrast:
        quality_issues.append(f"Low dynamic contrast (StdDev {contrast_std:.1f} < {QualityAssessmentConfig.MIN_CONTRAST_THRESHOLD})")
        image_quality_status = "low_contrast"
    else:
        image_quality_status = "acceptable"

    is_quality_acceptable = len(quality_issues) == 0

    # 4. Domain Shift Analysis
    # Classify observed image domain profile
    if mean_saturation < QualityAssessmentConfig.MAX_MONOCHROME_SATURATION and max_saturation < 30.0:
        detected_domain = "monochrome_high_contrast"
    elif mean_saturation >= QualityAssessmentConfig.COLOR_DOMAIN_MIN_SATURATION or max_saturation >= 50.0:
        detected_domain = "color_pcb"
    else:
        detected_domain = "transitional_mixed"

    # Compare against supported domain scope of active vision model
    is_domain_validated = detected_domain in supported_domains
    if is_domain_validated:
        domain_status = "within_domain"
        domain_issue = None
    else:
        domain_status = "unfamiliar_domain"
        domain_issue = (
            f"Unfamiliar image domain ({detected_domain}, mean saturation {mean_saturation:.1f}): "
            f"Active vision model only validated on {supported_domains}."
        )

    # 5. Coverage and Reference Verification
    coverage_issue = None
    if coverage < min_coverage_required:
        coverage_issue = f"Insufficient inspection field coverage ({coverage*100:.1f}% < {min_coverage_required*100:.1f}%)."

    alignment_issue = None
    if reference_alignment_score is not None and reference_alignment_score < 0.60:
        alignment_issue = f"Uncertain reference alignment score ({reference_alignment_score:.2f} < 0.60)."
    elif not has_valid_reference and "reference_comparison_required" in supported_domains:
        alignment_issue = "Missing golden reference template for dual-view comparison."

    # Synthesize reasons for operator review
    reasons = []
    if quality_issues:
        reasons.extend(quality_issues)
    if domain_issue:
        reasons.append(domain_issue)
    if coverage_issue:
        reasons.append(coverage_issue)
    if alignment_issue:
        reasons.append(alignment_issue)

    review_required = len(reasons) > 0
    reason_for_inspect = " | ".join(reasons) if reasons else None

    return {
        "image_quality_status": image_quality_status,
        "domain_status": domain_status,
        "detected_domain": detected_domain,
        "inspection_coverage": float(coverage),
        "is_quality_acceptable": is_quality_acceptable,
        "is_domain_validated": is_domain_validated,
        "review_required": review_required,
        "reason_for_inspect": reason_for_inspect,
        "blur_score": round(laplacian_var, 2),
        "mean_brightness": round(mean_brightness, 2),
        "contrast_score": round(contrast_std, 2),
        "saturation_score": round(mean_saturation, 2)
    }
