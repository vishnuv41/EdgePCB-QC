"""
Defect Severity Evaluation Module
Bharat AI-SoC Challenge 2026-27 | PS5

Maps defect attributes (class, confidence, bounding-box dimensions, area fraction, quality preset)
to severity levels:
- critical: Functionality-breaking defect (visually suspected trace break or copper bridging)
- major: Substantial defect reducing conductor width, spacing, or annular ring area
- minor: Acceptable surface variation or low-confidence anomaly flagged for operator review
"""

from typing import Dict, Any, Optional


def assess_defect_severity(
    defect: Dict[str, Any],
    config: Dict[str, Any],
    preset: str = "consumer"
) -> str:
    """
    Computes deterministic severity for a single defect record incorporating:
    - Defect class & domain physics
    - Detection confidence & quality preset thresholds
    - Bounding-box dimensions and area fraction
    - Low-confidence screening (flags for review rather than auto-major)
    """
    defect_cls = defect.get("class", "").lower()
    confidence = float(defect.get("confidence", 0.0))
    area_fraction = float(defect.get("area_fraction", 0.0))
    bbox = defect.get("bbox", [])

    # Extract bounding box dimensions if present
    bbox_w = 0.0
    bbox_h = 0.0
    if len(bbox) == 4:
        bbox_w = abs(float(bbox[2]) - float(bbox[0]))
        bbox_h = abs(float(bbox[3]) - float(bbox[1]))
        defect["bbox_width"] = round(bbox_w, 2)
        defect["bbox_height"] = round(bbox_h, 2)

    preset_cfg = config.get("presets", {}).get(preset, {})
    reject_conf = preset_cfg.get("reject_conf", 0.75)
    inspect_conf = preset_cfg.get("inspect_conf", 0.40)

    severity_map = config.get("severity_by_class", {})
    base_severity = severity_map.get(defect_cls, "major")
    large_area_thresh = config.get("area_thresholds", {}).get("large_area_fraction", 0.05)

    # 1. Low-confidence detections (< inspect_conf)
    # Must be flagged for review rather than automatically treated as confirmed major/critical
    if confidence < inspect_conf:
        defect["review_required"] = True
        defect["is_confirmed"] = False
        defect["verification_status"] = "Uncertain Anomaly (Low-Confidence Review Required)"
        return "minor"

    defect["is_confirmed"] = (confidence >= reject_conf)
    defect["review_required"] = not defect["is_confirmed"]

    # 2. Visually suspected electrical disruptions (open_circuit, short_circuit)
    if defect_cls in ["open_circuit", "short_circuit"]:
        defect["verification_status"] = "Visually Suspected (Electrical Continuity/Isolation Test Required)"
        return "critical"

    # 3. Annular ring and missing pad defects
    if defect_cls == "pin_hole_missing_pad":
        defect["verification_status"] = "Visually Confirmed Pad Anomaly"
        return "critical"

    # 4. Dimension & area escalation
    if base_severity == "major" and area_fraction > large_area_thresh:
        defect["verification_status"] = "Large Area Anomaly (Escalated to Critical)"
        return "critical"

    if base_severity == "minor" and area_fraction > large_area_thresh:
        defect["verification_status"] = "Large Minor Anomaly (Escalated to Major)"
        return "major"

    # 5. Very small anomalies (< 0.05% area fraction) with sub-reject confidence
    if area_fraction > 0.0 and area_fraction < 0.0005 and confidence < reject_conf and base_severity == "major":
        defect["verification_status"] = "Micro-Feature Anomaly (Flagged for Review)"
        return "minor"

    # 6. Default mapped severity
    defect["verification_status"] = "Visually Confirmed Surface Anomaly" if defect["is_confirmed"] else "Suspected Surface Anomaly"
    return base_severity
