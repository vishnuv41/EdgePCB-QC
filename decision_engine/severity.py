"""
Defect Severity Evaluation Module
Bharat AI-SoC Challenge 2026-27 | PS5

Maps defect attributes (class, confidence, size/area, position) to severity levels:
- critical: Functionality-breaking defect (electrical disconnection or bridging)
- major: High risk of failure, impedance mismatch, or solder bridging
- minor: Cosmetic or sub-critical surface anomaly within tolerance
"""

from typing import Dict, Any


def assess_defect_severity(defect: Dict[str, Any], config: Dict[str, Any]) -> str:
    """
    Computes severity for a single defect record.
    Defect structure expected:
    {
        "class": str,
        "confidence": float,
        "area_fraction": float (optional, 0.0 to 1.0),
        "location": str (optional)
    }
    """
    defect_cls = defect.get("class", "").lower()
    confidence = float(defect.get("confidence", 0.0))
    area_fraction = float(defect.get("area_fraction", 0.0))

    severity_map = config.get("severity_by_class", {})
    base_severity = severity_map.get(defect_cls, "major")

    large_area_thresh = config.get("area_thresholds", {}).get("large_area_fraction", 0.05)

    # Escalation rule: large anomalies or traces bridging large areas escalate
    if base_severity == "major" and area_fraction > large_area_thresh:
        return "critical"

    if base_severity == "minor" and area_fraction > large_area_thresh:
        return "major"

    # Always preserve critical classes (open circuit, short circuit)
    if defect_cls in ["open_circuit", "short_circuit"]:
        return "critical"

    return base_severity
