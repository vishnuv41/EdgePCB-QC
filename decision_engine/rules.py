"""
Deterministic Rule Definitions & Evaluation for Quality Decisions
Bharat AI-SoC Challenge 2026-27 | PS5

Enforces prioritized rules without probabilistic or LLM dependence.
Outcomes: PASS, INSPECT, REJECT
"""

from typing import List, Dict, Any, Tuple


class RuleResult:
    def __init__(self, decision: str, rule_id: str, description: str, requires_electrical_test: bool = False):
        self.decision = decision  # PASS | INSPECT | REJECT
        self.rule_id = rule_id
        self.description = description
        self.requires_electrical_test = requires_electrical_test


def evaluate_rules(
    defects: List[Dict[str, Any]],
    severities: List[str],
    coverage: float,
    preset_config: Dict[str, Any]
) -> RuleResult:
    """
    Evaluates defects against prioritized deterministic quality rules.
    Priority order:
    1. Critical defects at or above reject confidence -> REJECT
    2. Any defect count exceeding allowed tolerances -> REJECT
    3. Critical defects with moderate confidence (uncertain) -> INSPECT
    4. Major defects needing manual verification -> INSPECT
    5. Insufficient camera coverage -> INSPECT
    6. Minor defects exceeding limit -> INSPECT
    7. No defects & sufficient coverage -> PASS
    """
    reject_conf = preset_config.get("reject_conf", 0.75)
    inspect_conf = preset_config.get("inspect_conf", 0.40)
    max_major = preset_config.get("max_major_defects_allowed", 0)
    max_minor = preset_config.get("max_minor_defects_allowed", 2)
    min_cov = preset_config.get("min_coverage_required", 0.90)

    # Check for suspected electrical discontinuity or bridging across ALL defects
    has_electrical_risk = any(d.get("class") in ["open_circuit", "short_circuit"] for d in defects)

    # Count defects by severity
    critical_defects = []
    major_defects = []
    minor_defects = []
    low_conf_defects = []

    for d, sev in zip(defects, severities):
        conf = float(d.get("confidence", 0.0))
        if conf < inspect_conf:
            low_conf_defects.append(d)
            continue

        if sev == "critical":
            critical_defects.append(d)
        elif sev == "major":
            major_defects.append(d)
        else:
            minor_defects.append(d)

    # 1. High-confidence critical defect -> REJECT
    high_conf_critical = [d for d in critical_defects if float(d.get("confidence", 0.0)) >= reject_conf]
    if high_conf_critical:
        classes = {d.get("class") for d in high_conf_critical}
        return RuleResult(
            decision="REJECT",
            rule_id="RULE_001_HIGH_CONF_CRITICAL_DEFECT",
            description=f"Rejected: Detected {len(high_conf_critical)} critical defect(s) ({', '.join(classes)}) with confidence >= {reject_conf:.2f}.",
            requires_electrical_test=has_electrical_risk
        )

    # 2. Major defects exceeding preset allowance -> REJECT
    high_conf_major = [d for d in major_defects if float(d.get("confidence", 0.0)) >= reject_conf]
    if len(high_conf_major) > max_major:
        return RuleResult(
            decision="REJECT",
            rule_id="RULE_002_EXCESS_MAJOR_DEFECTS",
            description=f"Rejected: Detected {len(high_conf_major)} high-confidence major defect(s); preset allowance is {max_major}.",
            requires_electrical_test=has_electrical_risk
        )

    # 3. Medium-confidence critical defects -> INSPECT
    if critical_defects:
        classes = {d.get("class") for d in critical_defects}
        return RuleResult(
            decision="INSPECT",
            rule_id="RULE_003_SUSPECTED_CRITICAL_DEFECT",
            description=f"Manual Inspection Required: Visually suspected critical defect(s) ({', '.join(classes)}) detected with confidence < {reject_conf:.2f}.",
            requires_electrical_test=has_electrical_risk
        )

    # 4. Any major defect within threshold -> INSPECT
    if major_defects:
        classes = {d.get("class") for d in major_defects}
        return RuleResult(
            decision="INSPECT",
            rule_id="RULE_004_MAJOR_DEFECT_FLAGGED",
            description=f"Manual Inspection Required: Detected {len(major_defects)} major defect(s) ({', '.join(classes)}).",
            requires_electrical_test=has_electrical_risk
        )

    # 5. Insufficient inspection coverage -> INSPECT
    if coverage < min_cov:
        return RuleResult(
            decision="INSPECT",
            rule_id="RULE_005_INSUFFICIENT_COVERAGE",
            description=f"Manual Inspection Required: Inspection image coverage ({coverage*100:.1f}%) is below minimum threshold ({min_cov*100:.1f}%).",
            requires_electrical_test=has_electrical_risk
        )

    # 6. Minor defects exceeding allowed limit -> INSPECT
    if len(minor_defects) > max_minor:
        return RuleResult(
            decision="INSPECT",
            rule_id="RULE_006_EXCESS_MINOR_DEFECTS",
            description=f"Manual Inspection Required: Detected {len(minor_defects)} minor defect(s); threshold is {max_minor}.",
            requires_electrical_test=has_electrical_risk
        )

    # 7. Unclassified or uncertain anomalies -> INSPECT
    if low_conf_defects:
        return RuleResult(
            decision="INSPECT",
            rule_id="RULE_007_UNCERTAIN_SURFACE_ANOMALIES",
            description=f"Manual Inspection Required: Detected {len(low_conf_defects)} low-confidence surface anomalie(s) flagged for operator review.",
            requires_electrical_test=has_electrical_risk
        )

    # 8. All checks passed -> PASS
    return RuleResult(
        decision="PASS",
        rule_id="RULE_008_ZERO_CRITICAL_DEFECTS_PASS",
        description=f"Board passed inspection with {len(minor_defects)} acceptable minor variations and {coverage*100:.1f}% coverage.",
        requires_electrical_test=False
    )
