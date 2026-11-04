"""
Deterministic Fallback Advisory Generator for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Provides guaranteed, offline, domain-grounded explanations when:
- ExecuTorch model is downloading / pending export
- Inference exceeds real-time latency thresholds
- LLM outputs invalid JSON or fails validation
"""

from typing import Dict, Any, List
from .schema import AdvisoryExplanation

DEFECT_KNOWLEDGE_BASE = {
    "open_circuit": {
        "title": "Open Circuit",
        "electrical_risk": "Discontinuous copper trace creates a complete electrical break, preventing signal transmission and causing circuit failure.",
        "action": "Quarantine board. Inspect under optical magnification to verify trace continuity. Perform mandatory electrical continuity testing (DMM or flying-probe) across affected net. Follow approved facility engineering rework procedures if authorized; otherwise quarantine for scrap.",
        "requires_elec": True
    },
    "short_circuit": {
        "title": "Short Circuit",
        "electrical_risk": "Unintended conductive copper bridge between adjacent traces or ground plane, causing potential power rail collapse or component damage upon power-up.",
        "action": "Immediate quarantine. Perform electrical isolation test across adjacent nets to confirm short circuit. Do not attempt unguided manual scraping. Refer to approved engineering rework procedures for authorized evaluation; otherwise scrap board.",
        "requires_elec": True
    },
    "mouse_bite": {
        "title": "Mouse Bite",
        "electrical_risk": "Void or notch on trace perimeter reduces effective cross-sectional conductor area, causing localized current crowding, elevated impedance, and thermal hot spots under load.",
        "action": "Measure remaining conductor width using calibrated optical magnification. Compare against product engineering design specifications; if conductor width or cross-sectional area falls below design allowance, submit for quality engineering review.",
        "requires_elec": False
    },
    "spur": {
        "title": "Spur",
        "electrical_risk": "Pointed copper projection protruding from trace into dielectric spacing, dangerously degrading clearance distances and promoting dielectric breakdown or arcing.",
        "action": "Measure conductor-to-conductor spacing under magnification. Compare against minimum dielectric spacing rules for the board netlist. Do not perform unapproved mechanical scraping; submit for engineering review.",
        "requires_elec": False
    },
    "spurious_copper": {
        "title": "Spurious Copper",
        "electrical_risk": "Isolated island of unetched copper on dielectric substrate. High-frequency signals may suffer capacitive coupling, RF radiation, or inadvertent solder bridging during wave soldering.",
        "action": "Verify electrical isolation from adjacent nets using an insulation resistance or flying-probe test. Clean surface using filtered ionized air and an approved solvent wipe. Inspect laminate to ensure dielectric substrate is intact without conductive bridging or substrate damage.",
        "requires_elec": False
    },
    "pin_hole_missing_pad": {
        "title": "Pin Hole / Missing Pad",
        "electrical_risk": "Severe voiding in annular ring or missing solder pad prevents proper component lead wetting, leading to dry solder joints or poor mechanical bond.",
        "action": "Quarantine board for engineering review. Inspect pad and hole wall under optical magnification to determine if plating and contact area satisfy layout design rules.",
        "requires_elec": False
    },
    "conductor_scratch": {
        "title": "Conductor Scratch",
        "electrical_risk": "Superficial abrasion on trace surface. May expose bare copper to oxidation if protective mask is compromised.",
        "action": "Inspect scratch depth under magnification. If underlying copper thickness is intact according to design limits, reapply approved protective coating per standard operating procedures.",
        "requires_elec": False
    },
    "conductor_foreign_object": {
        "title": "Conductor Foreign Object",
        "electrical_risk": "Conductive metallic debris resting on circuitry, posing acute risk of intermittent short circuits under vibration.",
        "action": "Carefully remove loose particulate using a filtered ionizing air blower or ESD-safe tweezers. Clean with an approved solvent wipe and re-inspect visually. Verify electrical isolation across adjacent nets.",
        "requires_elec": False
    },
    "base_material_foreign_object": {
        "title": "Base Material Foreign Object",
        "electrical_risk": "Non-conductive particulate on FR-4 laminate. Typically benign unless interfering with surface finish or coating adhesion.",
        "action": "Clean surface using an approved lint-free solvent wipe. Inspect laminate to verify absence of blistering, delamination, or conductive contamination.",
        "requires_elec": False
    }
}


def generate_fallback_advisory(payload: Dict[str, Any]) -> AdvisoryExplanation:
    """
    Generates a deterministic AdvisoryExplanation matching the Pydantic schema.
    Provides engineering troubleshooting guidance (not certified IPC compliance sign-off).
    """
    decision = payload.get("decision", "INSPECT")
    defects: List[Dict[str, Any]] = payload.get("defects", [])
    requires_elec = payload.get("requires_electrical_test", False)

    if not defects:
        if decision == "PASS":
            return AdvisoryExplanation(
                what_detected="No defects were localized by the current vision model above the configured detection thresholds.",
                why_it_matters="No visible surface anomalies were localized across inspected regions. Electrical continuity and trace isolation remain unverified by optical inspection alone.",
                operator_action="Clear board for downstream in-circuit electrical testing (ICT / flying-probe) and assembly verification. Optical inspection evaluates top-level surface topology only.",
                electrical_confirmation_needed=False,
                confidence_caveat="Optical inspection confirms visible surface morphology only; it cannot measure electrical continuity, trace impedance, or subsurface layer integrity. Downstream contact electrical testing is required for functional verification."
            )
        else:
            return AdvisoryExplanation(
                what_detected="Insufficient optical inspection coverage, imaging quality anomaly, or domain shift without localized defect features.",
                why_it_matters="Incomplete visual inspection coverage or unvalidated imaging domain means portions of the conductive layers remain unverified.",
                operator_action="Verify board lighting, re-position PCB in fixture, confirm golden reference template alignment, and re-trigger visual inspection. Do not release board without verified inspection coverage.",
                electrical_confirmation_needed=False,
                confidence_caveat="Automated optical inspection flags quality and coverage warnings to prevent uninspected boards from passing into service."
            )

    # Summarize detected defects
    unique_classes = list({d.get("class") for d in defects if d.get("class")})
    descriptions = []
    actions = []
    risks = []
    needs_elec_flag = False

    for c in unique_classes:
        info = DEFECT_KNOWLEDGE_BASE.get(c, {
            "title": c.replace("_", " ").title(),
            "electrical_risk": "Surface anomaly with potential impact on board integrity.",
            "action": "Perform optical examination under magnification and measure defect dimensions against layout design rules.",
            "requires_elec": False
        })
        descriptions.append(f"{info['title']} ({sum(1 for d in defects if d.get('class') == c)} instance[s])")
        risks.append(info["electrical_risk"])
        actions.append(info["action"])
        if info["requires_elec"]:
            needs_elec_flag = True

    what_text = f"Detected {len(defects)} surface defect(s): {', '.join(descriptions)}."
    why_text = " ".join(risks[:2])
    action_text = " ".join(actions[:2])

    disclaimer = "[Engineering Guidance — Not Official Certification]"
    if decision == "REJECT":
        action_text = f"{disclaimer} [REJECT VERDICT] {action_text}"
    elif decision == "INSPECT":
        action_text = f"{disclaimer} [MANUAL INSPECTION REQUIRED] {action_text}"
    else:
        action_text = f"{disclaimer} {action_text}"

    return AdvisoryExplanation(
        what_detected=what_text,
        why_it_matters=why_text,
        operator_action=action_text,
        electrical_confirmation_needed=needs_elec_flag or requires_elec,
        confidence_caveat="Vision model identifies visible morphological patterns. Optical defect detection is separate from electrical testing; electrical confirmation via flying-probe or DMM is mandatory for open or short circuits before disposition."
    )
