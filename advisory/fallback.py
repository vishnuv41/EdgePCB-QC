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
        "action": "Quarantine board. Inspect under 20x stereo microscope to confirm trace fracture. Perform flying-probe or DMM continuity test across the affected net. If permitted by IPC-7711/7721, perform micro-wire jumper rework; otherwise scrap.",
        "requires_elec": True
    },
    "short_circuit": {
        "title": "Short Circuit",
        "electrical_risk": "Unintended conductive copper bridge between adjacent traces or ground plane, causing potential power rail collapse or component damage upon power-up.",
        "action": "Immediate reject. Mark defect location with non-conductive dye. Perform high-resistance short circuit or Kelvin probe test. If bridge is superficial foil residual, evaluate precision micro-scalpel de-bridging; otherwise reject board.",
        "requires_elec": True
    },
    "mouse_bite": {
        "title": "Mouse Bite",
        "electrical_risk": "Void or notch on trace perimeter reduces effective cross-sectional conductor area, causing localized current crowding, elevated impedance, and thermal hot spots under load.",
        "action": "Measure remaining minimum trace width using calibrated optical reticle. If trace reduction exceeds 20% of design width (IPC-A-600 Class 2/3), scrap or submit for engineering review.",
        "requires_elec": False
    },
    "spur": {
        "title": "Spur",
        "electrical_risk": "Pointed copper projection protruding from trace into dielectric spacing, dangerously degrading creepage and clearance distances and promoting dielectric breakdown or arcing.",
        "action": "Measure distance to adjacent conductor. If minimum dielectric clearance is violated, verify whether mechanical trimming is allowable under IPC rework procedures.",
        "requires_elec": False
    },
    "spurious_copper": {
        "title": "Spurious Copper",
        "electrical_risk": "Isolated island of unetched copper on dielectric substrate. High-frequency signals may suffer capacitive coupling, RF radiation, or inadvertent solder bridging during wave soldering.",
        "action": "Verify isolation from all active nets. Scrape or remove loose particulate. Confirm dielectric substrate surface is undamaged.",
        "requires_elec": False
    },
    "pin_hole_missing_pad": {
        "title": "Pin Hole / Missing Pad",
        "electrical_risk": "Severe voiding in annular ring or missing solder pad prevents proper component lead wetting, leading to dry solder joints or poor mechanical bond.",
        "action": "Quarantine board. Inspect via-hole wall and annular ring under microscope. Check barrel plating thickness before soldering components.",
        "requires_elec": False
    },
    "conductor_scratch": {
        "title": "Conductor Scratch",
        "electrical_risk": "Superficial abrasion on trace surface. May expose bare copper to oxidation if solder mask is compromised.",
        "action": "Check depth of scratch. If base copper thickness is unaffected, reapply conformal coating or solder mask touch-up pen.",
        "requires_elec": False
    },
    "conductor_foreign_object": {
        "title": "Conductor Foreign Object",
        "electrical_risk": "Conductive metallic debris resting on circuitry, posing acute risk of intermittent short circuits under vibration.",
        "action": "Blow off debris using filtered ionizing air gun; re-inspect visually. If adhered, remove with ESD-safe tweezers and solvent clean.",
        "requires_elec": False
    },
    "base_material_foreign_object": {
        "title": "Base Material Foreign Object",
        "electrical_risk": "Non-conductive particulate on FR-4 laminate. Typically benign unless interfering with solder mask adhesion.",
        "action": "Clean surface with isopropyl alcohol wipe. Verify no physical laminate delamination or blistering.",
        "requires_elec": False
    }
}


def generate_fallback_advisory(payload: Dict[str, Any]) -> AdvisoryExplanation:
    """
    Generates a deterministic AdvisoryExplanation matching the Pydantic schema.
    """
    decision = payload.get("decision", "INSPECT")
    defects: List[Dict[str, Any]] = payload.get("defects", [])
    requires_elec = payload.get("requires_electrical_test", False)

    if not defects:
        if decision == "PASS":
            return AdvisoryExplanation(
                what_detected="Defect-free PCB surface; all trace geometries and clearances conform to visual threshold standards.",
                why_it_matters="Trace continuity, spacing, and pad geometry satisfy the configured quality preset with no visible anomalies.",
                operator_action="Clear board for downstream assembly / solder paste printing. Proceed with standard production flow.",
                electrical_confirmation_needed=False,
                confidence_caveat="Optical inspection confirms surface topology only. Downstream in-circuit testing (ICT) verifies electrical continuity."
            )
        else:
            return AdvisoryExplanation(
                what_detected="Insufficient optical inspection coverage or imaging uncertainty without localized defect features.",
                why_it_matters="Incomplete visual inspection coverage means portions of the conductive layers remain unverified.",
                operator_action="Re-position PCB in fixture, clean lens, ensure uniform lighting, and re-trigger visual inspection.",
                electrical_confirmation_needed=False,
                confidence_caveat="Automated optical inspection flags coverage warnings to prevent uninspected boards passing into service."
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
            "action": "Perform 20x microscopic visual examination and measure defect dimensions against IPC criteria.",
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

    if decision == "REJECT":
        action_text = f"[REJECT VERDICT] {action_text}"
    elif decision == "INSPECT":
        action_text = f"[MANUAL INSPECTION REQUIRED] {action_text}"

    return AdvisoryExplanation(
        what_detected=what_text,
        why_it_matters=why_text,
        operator_action=action_text,
        electrical_confirmation_needed=needs_elec_flag or requires_elec,
        confidence_caveat="Vision model identifies visible morphological patterns. Electrical confirmation via flying-probe or DMM is mandatory before scrapping high-value boards."
    )
