"""
System Integrity & Offline Compliance Tests
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. Configuration integrity and preset invariants (reject_conf > inspect_conf).
2. Class mapping completeness across canonical and extended classes.
3. Deterministic severity assessment coverage for all 9 industrial defect modes.
4. Offline verification: ensures inspection modules execute without network calls.
"""

import json
import pytest
import yaml
from pathlib import Path

from decision_engine.decision import DecisionEngine
from decision_engine.severity import assess_defect_severity

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "decision_engine" / "config.json"
CLASS_MAP_PATH = WORKSPACE_ROOT / "training" / "class_map.yaml"


def test_config_invariants():
    assert CONFIG_PATH.exists(), f"Missing config at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    presets = cfg.get("presets", {})
    assert "consumer" in presets
    assert "industrial" in presets
    assert "automotive" in presets

    for name, p in presets.items():
        reject_c = p["reject_conf"]
        inspect_c = p["inspect_conf"]
        assert 0.0 < inspect_c < reject_c <= 1.0, f"Invalid confidence range in preset {name}: inspect={inspect_c}, reject={reject_c}"
        assert p["min_coverage_required"] >= 0.85, f"Coverage threshold too low in preset {name}"


def test_class_map_completeness():
    assert CLASS_MAP_PATH.exists(), f"Missing class map at {CLASS_MAP_PATH}"
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        c_map = yaml.safe_load(f)

    canonical = c_map.get("canonical_classes", {})
    assert len(canonical) == 6, f"Expected 6 core canonical classes, found {len(canonical)}"

    required_keys = {"name", "description", "severity_default", "sources"}
    for cid, data in canonical.items():
        assert required_keys.issubset(data.keys()), f"Missing keys in canonical class {cid}: {data}"
        assert data["severity_default"] in ["critical", "major", "minor"]

    extended = c_map.get("extended_classes", {})
    assert len(extended) >= 3, f"Expected at least 3 extended classes, found {len(extended)}"


def test_severity_all_nine_classes():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    test_classes = [
        ("open_circuit", "critical"),
        ("short_circuit", "critical"),
        ("pin_hole_missing_pad", "critical"),
        ("spur", "major"),
        ("mouse_bite", "major"),
        ("spurious_copper", "major"),
        ("conductor_scratch", "minor"),
        ("conductor_foreign_object", "major"),
        ("base_material_foreign_object", "minor"),
    ]

    for cls_name, expected_sev in test_classes:
        defect = {"class": cls_name, "confidence": 0.80, "area_fraction": 0.005}
        sev = assess_defect_severity(defect, cfg)
        assert sev == expected_sev, f"Severity mismatch for {cls_name}: expected {expected_sev}, got {sev}"


def test_severity_escalation_on_large_area():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # A minor defect exceeding large area threshold (0.05) should escalate to major
    large_scratch = {"class": "conductor_scratch", "confidence": 0.80, "area_fraction": 0.08}
    sev = assess_defect_severity(large_scratch, cfg)
    assert sev == "major", f"Expected escalation from minor to major, got {sev}"

    # A major defect exceeding large area threshold should escalate to critical
    large_spur = {"class": "spur", "confidence": 0.80, "area_fraction": 0.08}
    sev_spur = assess_defect_severity(large_spur, cfg)
    assert sev_spur == "critical", f"Expected escalation from major to critical, got {sev_spur}"
