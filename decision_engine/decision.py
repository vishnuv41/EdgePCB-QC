"""
Deterministic Quality Decision Engine
Bharat AI-SoC Challenge 2026-27 | PS5

Independent decision engine that receives vision inference outputs and deterministically
produces PASS, INSPECT, or REJECT verdicts based on configurable product presets.
LLMs NEVER make this decision.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from .severity import assess_defect_severity
from .rules import evaluate_rules


class DecisionEngine:
    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.json"
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def evaluate(self, payload: Dict[str, Any], preset: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates an inspection payload and returns a deterministic quality verdict.
        """
        board_id = payload.get("board_id", "UNKNOWN-BOARD")
        defects = payload.get("defects", [])
        coverage = float(payload.get("coverage", 1.0))

        if preset is None:
            preset = self.config.get("default_preset", "consumer")

        preset_config = self.config.get("presets", {}).get(preset)
        if not preset_config:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(self.config.get('presets', {}).keys())}")

        # Assess severity for each defect incorporating preset thresholds
        severities = []
        for defect in defects:
            sev = assess_defect_severity(defect, self.config, preset=preset)
            defect["severity"] = sev
            severities.append(sev)

        # Count severities
        sev_counts = {"critical": 0, "major": 0, "minor": 0}
        for s in severities:
            if s in sev_counts:
                sev_counts[s] += 1

        # Evaluate rules
        rule_result = evaluate_rules(defects, severities, coverage, preset_config)

        timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "board_id": board_id,
            "timestamp": timestamp,
            "decision": rule_result.decision,
            "rule_id": rule_result.rule_id,
            "rule_description": rule_result.description,
            "requires_electrical_test": rule_result.requires_electrical_test,
            "preset_applied": preset,
            "coverage": coverage,
            "defects": defects,
            "summary_counts": sev_counts,
            "config_version": self.config.get("version", "1.0"),
            "operator_override": None
        }

        return record

    def apply_override(
        self,
        decision_record: Dict[str, Any],
        operator_id: str,
        new_decision: str,
        reason: str
    ) -> Dict[str, Any]:
        """
        Records an operator confirmation or override with audit metadata.
        """
        if new_decision not in ["PASS", "INSPECT", "REJECT"]:
            raise ValueError(f"Invalid decision override: {new_decision}. Must be PASS, INSPECT, or REJECT.")

        updated_record = dict(decision_record)
        override_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operator_id": operator_id,
            "original_decision": decision_record.get("decision"),
            "override_decision": new_decision,
            "reason": reason
        }

        updated_record["operator_override"] = override_entry
        updated_record["decision"] = new_decision
        return updated_record
