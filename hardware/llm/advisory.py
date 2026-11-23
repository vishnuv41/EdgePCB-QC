"""
Advisory Helper & IPC Rule Mapper for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Maps deterministic quality verdicts and vision detections into structured
operator guidance payloads conforming to IPC-A-600 standards.
"""

from typing import Dict, Any, List
from .executorch_manager import ExecuTorchManager

_global_manager = None


def get_executorch_manager() -> ExecuTorchManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = ExecuTorchManager()
    return _global_manager


def build_inspection_payload(
    board_id: str,
    verdict: str,
    rule_id: str,
    rule_description: str,
    detections: List[Dict[str, Any]],
    coverage: float = 1.0,
    preset: str = "consumer"
) -> Dict[str, Any]:
    """Formats inspection attributes into standard Advisory input schema."""
    return {
        "board_id": board_id,
        "decision": verdict,
        "rule_id": rule_id,
        "rule_description": rule_description,
        "quality_preset": preset,
        "coverage_score": coverage,
        "defects": detections,
        "total_defects": len(detections)
    }
