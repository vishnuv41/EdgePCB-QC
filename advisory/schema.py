"""
Pydantic Schemas for Structured Quality Control Data and LLM Advisory
Bharat AI-SoC Challenge 2026-27 | PS5

Enforces schema contracts between:
- Pipeline A (Vision + Deterministic Decision Engine)
- Pipeline B (Llama 3.2 1B Instruct Advisory)
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class DefectRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    defect_class: str = Field(..., alias="class", description="Canonical defect class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    severity: str = Field(..., description="Assessed severity: critical, major, minor")
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box [x1, y1, x2, y2] or [xc, yc, w, h]")
    location: Optional[str] = Field(default="general", description="Board region / quadrant")
    area_fraction: Optional[float] = Field(default=0.0, description="Defect area relative to board")


class InspectionPayload(BaseModel):
    board_id: str = Field(..., description="Identifier of the PCB board")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")
    decision: str = Field(..., description="Deterministic decision: PASS, INSPECT, REJECT")
    rule_id: str = Field(..., description="ID of the deterministic rule that triggered the decision")
    rule_description: str = Field(..., description="Human-readable rule explanation")
    preset_applied: str = Field(default="consumer", description="Preset name: consumer, industrial, automotive")
    requires_electrical_test: bool = Field(default=False, description="Flag indicating if electrical probing is needed")
    coverage: float = Field(default=1.0, description="Visual coverage fraction (0.0 to 1.0)")
    defects: List[DefectRecord] = Field(default_factory=list, description="List of detected defects")
    summary_counts: Dict[str, int] = Field(default_factory=dict, description="Counts of critical, major, minor")


class AdvisoryExplanation(BaseModel):
    what_detected: str = Field(..., description="Clear summary of what visual anomalies were detected")
    why_it_matters: str = Field(..., description="Technical rationale, electrical implications, or failure risks")
    operator_action: str = Field(..., description="Actionable recommendation for the line operator or technician")
    electrical_confirmation_needed: bool = Field(..., description="Whether electrical multimeter/flying-probe test is required")
    confidence_caveat: str = Field(..., description="Explicit disclaimer on optical detection limitations")
    generated_text: Optional[str] = Field(default=None, description="Actual on-device Llama 3.2 generated text")
