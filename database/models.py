"""
Database Models & Persistent Inspection Logging for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

SQLite storage using SQLAlchemy ORM for local, offline audit records:
- Inspection ID, Board ID, Timestamp, Image Path
- Defects JSON, Deterministic Decision, Rule Details
- Llama Advisory Explanation
- Operator Override Audit Records
- Hardware & Inference Timings
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text, Boolean, DateTime, desc
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class InspectionRecord(Base):
    __tablename__ = "inspection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    inspection_id = Column(String(64), unique=True, index=True, nullable=False)
    board_id = Column(String(64), index=True, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    image_path = Column(String(512), nullable=False)
    
    # Decision Engine outputs
    decision = Column(String(16), index=True, nullable=False)  # PASS, INSPECT, REJECT
    rule_id = Column(String(64), nullable=False)
    rule_description = Column(Text, nullable=False)
    preset_applied = Column(String(32), default="consumer")
    requires_electrical_test = Column(Boolean, default=False)
    coverage = Column(Float, default=1.0)
    
    # Detailed JSON payloads
    defects_json = Column(Text, nullable=False)  # List of defect records
    defect_count = Column(Integer, default=0)
    summary_counts_json = Column(Text, nullable=False)  # {"critical": x, "major": y, "minor": z}
    
    # Advisory outputs & explicit backend attribution
    llm_explanation_json = Column(Text, nullable=True)  # AdvisoryExplanation dict
    advisory_backend = Column(String(64), default="deterministic_fallback", nullable=False)
    advisory_model_id = Column(String(64), default="ipc_expert_rules_engine", nullable=False)
    advisory_status = Column(String(32), default="active", nullable=False)
    advisory_fallback_reason = Column(String(256), nullable=True)
    
    # Operator Override
    operator_override_json = Column(Text, nullable=True)
    final_decision = Column(String(16), index=True, nullable=False)
    
    # Versions & Hardware benchmarks
    vision_model_version = Column(String(64), default="YOLO11n-pcb-v1.0")
    llm_version = Column(String(64), default="None (Deterministic Expert Fallback)")
    config_version = Column(String(32), default="1.0-deterministic")
    vision_latency_ms = Column(Float, default=0.0)
    advisory_latency_ms = Column(Float, default=0.0)
    total_latency_ms = Column(Float, default=0.0)


class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent / "inspections.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._migrate_schema()
        self.Session = sessionmaker(bind=self.engine)

    def _migrate_schema(self):
        """Ensures all audit and attribution columns exist in the SQLite database and reconciles history."""
        try:
            with self.engine.connect() as conn:
                result = conn.exec_driver_sql("PRAGMA table_info(inspection_records)")
                existing_cols = {row[1] for row in result.fetchall()}
                
                columns_to_add = [
                    ("advisory_backend", "VARCHAR(64) DEFAULT 'deterministic_fallback'"),
                    ("advisory_model_id", "VARCHAR(64) DEFAULT 'ipc_expert_rules_engine'"),
                    ("advisory_status", "VARCHAR(32) DEFAULT 'active'"),
                    ("advisory_fallback_reason", "VARCHAR(256) DEFAULT NULL")
                ]
                for col_name, col_type in columns_to_add:
                    if col_name not in existing_cols:
                        conn.exec_driver_sql(f"ALTER TABLE inspection_records ADD COLUMN {col_name} {col_type}")
                
                # Historical integrity reconciliation:
                # Correct past records where fallback latency (<1ms) was falsely attributed to Llama
                conn.exec_driver_sql("""
                    UPDATE inspection_records
                    SET llm_version = 'None (Deterministic Expert Fallback)',
                        advisory_backend = 'deterministic_fallback',
                        advisory_model_id = 'ipc_expert_rules_engine',
                        advisory_status = 'active',
                        advisory_fallback_reason = 'historical_sub_millisecond_fallback'
                    WHERE advisory_latency_ms < 1.0 AND (llm_version LIKE '%Llama%' OR advisory_backend IS NULL)
                """)
                conn.commit()
        except Exception as e:
            print(f"[i] Schema migration note: {e}")

    def close(self):
        """Disposes the database connection pool."""
        self.engine.dispose()

    def log_inspection(
        self,
        board_id: str,
        image_path: str,
        decision_record: Dict[str, Any],
        advisory_record: Optional[Dict[str, Any]] = None,
        advisory_metrics: Optional[Dict[str, Any]] = None,
        timings: Optional[Dict[str, float]] = None,
        vision_model_version: str = "YOLO11n-pcb-v1.0",
        advisory_backend: Optional[str] = None,
        advisory_model_id: Optional[str] = None,
        advisory_status: Optional[str] = None,
        advisory_fallback_reason: Optional[str] = None,
        llm_version: Optional[str] = None
    ) -> str:
        """
        Persists a complete end-to-end inspection record to SQLite with truthful attribution.
        Returns the unique inspection_id.
        """
        inspection_id = str(uuid.uuid4())
        timings = timings or {}
        advisory_metrics = advisory_metrics or {}
        
        defects = decision_record.get("defects", [])
        decision = decision_record.get("decision", "INSPECT")
        summary_counts = decision_record.get("summary_counts", {})
        
        # Resolve explicit backend attribution
        backend = (
            advisory_backend or
            advisory_metrics.get("backend") or
            "deterministic_fallback"
        )
        model_id = (
            advisory_model_id or
            advisory_metrics.get("model_id") or
            ("meta-llama/Llama-3.2-1B-Instruct" if backend == "executorch_llama_3.2_1b" else "ipc_expert_rules_engine")
        )
        status = (
            advisory_status or
            advisory_metrics.get("status") or
            ("executed" if backend == "executorch_llama_3.2_1b" else "active")
        )
        fallback_reason = (
            advisory_fallback_reason or
            advisory_metrics.get("fallback_reason")
        )

        # Strictly enforce Llama attribution only when ExecuTorch actually generated/executed
        if backend == "executorch_llama_3.2_1b" and status == "executed":
            actual_llm_version = llm_version or advisory_metrics.get("model_version", "Llama-3.2-1B-Instruct")
        else:
            actual_llm_version = "None (Deterministic Expert Fallback)"

        record = InspectionRecord(
            inspection_id=inspection_id,
            board_id=board_id,
            timestamp=datetime.now(timezone.utc),
            image_path=str(image_path),
            decision=decision,
            final_decision=decision,
            rule_id=decision_record.get("rule_id", "UNKNOWN"),
            rule_description=decision_record.get("rule_description", ""),
            preset_applied=decision_record.get("preset_applied", "consumer"),
            requires_electrical_test=decision_record.get("requires_electrical_test", False),
            coverage=decision_record.get("coverage", 1.0),
            defects_json=json.dumps(defects),
            defect_count=len(defects),
            summary_counts_json=json.dumps(summary_counts),
            llm_explanation_json=json.dumps(advisory_record) if advisory_record else None,
            advisory_backend=backend,
            advisory_model_id=model_id,
            advisory_status=status,
            advisory_fallback_reason=fallback_reason,
            operator_override_json=None,
            vision_model_version=vision_model_version,
            llm_version=actual_llm_version,
            config_version=decision_record.get("config_version", "1.0-deterministic"),
            vision_latency_ms=timings.get("vision_ms", 0.0),
            advisory_latency_ms=timings.get("advisory_ms", 0.0),
            total_latency_ms=timings.get("total_ms", 0.0)
        )

        with self.Session() as session:
            session.add(record)
            session.commit()

        return inspection_id

    def get_inspection(self, inspection_id: str) -> Optional[Dict[str, Any]]:
        with self.Session() as session:
            record = session.query(InspectionRecord).filter_by(inspection_id=inspection_id).first()
            if not record:
                return None
            return self._record_to_dict(record)

    def list_inspections(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self.Session() as session:
            records = (
                session.query(InspectionRecord)
                .order_by(desc(InspectionRecord.timestamp))
                .offset(offset)
                .limit(limit)
                .all()
            )
            return [self._record_to_dict(r) for r in records]

    def record_override(
        self,
        inspection_id: str,
        operator_id: str,
        new_decision: str,
        reason: str
    ) -> bool:
        with self.Session() as session:
            record = session.query(InspectionRecord).filter_by(inspection_id=inspection_id).first()
            if not record:
                return False

            override_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "operator_id": operator_id,
                "original_decision": record.decision,
                "override_decision": new_decision,
                "reason": reason
            }
            record.operator_override_json = json.dumps(override_data)
            record.final_decision = new_decision
            session.commit()
            return True

    def get_defect_statistics(self) -> Dict[str, Any]:
        """Calculates Pareto distribution and counts across inspection history."""
        with self.Session() as session:
            records = session.query(InspectionRecord).all()
            total_inspections = len(records)
            decision_counts = {"PASS": 0, "INSPECT": 0, "REJECT": 0}
            class_counts = {}

            for r in records:
                decision_counts[r.final_decision] = decision_counts.get(r.final_decision, 0) + 1
                try:
                    defects = json.loads(r.defects_json)
                    for d in defects:
                        cls_name = d.get("class", "unknown")
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                except Exception:
                    pass

            return {
                "total_inspections": total_inspections,
                "decisions": decision_counts,
                "defect_frequencies": class_counts
            }

    @staticmethod
    def _record_to_dict(record: InspectionRecord) -> Dict[str, Any]:
        return {
            "inspection_id": record.inspection_id,
            "board_id": record.board_id,
            "timestamp": record.timestamp.isoformat(),
            "image_path": record.image_path,
            "decision": record.decision,
            "final_decision": record.final_decision,
            "rule_id": record.rule_id,
            "rule_description": record.rule_description,
            "preset_applied": record.preset_applied,
            "requires_electrical_test": record.requires_electrical_test,
            "coverage": record.coverage,
            "defects": json.loads(record.defects_json) if record.defects_json else [],
            "defect_count": record.defect_count,
            "summary_counts": json.loads(record.summary_counts_json) if record.summary_counts_json else {},
            "llm_explanation": json.loads(record.llm_explanation_json) if record.llm_explanation_json else None,
            "advisory_backend": getattr(record, "advisory_backend", "deterministic_fallback"),
            "advisory_model_id": getattr(record, "advisory_model_id", "ipc_expert_rules_engine"),
            "advisory_status": getattr(record, "advisory_status", "active"),
            "advisory_fallback_reason": getattr(record, "advisory_fallback_reason", None),
            "operator_override": json.loads(record.operator_override_json) if record.operator_override_json else None,
            "vision_model_version": record.vision_model_version,
            "llm_version": record.llm_version,
            "config_version": record.config_version,
            "vision_latency_ms": record.vision_latency_ms,
            "advisory_latency_ms": record.advisory_latency_ms,
            "total_latency_ms": record.total_latency_ms
        }
