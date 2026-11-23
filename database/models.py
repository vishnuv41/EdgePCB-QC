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
    requires_electrical_test_for_suspected_defect = Column(Boolean, default=False)
    electrical_functionality_verified = Column(Boolean, default=False)
    electrical_test_required_for_final_signoff = Column(Boolean, default=True)
    coverage = Column(Float, default=1.0)
    clipping_warnings = Column(Integer, default=0)
    
    # Domain & Quality audit fields
    domain_status = Column(String(32), default="within_domain", nullable=False)
    image_quality_status = Column(String(32), default="acceptable", nullable=False)
    review_required = Column(Boolean, default=False, nullable=False)
    reason_for_inspect = Column(Text, nullable=True)
    
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

    # Provenance and Auditability Snapshot (Sections 6 & 8)
    inspection_source = Column(String(32), default="unknown", nullable=False)  # 'real', 'demo', 'test', 'unknown'
    config_snapshot_json = Column(Text, nullable=True)
    image_sha256 = Column(String(64), nullable=True)
    model_sha256 = Column(String(64), nullable=True)


class HumanReviewRecord(Base):
    __tablename__ = "human_reviews"

    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(String(64), unique=True, index=True, nullable=False)
    inspection_id = Column(String(64), index=True, nullable=False)
    board_id = Column(String(64), index=True, nullable=False)
    defect_index = Column(Integer, nullable=False)
    ai_class = Column(String(64), nullable=False)
    ai_confidence = Column(Float, nullable=False)
    ai_severity = Column(String(32), nullable=False)
    rule_id = Column(String(64), nullable=False)
    reviewer_decision = Column(String(32), nullable=False)  # 'confirmed_defect', 'false_call', 'uncertain'
    reason = Column(String(256), nullable=False)
    comment = Column(Text, nullable=True)
    reviewer_id = Column(String(64), default="OPERATOR-1", nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)


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
                    ("advisory_fallback_reason", "VARCHAR(256) DEFAULT NULL"),
                    ("domain_status", "VARCHAR(32) DEFAULT 'within_domain'"),
                    ("image_quality_status", "VARCHAR(32) DEFAULT 'acceptable'"),
                    ("review_required", "BOOLEAN DEFAULT 0"),
                    ("reason_for_inspect", "TEXT DEFAULT NULL"),
                    ("clipping_warnings", "INTEGER DEFAULT 0"),
                    ("requires_electrical_test_for_suspected_defect", "BOOLEAN DEFAULT 0"),
                    ("electrical_functionality_verified", "BOOLEAN DEFAULT 0"),
                    ("electrical_test_required_for_final_signoff", "BOOLEAN DEFAULT 1"),
                    ("inspection_source", "VARCHAR(32) DEFAULT 'unknown'"),
                    ("config_snapshot_json", "TEXT DEFAULT NULL"),
                    ("image_sha256", "VARCHAR(64) DEFAULT NULL"),
                    ("model_sha256", "VARCHAR(64) DEFAULT NULL")
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
        vision_model_version: str = "YOLOv8n-pcb-v1.0",
        advisory_backend: Optional[str] = None,
        advisory_model_id: Optional[str] = None,
        advisory_status: Optional[str] = None,
        advisory_fallback_reason: Optional[str] = None,
        llm_version: Optional[str] = None,
        inspection_source: str = "real",
        config_snapshot: Optional[Dict[str, Any]] = None,
        image_sha256: Optional[str] = None,
        model_sha256: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Persists a complete end-to-end inspection record to SQLite with truthful attribution.
        Returns the unique inspection_id.
        """
        inspection_id = str(uuid.uuid4())
        timings = timings or {}
        advisory_metrics = advisory_metrics or {}
        
        # Safely reconcile any kwargs passed dynamically or by older wrappers
        inspection_source = kwargs.get("inspection_source", inspection_source)
        config_snapshot = kwargs.get("config_snapshot", config_snapshot)
        image_sha256 = kwargs.get("image_sha256", image_sha256)
        model_sha256 = kwargs.get("model_sha256", model_sha256)
        
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
            requires_electrical_test_for_suspected_defect=decision_record.get("requires_electrical_test_for_suspected_defect", decision_record.get("requires_electrical_test", False)),
            electrical_functionality_verified=False,
            electrical_test_required_for_final_signoff=True,
            coverage=decision_record.get("coverage", 1.0),
            clipping_warnings=decision_record.get("clipping_warnings", 0),
            domain_status=decision_record.get("domain_status", "within_domain"),
            image_quality_status=decision_record.get("image_quality_status", "acceptable"),
            review_required=decision_record.get("review_required", decision == "INSPECT"),
            reason_for_inspect=decision_record.get("reason_for_inspect"),
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
            total_latency_ms=timings.get("total_ms", 0.0),
            inspection_source=inspection_source,
            config_snapshot_json=json.dumps(config_snapshot) if config_snapshot else None,
            image_sha256=image_sha256,
            model_sha256=model_sha256
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

    def list_inspections(self, limit: int = 50, offset: int = 0, source_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.Session() as session:
            query = session.query(InspectionRecord)
            if source_filter:
                query = query.filter(InspectionRecord.inspection_source == source_filter)
            records = (
                query
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

    def record_human_review(
        self,
        inspection_id: str,
        board_id: str,
        defect_index: int,
        ai_class: str,
        ai_confidence: float,
        ai_severity: str,
        rule_id: str,
        reviewer_decision: str,
        reason: str,
        comment: Optional[str] = None,
        reviewer_id: str = "OPERATOR-1"
    ) -> str:
        """
        Records human verification of an individual defect detection.
        reviewer_decision: 'confirmed_defect', 'false_call', or 'uncertain'
        """
        review_id = str(uuid.uuid4())
        rev = HumanReviewRecord(
            review_id=review_id,
            inspection_id=inspection_id,
            board_id=board_id,
            defect_index=defect_index,
            ai_class=ai_class,
            ai_confidence=float(ai_confidence),
            ai_severity=ai_severity,
            rule_id=rule_id,
            reviewer_decision=reviewer_decision,
            reason=reason,
            comment=comment,
            reviewer_id=reviewer_id,
            timestamp=datetime.now(timezone.utc)
        )
        with self.Session() as session:
            session.add(rev)
            session.commit()
        return review_id

    def list_human_reviews(self, inspection_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self.Session() as session:
            query = session.query(HumanReviewRecord)
            if inspection_id:
                query = query.filter_by(inspection_id=inspection_id)
            records = query.order_by(desc(HumanReviewRecord.timestamp)).limit(limit).all()
            return [
                {
                    "review_id": r.review_id,
                    "inspection_id": r.inspection_id,
                    "board_id": r.board_id,
                    "defect_index": r.defect_index,
                    "ai_class": r.ai_class,
                    "ai_confidence": r.ai_confidence,
                    "ai_severity": r.ai_severity,
                    "rule_id": r.rule_id,
                    "reviewer_decision": r.reviewer_decision,
                    "reason": r.reason,
                    "comment": r.comment,
                    "reviewer_id": r.reviewer_id,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in records
            ]

    def get_false_call_metrics(self) -> Dict[str, Any]:
        """
        Calculates:
        False-call rate = Human-confirmed false detections / Reviewed detections * 100
        Excludes 'uncertain' and unreviewed detections from the denominator.
        """
        with self.Session() as session:
            reviews = session.query(HumanReviewRecord).all()
            false_calls = [r for r in reviews if r.reviewer_decision == "false_call"]
            confirmed_defects = [r for r in reviews if r.reviewer_decision == "confirmed_defect"]
            uncertain = [r for r in reviews if r.reviewer_decision == "uncertain"]

            reviewed_count = len(false_calls) + len(confirmed_defects)
            false_call_rate = (len(false_calls) / reviewed_count * 100.0) if reviewed_count > 0 else None

            class_breakdown = {}
            for r in reviews:
                c = r.ai_class
                if c not in class_breakdown:
                    class_breakdown[c] = {"false_call": 0, "confirmed_defect": 0, "uncertain": 0, "reviewed": 0, "false_call_rate": None}
                if r.reviewer_decision == "false_call":
                    class_breakdown[c]["false_call"] += 1
                    class_breakdown[c]["reviewed"] += 1
                elif r.reviewer_decision == "confirmed_defect":
                    class_breakdown[c]["confirmed_defect"] += 1
                    class_breakdown[c]["reviewed"] += 1
                elif r.reviewer_decision == "uncertain":
                    class_breakdown[c]["uncertain"] += 1

            for c, stats in class_breakdown.items():
                if stats["reviewed"] > 0:
                    stats["false_call_rate"] = round(stats["false_call"] / stats["reviewed"] * 100.0, 2)

            return {
                "total_reviews": len(reviews),
                "reviewed_detections": reviewed_count,
                "false_detections": len(false_calls),
                "confirmed_detections": len(confirmed_defects),
                "uncertain_detections": len(uncertain),
                "false_call_rate": round(false_call_rate, 2) if false_call_rate is not None else None,
                "class_breakdown": class_breakdown
            }

    def get_defect_statistics(self, source_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates Pareto distribution and counts across inspection history.
        source_filter: if 'real', filters strictly operational inspections, excluding test/demo/unknown.
        """
        with self.Session() as session:
            query = session.query(InspectionRecord)
            if source_filter:
                query = query.filter(InspectionRecord.inspection_source == source_filter)
            records = query.all()
            total_inspections = len(records)
            decision_counts = {"PASS": 0, "INSPECT": 0, "REJECT": 0}
            class_counts = {}
            heatmap_cells = [0] * 9
            total_defects_count = 0

            for r in records:
                decision_counts[r.final_decision] = decision_counts.get(r.final_decision, 0) + 1
                try:
                    defects = json.loads(r.defects_json)
                    for d in defects:
                        cls_name = d.get("class") or d.get("defect_class") or "unknown"
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                        total_defects_count += 1
                        bbox = d.get("bbox")
                        if bbox and len(bbox) == 4:
                            cx = (bbox[0] + bbox[2]) / 2.0
                            cy = (bbox[1] + bbox[3]) / 2.0
                            col = min(2, max(0, int(cx / 427.0)))
                            row = min(2, max(0, int(cy / 240.0)))
                            heatmap_cells[row * 3 + col] += 1
                except Exception:
                    pass

            return {
                "total_inspections": total_inspections,
                "total_defects": total_defects_count,
                "decisions": decision_counts,
                "defect_frequencies": class_counts,
                "heatmap_3x3": heatmap_cells
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
            "requires_electrical_test_for_suspected_defect": record.requires_electrical_test,
            "electrical_functionality_verified": False,
            "electrical_test_required_for_final_signoff": True,
            "coverage": record.coverage,
            "clipping_warnings": getattr(record, "clipping_warnings", 0),
            "domain_status": getattr(record, "domain_status", "within_domain"),
            "image_quality_status": getattr(record, "image_quality_status", "acceptable"),
            "review_required": getattr(record, "review_required", False),
            "reason_for_inspect": getattr(record, "reason_for_inspect", None),
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
            "total_latency_ms": record.total_latency_ms,
            "inspection_source": getattr(record, "inspection_source", "unknown"),
            "config_snapshot": json.loads(record.config_snapshot_json) if getattr(record, "config_snapshot_json", None) else None,
            "image_sha256": getattr(record, "image_sha256", None),
            "model_sha256": getattr(record, "model_sha256", None)
        }
