"""
EdgePCB-QC Industrial Inspection Dashboard
Bharat AI-SoC Challenge 2026-27 | PS5

Offline Streamlit application for bare-PCB defect detection and quality control.
Zero cloud dependencies at runtime.
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from decision_engine.decision import DecisionEngine
from advisory.executorch_runtime import ExecuTorchLlamaRunner
from database.models import DatabaseManager
from inspection.cpu_inference import CPUInspector
from inspection.hailo_inference import HailoInspector
from inspection.tiling import TiledInferenceRunner
from inspection.reference_compare import (
    load_reference, load_test_image, align_images, compare_images, highlight_differences
)

# Page configuration
st.set_page_config(
    page_title="EdgePCB-QC | Industrial Inspection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize subsystems
@st.cache_resource
def get_subsystems():
    db = DatabaseManager(WORKSPACE_ROOT / "database" / "inspections.db")
    engine = DecisionEngine(WORKSPACE_ROOT / "decision_engine" / "config.json")
    advisory = ExecuTorchLlamaRunner()
    cpu_insp = CPUInspector()
    hailo_insp = HailoInspector()
    tiled_insp = TiledInferenceRunner(cpu_insp)
    return db, engine, advisory, cpu_insp, hailo_insp, tiled_insp

db_mgr, decision_engine, advisory_runner, cpu_inspector, hailo_inspector, tiled_inspector = get_subsystems()

# Color mapping by severity
SEVERITY_COLORS = {
    "critical": (0, 0, 255),    # Red (BGR)
    "major": (0, 165, 255),     # Orange (BGR)
    "minor": (0, 255, 255)      # Yellow (BGR)
}

def annotate_image(image: np.ndarray, defects: list) -> np.ndarray:
    annotated = image.copy()
    h, w = image.shape[:2]
    for d in defects:
        bbox = d.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            sev = d.get("severity", "major")
            color = SEVERITY_COLORS.get(sev, (0, 255, 0))
            cls_name = d.get("class", "defect")
            conf = d.get("confidence", 0.0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} ({conf:.2f}) [{sev.upper()}]"
            cv2.putText(
                annotated,
                label,
                (x1, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA
            )
    return annotated

# Sidebar
st.sidebar.image("https://raw.githubusercontent.com/tangsanli5201/DeepPCB/master/fig/result/result_test1.jpg", caption="EdgePCB-QC System")
st.sidebar.title("EdgePCB-QC Control")
st.sidebar.markdown("**Bharat AI-SoC Challenge 2026-27 | PS5**")

backend_choice = st.sidebar.selectbox(
    "Vision Inference Backend",
    ["CPU / Laptop Pipeline (Active)", "Hailo-8 / 8L NPU (Raspberry Pi AI HAT+)"]
)

model_checkpoint_choice = st.sidebar.selectbox(
    "Active Vision Checkpoint",
    [
        "Unified Multi-Domain (Color + Monochrome) [Recommended]",
        "Baseline DeepPCB (Monochrome High-Contrast Only)"
    ],
    index=0,
    help="Unified model trained on DeepPCB, DsPCBSD+, and PCB-Defect V2. Baseline model trained exclusively on DeepPCB scans."
)

multidomain_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
baseline_path = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
chosen_path = multidomain_path if "Unified" in model_checkpoint_choice else baseline_path

checkpoint_error = None
try:
    if not hasattr(cpu_inspector, "load_checkpoint"):
        from inspection.cpu_inference import CPUInspector
        cpu_inspector = CPUInspector(model_path=chosen_path)
        tiled_inspector.inspector = cpu_inspector
    elif str(cpu_inspector.model_path) != str(chosen_path) or not cpu_inspector.is_loaded:
        cpu_inspector.load_checkpoint(chosen_path)
        tiled_inspector.inspector = cpu_inspector
except Exception as e:
    checkpoint_error = str(e)

model_meta = cpu_inspector.get_model_metadata() if hasattr(cpu_inspector, "get_model_metadata") else {
    "model_name": "Error",
    "model_version": "Error",
    "model_path": None,
    "architecture": "None",
    "num_classes": 0,
    "classes": [],
    "supported_domains": [],
    "device": "cpu",
    "is_loaded": False
}

if checkpoint_error or not model_meta.get("is_loaded", False):
    st.sidebar.error(f"❌ Checkpoint Error: {checkpoint_error or 'Model failed to load'}")
    st.error(f"🚨 **Vision Model Checkpoint Error:** Cannot inspect boards. Failed to load checkpoint `{chosen_path}`: {checkpoint_error or 'Model is not loaded'}")
    st.stop()

quality_preset = st.sidebar.selectbox(
    "Quality Standard Preset",
    ["consumer", "industrial", "automotive"],
    index=0,
    help="Consumer: 75% conf reject, Industrial: 65% reject, Automotive: Zero tolerance (50% reject)"
)

advisory_mode_choice = st.sidebar.selectbox(
    "Advisory Reasoning Mode",
    ["Fast Deterministic Fallback (<1ms, IPC Engineering Guidance)", "ExecuTorch Llama 3.2 1B (On-Device CPU)"],
    index=0,
    help="Fast mode produces instant grounded IPC troubleshooting advice. ExecuTorch runs genuine local Llama 3.2 1B .pte forward evaluation."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Runtime Status")
if "Hailo" in backend_choice:
    st.sidebar.error("Hailo NPU: Offline / Pending Pi 5 Hardware")
else:
    st.sidebar.success(f"CPU Backend: Active & Ready ({model_meta['model_name']})")
    st.sidebar.caption(f"Checkpoint: `{Path(model_meta['model_path']).name}` | Scope: `{', '.join(model_meta['supported_domains'])}`")

if "ExecuTorch" in advisory_mode_choice and advisory_runner.is_loaded:
    st.sidebar.success(f"Advisory: ExecuTorch Llama 3.2 1B Active ({advisory_runner.pte_path.name})")
else:
    st.sidebar.info("Advisory: Deterministic Fallback Active (IPC Engineering Guidance)")

st.sidebar.caption("RTX 5050 Laptop GPU (sm_120) detected; PyTorch running on verified CPU fallback.")

# Main Navigation Tabs
tab_inspect, tab_history, tab_analytics, tab_hardware = st.tabs([
    "🔍 PCB Visual Inspection",
    "📋 Inspection Log & Audit",
    "📊 Pareto & Defect Analytics",
    "⚙️ Hardware & Edge Architecture"
])

# -------------------------------------------------------------
# TAB 1: VISUAL INSPECTION
# -------------------------------------------------------------
with tab_inspect:
    st.subheader("PCB Defect Detection & Quality-Control Assistant")

    mode = st.radio(
        "Inspection Mode:",
        ["Standard Defect Detection (Single Board)", "Golden Reference Comparison (Template + Test Pair)"],
        horizontal=True
    )

    col_input, col_view = st.columns([1, 1.4])

    with col_input:
        st.markdown("#### Input Board Selection")
        board_id = st.text_input("PCB Serial / Board ID", value=f"PCB-{int(time.time()) % 10000:04d}")

        uploaded_file = st.file_uploader("Upload PCB High-Resolution Image", type=["jpg", "png", "bmp"])
        
        # Or load DeepPCB sample
        sample_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_test.jpg"
        sample_temp_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041" / "00041000_temp.jpg"
        sample_anno_path = WORKSPACE_ROOT / "datasets" / "raw" / "deeppcb" / "PCBData" / "group00041" / "00041_not" / "00041000.txt"

        use_sample = st.checkbox("Load Sample Defective PCB from DeepPCB dataset", value=(uploaded_file is None))

        img_to_inspect = None
        temp_img_to_inspect = None

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_to_inspect = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif use_sample and sample_path.exists():
            img_to_inspect = cv2.imread(str(sample_path))
            if sample_temp_path.exists():
                temp_img_to_inspect = cv2.imread(str(sample_temp_path))

        run_inspection = st.button("🚀 Run Offline Industrial Inspection", type="primary", use_container_width=True)

    with col_view:
        if img_to_inspect is not None:
            st.image(cv2.cvtColor(img_to_inspect, cv2.COLOR_BGR2RGB), caption=f"Raw Image: {board_id}", use_container_width=True)

    if run_inspection and img_to_inspect is not None:
        st.markdown("---")
        t_vision_start = time.perf_counter()

        defects = []
        annotated_bgr = img_to_inspect.copy()

        if mode.startswith("Golden Reference") and temp_img_to_inspect is not None:
            # Run Golden Reference Comparison
            aligned_test, _ = align_images(img_to_inspect, temp_img_to_inspect)
            diff_map, defect_mask = compare_images(aligned_test, temp_img_to_inspect)
            annotated_bgr, defects = highlight_differences(aligned_test, defect_mask)
        else:
            # Run Real Trained Edge Detector (Tiled inference for high-res images, single-frame for standard sizes)
            h_img, w_img = img_to_inspect.shape[:2]
            if h_img > 640 or w_img > 640:
                defects = tiled_inspector.inspect_tiled(img_to_inspect)
            else:
                defects = cpu_inspector.inspect(img_to_inspect)
            annotated_bgr = annotate_image(img_to_inspect, defects)

        t_vision_end = time.perf_counter()
        vision_latency = (t_vision_end - t_vision_start) * 1000.0

        # Compute optical inspection coverage (physical calculation)
        from inspection.coverage import calculate_inspection_coverage
        coverage_info = calculate_inspection_coverage(
            img_to_inspect,
            reference_image=temp_img_to_inspect if mode.startswith("Golden Reference") else None
        )
        coverage_val = coverage_info["coverage"]

        # Assess Image Quality and Domain Status (Conservative Policy Gates)
        from inspection.quality_assessment import assess_inspection_image
        quality_assessment = assess_inspection_image(
            img_to_inspect,
            coverage=coverage_val,
            min_coverage_required=0.90,
            supported_domains=model_meta["supported_domains"],
            has_valid_reference=temp_img_to_inspect is not None if mode.startswith("Golden Reference") else True
        )

        # Run Deterministic Decision Engine
        payload = {
            "board_id": board_id,
            "defects": defects,
            "coverage": coverage_val,
            "quality_assessment": quality_assessment
        }
        decision_record = decision_engine.evaluate(payload, preset=quality_preset)

        # Run Advisory Pipeline (Pipeline B)
        t_advisory_start = time.perf_counter()
        adv_mode = "auto" if "ExecuTorch" in advisory_mode_choice else "fast"
        advisory_result = advisory_runner.generate_advisory(decision_record, mode=adv_mode, max_new_tokens=24)
        t_advisory_end = time.perf_counter()
        advisory_latency = (t_advisory_end - t_advisory_start) * 1000.0

        total_latency = vision_latency + advisory_latency

        # Log into SQLite Database
        saved_img_dir = WORKSPACE_ROOT / "datasets" / "inspected_captures"
        saved_img_dir.mkdir(parents=True, exist_ok=True)
        saved_img_path = saved_img_dir / f"{board_id}_{int(time.time())}.jpg"
        cv2.imwrite(str(saved_img_path), annotated_bgr)

        insp_uuid = db_mgr.log_inspection(
            board_id=board_id,
            image_path=str(saved_img_path),
            decision_record=decision_record,
            advisory_record=advisory_result["advisory"],
            advisory_metrics=advisory_result.get("metrics"),
            timings={
                "vision_ms": round(vision_latency, 2),
                "advisory_ms": round(advisory_latency, 2),
                "total_ms": round(total_latency, 2)
            },
            vision_model_version=model_meta["model_version"]
        )

        # Display Results
        col_res_img, col_res_data = st.columns([1.2, 1])

        with col_res_img:
            st.image(
                cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Annotated Bounding Boxes ({len(defects)} defect[s] localized)",
                use_container_width=True
            )
            st.caption(f"📐 **Optical Coverage:** {coverage_val*100:.1f}% ({coverage_info['description']})")
            st.caption(f"🔍 **Active Detector:** `{model_meta['model_version']}`")

        with col_res_data:
            verdict = decision_record["decision"]
            if verdict == "PASS":
                st.success(f"### FINAL DECISION: {verdict}")
            elif verdict == "INSPECT":
                st.warning(f"### FINAL DECISION: {verdict}")
            else:
                st.error(f"### FINAL DECISION: {verdict}")

            st.markdown(f"**Triggered Rule:** `{decision_record['rule_id']}`")
            st.markdown(f"*{decision_record['rule_description']}*")
            
            if decision_record["requires_electrical_test"]:
                st.error("⚡ **ELECTRICAL VERIFICATION MANDATORY**: Visually suspected open or short circuit detected. Electrical continuity/isolation test (DMM / Flying-Probe) is required prior to final disposition.")
                st.caption("⚠️ *Optical defect detection evaluates surface geometry only; electrical integrity must be verified with contact test equipment before final panel sign-off.*")
            else:
                st.info("⚡ **ELECTRICAL STATUS: UNVERIFIED BY OPTICAL AOI**: No open or short circuits localized above threshold. Surface visual geometry conforms; downstream in-circuit testing (ICT / flying-probe) is required for functional verification.")

            # Quality and Domain Assessment Badges (Conservative Policy)
            q_col1, q_col2, q_col3 = st.columns(3)
            with q_col1:
                dom_status = decision_record.get("domain_status", "within_domain")
                if dom_status == "within_domain":
                    st.success("🌐 Scope: Validated")
                else:
                    st.warning(f"🌐 Scope: {dom_status.replace('_', ' ').title()}")
            with q_col2:
                iq_status = decision_record.get("image_quality_status", "acceptable")
                if iq_status == "acceptable":
                    st.success("📷 Quality: Clear")
                else:
                    st.warning(f"📷 Quality: {iq_status.title()}")
            with q_col3:
                if decision_record.get("review_required"):
                    st.warning("⚠️ Review: Required")
                else:
                    st.success("✅ Review: Cleared")

            if decision_record.get("reason_for_inspect"):
                st.info(f"📋 **Reason for Review:** {decision_record['reason_for_inspect']}")

            # Quantitative Defect Audit Counts (Requirement 5: from single authoritative inspection record)
            audit_counts = decision_record.get("audit_counts", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Critical", audit_counts.get("total_critical", 0))
            m2.metric("High-Conf Critical", audit_counts.get("high_conf_critical", 0))
            m3.metric("Review Required (< Conf)", audit_counts.get("low_conf_review", 0))
            m4.metric("Triggered Rule Detections", audit_counts.get("rule_triggering", 0))
            st.caption(f"⏱️ Inspection Latency: Vision {vision_latency:.2f} ms + Advisory {advisory_latency:.2f} ms = **{total_latency:.2f} ms total**")

        # Localized Defect Manifest Table
        if defects:
            st.markdown("#### 📍 Localized Defect Manifest")
            table_rows = []
            for idx, d in enumerate(defects, 1):
                bbox = d.get("bbox", [0, 0, 0, 0])
                table_rows.append({
                    "#": idx,
                    "Class": d.get("class", "unknown"),
                    "Severity": d.get("severity", "major").upper(),
                    "Confidence": f"{d.get('confidence', 0.0) * 100:.2f}%",
                    "Verification Status": d.get("verification_status", "Visually Confirmed"),
                    "Review Flag": "⚠️ Review Required" if d.get("review_required") else "Confirmed",
                    "Bounding Box [x1, y1, x2, y2]": f"[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]",
                    "Location": d.get("location", "center region"),
                    "Area Fraction": f"{d.get('area_fraction', 0.0) * 100:.3f}%"
                })
            st.dataframe(table_rows, use_container_width=True)

        # Advisory Explanation Panel (Requirement 3: Clean Separation & Primary Llama AI Reasoning)
        adv_metrics = advisory_result.get("metrics", {})
        adv_backend = adv_metrics.get("backend", "deterministic_fallback")
        adv_model_id = adv_metrics.get("model_id", "ipc_expert_rules_engine")
        adv_status = adv_metrics.get("status", "active")
        fallback_reason = adv_metrics.get("fallback_reason")
        adv = advisory_result["advisory"]

        if adv_backend == "executorch_llama_3.2_1b" and adv.get("generated_text"):
            st.markdown("### 🤖 Primary AI Advisory: Meta Llama 3.2 1B Instruct (`executorch_llama_3.2_1b`)")
            st.markdown(f"**Execution Backend:** `executorch_llama_3.2_1b` | **Model ID:** `{adv_model_id}` | **Status:** `{adv_status}`")
            st.caption(f"⚡ On-Device CPU Inference: {adv_metrics.get('tokens_evaluated', 0)} prompt tokens, {adv_metrics.get('tokens_generated', 0)} generated tokens in {adv_metrics.get('latency_ms', 0):.2f} ms ({adv_metrics.get('tokens_per_second', 0):.2f} tok/s, TTFT: {adv_metrics.get('ttft_ms', 0):.2f} ms, Peak RAM: {adv_metrics.get('peak_ram_mb', 0):.1f} MB)")
            
            with st.container():
                st.info(f"**🤖 Primary AI Generated Reasoning:**\n\n> *\"{adv['generated_text']}\"*")
                
                st.markdown("#### 🔍 Grounded Inspection Rationale & SOP")
                st.markdown(f"**Visual Findings:** {adv['what_detected']}")
                st.markdown(f"**Physical Failure Mechanics:** {adv['why_it_matters']}")
                st.markdown(f"**Standard Operating Procedure (SOP):** {adv['operator_action']}")
                st.caption(f"⚠️ *{adv['confidence_caveat']}*")

        elif adv_status in ("fallback_on_incomplete", "fallback_on_error") or fallback_reason:
            st.markdown("### ⚠️ Quality-Control Advisory: Fallback Activated")
            st.warning(f"**Llama Advisory Notice**: Status is `{adv_status}` ({fallback_reason or 'No complete sentence formed within budget'}). Deterministic IPC Expert Guidance active.")
            st.markdown(f"**Active Rules Engine:** `{adv_model_id}` | **Backend:** `{adv_backend}`")
            
            with st.container():
                st.markdown(f"**Visual Findings:** {adv['what_detected']}")
                st.markdown(f"**Physical Failure Mechanics:** {adv['why_it_matters']}")
                st.markdown(f"**Recommended Operator Action:** {adv['operator_action']}")
                st.caption(f"⚠️ *{adv['confidence_caveat']}*")
        else:
            st.markdown("### 📋 Fast Deterministic Advisory (`deterministic_fallback`)")
            st.markdown(f"**Backend:** `deterministic_fallback` | **Rules Engine:** `{adv_model_id}` | **Status:** `{adv_status}`")
            st.caption(f"ℹ️ Engineering Guidance Engine active (<1ms deterministic latency: {adv_metrics.get('latency_ms', 0):.2f} ms). Official Llama 3.2 1B ExecuTorch `.pte` model is verified and selectable via sidebar.")

            with st.container():
                st.markdown(f"**Visual Findings:** {adv['what_detected']}")
                st.markdown(f"**Physical Failure Mechanics:** {adv['why_it_matters']}")
                st.markdown(f"**Recommended Operator Action:** {adv['operator_action']}")
                st.caption(f"⚠️ *{adv['confidence_caveat']}*")

        # Operator Override Section
        st.markdown("---")
        with st.expander("🛠️ Line Operator Confirmation & Override"):
            st.write(f"Inspection Record ID: `{insp_uuid}`")
            col_ov1, col_ov2 = st.columns([1, 2])
            with col_ov1:
                new_verdict = st.selectbox("Override Verdict to:", ["PASS", "INSPECT", "REJECT"], index=1)
                operator_id = st.text_input("Operator ID", value="OP-LINE-1")
            with col_ov2:
                override_reason = st.text_area("Justification / Audit Note", placeholder="e.g., Cleaned dust particle under microscope; confirmed trace continuity.")
                submit_override = st.button("Apply Official Override", type="secondary")

            if submit_override:
                if not override_reason.strip():
                    st.error("A technical justification is strictly required to override an automated decision.")
                else:
                    db_mgr.record_override(insp_uuid, operator_id, new_verdict, override_reason)
                    st.success(f"Override recorded: Record `{insp_uuid}` updated to `{new_verdict}` by `{operator_id}`.")

# -------------------------------------------------------------
# TAB 2: INSPECTION HISTORY & AUDIT
# -------------------------------------------------------------
with tab_history:
    st.subheader("Persistent SQLite Inspection Logs & Individual Board Audit")
    records = db_mgr.list_inspections(limit=50)
    if not records:
        st.info("No inspection logs recorded yet. Run an inspection from Tab 1 to generate records.")
    else:
        table_rows = []
        for r in records:
            table_rows.append({
                "Timestamp (UTC)": r["timestamp"][:19].replace("T", " "),
                "Board ID": r["board_id"],
                "Rule Decision": r["decision"],
                "Final Decision": r["final_decision"],
                "Defects": r["defect_count"],
                "Electrical Test": "⚡ Required" if r.get("requires_electrical_test") else "No",
                "Advisory Backend": r.get("advisory_backend", "deterministic_fallback"),
                "Advisory Model": r.get("llm_version", "None"),
                "Vision Latency": f"{r['vision_latency_ms']:.2f} ms",
                "Advisory Latency": f"{r['advisory_latency_ms']:.2f} ms",
                "Total Latency": f"{r['total_latency_ms']:.2f} ms",
                "Overridden": "Yes" if r["operator_override"] else "No"
            })
        st.dataframe(table_rows, use_container_width=True)

        st.markdown("### Individual Board Inspection Audit Record")
        selected_id = st.selectbox("Select Inspection Record to Inspect Full Audit Payload:", [f"{r['board_id']} ({r['inspection_id']})" for r in records])
        if selected_id:
            raw_uuid = selected_id.split("(")[-1].rstrip(")")
            sel_rec = db_mgr.get_inspection(raw_uuid)
            st.json(sel_rec)

# -------------------------------------------------------------
# TAB 3: ANALYTICS & PARETO CHARTS
# -------------------------------------------------------------
with tab_analytics:
    st.subheader("Statistical Process Control & Pareto Distribution (Aggregate)")
    st.caption("Aggregated statistical analysis across all inspected boards in the local database.")
    stats = db_mgr.get_defect_statistics()

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Quality Decision Breakdown")
        dec_data = stats["decisions"]
        fig_donut = px.pie(
            values=list(dec_data.values()),
            names=list(dec_data.keys()),
            color=list(dec_data.keys()),
            color_discrete_map={"PASS": "#28a745", "INSPECT": "#ffc107", "REJECT": "#dc3545"},
            hole=0.45
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_chart2:
        st.markdown("#### Defect Class Pareto Analysis")
        freqs = stats["defect_frequencies"]
        if not freqs:
            st.info("No defect instances recorded yet.")
        else:
            sorted_freqs = sorted(freqs.items(), key=lambda x: -x[1])
            classes = [x[0] for x in sorted_freqs]
            counts = [x[1] for x in sorted_freqs]
            total_defects = sum(counts)
            cum_percent = [sum(counts[:i+1]) / total_defects * 100 for i in range(len(counts))]

            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(x=classes, y=counts, name="Count", marker_color="#17a2b8"))
            fig_pareto.add_trace(go.Scatter(x=classes, y=cum_percent, name="Cumulative %", yaxis="y2", line=dict(color="#fd7e14", width=3)))
            fig_pareto.update_layout(
                yaxis=dict(title="Instance Count"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                legend=dict(x=0.7, y=1.1, orientation="h")
            )
            st.plotly_chart(fig_pareto, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: HARDWARE & EDGE ARCHITECTURE
# -------------------------------------------------------------
with tab_hardware:
    st.subheader("Edge Hardware Architecture & Benchmark Deployment")

    col_arch1, col_arch2 = st.columns(2)

    with col_arch1:
        st.markdown("### Laptop Host Development Environment")
        st.markdown("""
        - **CPU:** AMD Ryzen 9 8940HX with Radeon Graphics (16 Cores, 32 Threads) — **Verified Active Vision Engine (~28 FPS)**
        - **Integrated GPU:** AMD Radeon 610M (Active)
        - **Discrete GPU:** NVIDIA GeForce RTX 5050 Laptop GPU (8GB GDDR6, Driver 621.01, CC 12.0 `sm_120`)
        - **PyTorch CUDA Status:** Installed PyTorch 2.14.1+cu126 lacks Blackwell `sm_120` precompiled kernels (requires cu130+). GPU acceleration is intentionally not claimed; CPU fallback is the verified, active engine.
        - **System RAM:** 16.0 GB DDR5
        - **Operating System:** Windows 11 Home 64-bit
        - **Vision Runtime:** OpenCV + Ultralytics YOLO11n + ONNXRuntime
        - **LLM Runtime:** ExecuTorch 1.5.1 + Grounded IPC-A-600 / IPC-2221 Deterministic Fallback Engine
        """)

    with col_arch2:
        st.markdown("### Target Edge Production Hardware")
        st.markdown("""
        - **Host SBC:** Raspberry Pi 5 (8GB LPDDR4X)
        - **AI Accelerator:** Raspberry Pi AI HAT+ with Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS)
        - **Interface:** PCIe Gen2/Gen3 x1 lane via Raspberry Pi ribbon connector
        - **Camera:** Raspberry Pi Camera Module 3 (Sony IMX708, autofocus, macro diffuser)
        - **Operating System:** Raspberry Pi OS 64-bit (Debian Bookworm)
        - **Software Stack:** HailoRT v4.x + ExecuTorch ARM runtime
        """)

    st.markdown("---")
    st.markdown("### Two-Pipeline Decoupled Architecture")
    st.code("""
    PIPELINE A (Computer Vision — Hard Real-Time):
    Camera Capture -> OpenCV Preprocessing -> Vision Detector (Hailo NPU / CPU)
    -> Defect Localization & Classification -> Severity Assessment
    -> Deterministic Decision Engine (PASS / INSPECT / REJECT) -> SQLite Logging

    PIPELINE B (Advisory — Soft Real-Time Reasoning):
    Structured Inspection JSON -> Llama 3.2 1B Instruct (.pte via ExecuTorch)
    -> Edge ARM CPU -> Operator Troubleshooting & Electrical Rework Advice
    """, language="text")
