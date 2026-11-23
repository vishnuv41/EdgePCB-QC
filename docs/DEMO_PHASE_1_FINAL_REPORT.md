# EdgePCB-QC — First Demo Phase Finalization Freeze Report

**Project:** EdgePCB-QC (Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5)  
**Document Classification:** Formal Phase Finalization & Demonstration Baseline  
**Date:** 2026-11-06  
**Status:** **FROZEN & VERIFIED FOR PRODUCTION DEMO**

---

## 1. Executive Summary & Absolute Freeze Declaration

As authorized by project governance, all future model experimentation (including Experiments H2, H3, and beyond) has been **OFFICIALLY FROZEN**. EdgePCB-QC is locked in its first complete, stable, honest, and offline-capable demonstration build.

* **Production Model Checkpoint:** `models/checkpoints/pcb_multidomain_experiment/weights/best.pt`
* **Verified Checkpoint SHA-256:** `66d8d225fe2137c160b8038d0e68200a54d1544152b0941050f749ee0741be04`
* **Model Checkpoint Size:** 6,499,634 bytes (6.20 MB)
* **Architecture:** YOLOv8n (73 layers fused, 3.01M parameters, 8.2 GFLOPs)
* **Active Production Backend:** `CPUInspector` (`inspection/cpu_inference.py`) with 640×640 Tiled Inference (`inspection/tiling.py`)
* **Decision Engine:** Deterministic rule-based gating (`decision_engine/decision.py`)
* **Advisory Engine:** ExecuTorch runtime (`advisory/executorch_runtime.py`) using `llama3_2_1b_fp16.pte` (2.36 GB) with instant rule-based fallback (`advisory/fallback.py`)
* **Production Demo Interface:** Standalone Streamlit Industrial Workstation on `http://localhost:8501`
* **Regression Test Status:** **129 / 129 automated unit & integration tests passing (100% in 18.36s)**
* **Architecture Consolidation:** React and FastAPI components permanently decommissioned per DECISION 050; Streamlit retained as the sole, self-contained, offline UI.

---

## 2. Research Checkpoint & Artifact Archival Matrix

In accordance with strict integrity rules, all prior experimental checkpoints, dataset manifests, and benchmark reports have been archived and preserved without modification:

| Checkpoint Name | Directory Path | SHA-256 Hash | Status / Role |
|---|---|---|---|
| **Production Baseline** | `models/checkpoints/pcb_multidomain_experiment/weights/best.pt` | `66d8d225fe2137c160b8038d0e68200a54d1544152b0941050f749ee0741be04` | **FROZEN PRODUCTION MODEL** |
| **Clean Experiment C2** | `models/checkpoints/exp_c2_color_hard_negatives_clean/weights/best.pt` | `281292144eb637f0022f4ce8e7bbefd5656122d1f760da6e02611765c92c89aa` | Archived Research Artifact |
| **Experiment G (11-Class)**| `models/checkpoints/exp_g_11class/weights/best.pt` | `8194e1918d3d74c4bbf7029711e3f80ef18571b9d13c8cc1c2518a8023d7a5f6` | Archived Research Artifact |
| **Experiment H1 (Hard Neg)**| `models/checkpoints/exp_h1_hard_negatives/weights/best.pt` | `37cc388262c2e5162cf70214ada8dc7f61f2b72762b214a886635515c39ed5ab` | Archived Forensic Baseline |
| **Llama 3.2 1B Instruct** | `models/checkpoints/llama3_2_1b_fp16.pte` | `2,472,432,290 bytes` (Verified ExecuTorch) | Production Advisory Engine |

---

## 3. Standalone Streamlit Workstation Implementation & Operational Tabs

The EdgePCB-QC Streamlit Workstation (`dashboard/app.py` on `http://localhost:8501`) operates completely offline with zero external network dependencies, featuring five complete, verified operational tabs:

| Tab | Name | Status | Verified Operational Features |
|---|---|---|---|
| **Tab 1** | **🔍 PCB Visual Inspection** | **COMPLETE** | • Dual Ingestion: Drag-and-drop test board upload + Golden-reference template selection (DeepPCB paired boards).<br>• Real backend inference via `CPUInspector` (YOLOv8n + DecisionEngine).<br>• Annotated PCB visual display with color-coded bounding boxes and confidence tags.<br>• Defect ROI Magnifier (`dashboard/utils.py::extract_defect_crop`) rendering genuine coordinate crops from the original image array.<br>• Deterministic Decision Panel: Displays rule ID, applied preset, quality coverage, and defect metrics.<br>• Advisory Panel: Displays ExecuTorch Llama 3.2 1B disposition, risk assessment, and operator action. |
| **Tab 2** | **🔬 Human Review & Verification** | **COMPLETE** | • Pending Queue: Lists unreviewed inspections requiring operator verification.<br>• Contained PCB viewer with defect selection.<br>• Defect Audit Form: Reviewer ID, AI Class, AI Confidence, Bounding Box coordinates, Reason / Rationale dropdown, and Operator Comments.<br>• One-click hotkey verification: `Confirmed [C]`, `False Call [F]`, `Uncertain [U]`.<br>• Non-destructive persistence to SQLite `defect_reviews` table. |
| **Tab 3** | **📋 Inspection Log & Audit** | **COMPLETE** | • Searchable historical audit records with filter by Board ID, Decision (`PASS`, `INSPECT`, `REJECT`), and date range.<br>• Detailed drawer displaying raw image, annotated detections, decision metadata, and operator reviews.<br>• Direct CSV and JSON export buttons for regulatory traceability. |
| **Tab 4** | **📊 Pareto & Defect Analytics** | **COMPLETE** | • Overall PASS / INSPECT / REJECT yield distribution.<br>• True physical boards vs. repeated runs tracking.<br>• Defect Pareto ranking chart highlighting top defect modes.<br>• False-call rate metrics per defect class based on operator audit logs. |
| **Tab 5** | **⚙️ Hardware & Edge Architecture** | **COMPLETE** | • Station Telemetry: Host platform (Windows 11 x86_64), CPU / RAM / Disk usage.<br>• Edge Benchmark Comparison: Measured PyTorch CPU inference latency vs Projected Hailo-8 NPU / Raspberry Pi 5 throughput.<br>• Active Checkpoint metadata: Model name, SHA-256 hash, architecture (YOLOv8n, 8.2 GFLOPs).<br>• Offline air-gap compliance confirmation. |

---

## 4. Data Provenance & Integrity Corrections

1. **Synthetic Integration Test Clean-Up:**
   - Modified integration test suites so test assertions use `input_source = "test"` exclusively, preventing synthetic OpenCV images from being saved with `"real"`.
2. **Web Upload Defaulting:**
   - Manual web uploads through the Streamlit dashboard strictly default to `input_source = "demo"`.
3. **Restricting Production Classification:**
   - The `PRODUCTION` classification is strictly reserved for authenticated physical camera streams or payloads bearing hardware attestation.
4. **Physical Board Deduplication:**
   - In the live SQLite database (`database/inspections.db`): **187 unique physical boards** and **99 repeated inspection runs** are tracked and reported separately.
5. **Database Immutability Preserved:**
   - Historical records in `database/inspections.db` were preserved untouched without unapproved destructive migrations.

---

## 5. System Stability & Air-Gap Verification

* **Memory Footprint:** Streamlit server RSS is stable at ~64.6 MB. Image bitmaps are decoded in-memory and immediately garbage-collected after inference.
* **Concurrency & Connection Handling:** Thread-safe SQLite connection sessions managed through context managers (`with self.db_mgr.Session() as session:`).
* **Air-Gap Verification:** Tested and confirmed 100% offline:
  - Zero external CDN scripts or stylesheet links.
  - Zero Google Fonts or external typography requests.
  - Zero analytics trackers or remote telemetry beacons.
  - Zero Node.js or npm runtime dependencies.

---

## 6. Hardware Status & Honest Disclaimers

EdgePCB-QC enforces absolute honesty in reporting hardware capabilities:

* **Host Workstation:** x86_64 Development Workstation (Windows 11) with PyTorch CPU runtime.
* **Target Embedded SBC:** Raspberry Pi 5 8GB (**Target Deployment — Physical Procurement Pending**).
* **Target NPU Accelerator:** Raspberry Pi AI HAT+ with Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) (**UNAVAILABLE — Physical Procurement Pending**).
* **Development GPU Disclaimer:** The workstation's NVIDIA GeForce RTX 5050 Laptop GPU was utilized strictly for training acceleration and dataset benchmarking. **RTX 5050 measurements are NEVER claimed as Raspberry Pi or Hailo performance.**
* **Advisory Pipeline Decoupling:** Generative Llama inference is purely advisory. The deterministic quality verdict is established in $< 0.1\text{ ms}$ before the LLM is invoked, ensuring that the inspection pipeline never stalls.

---

## 7. Exact Startup Commands for First Demonstration

To launch the frozen demonstration build, run the following commands in the workspace root:

### Step 1: Launch the Streamlit Industrial Workstation
```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Launch Streamlit on Port 8501
.\.venv\Scripts\streamlit.exe run dashboard/app.py --server.port 8501
```
* **Access URL:** Open your web browser to `http://localhost:8501`.

### Step 2: Automated Regression Verification
```powershell
# Run the complete test suite (129 unit & integration tests)
.\.venv\Scripts\pytest.exe -v
```

---

## 8. Itemized Status Audit

| Scope Item | Deliverable / Component | Status | Empirical Note |
|---|---|---|---|
| **Vision Model** | Baseline Multi-Domain (`pcb_multidomain_best.pt`) | **COMPLETE** | Verified SHA-256: `66d8d225fe...`, 8.2 GFLOPs, 3.01M params. |
| **Research Artifacts** | Clean C2, Exp G, Exp H1 checkpoints | **COMPLETE** | Preserved in separate directories without overwriting. |
| **Model Freezing** | Experimentation & Retraining Freeze | **COMPLETE** | Absolute freeze enforced. Zero training processes active. |
| **Decision Engine** | Deterministic Quality Gating & Safety Rules | **COMPLETE** | 100% rule-based; zero generative AI decision leakage. |
| **Advisory Engine** | ExecuTorch Llama 3.2 1B + Instant Rule Fallback | **COMPLETE** | FP16 `.pte` verified on CPU; non-blocking fallback active. |
| **Inspection Tab** | YOLO Inference, Golden Reference, Defect Table | **COMPLETE** | Side-by-side template comparison, defect ROI magnifier. |
| **Review Tab** | Human Review Queue & Defect Audit Workflow | **COMPLETE** | Hotkeys (`[C]`, `[F]`, `[U]`), live stats update, non-destructive. |
| **Records Tab** | Searchable Audit Logs, Overrides, CSV/JSON Export | **COMPLETE** | Real-time filtering, drawer inspection, SQLite audit trail. |
| **Analytics Tab** | Pareto Distribution, Physical Board Deduplication | **COMPLETE** | 187 unique boards / 99 repeated runs reported separately. |
| **Hardware Tab** | Honest Hardware Telemetry & Benchmarks | **COMPLETE** | Clear `MEASURED` vs `PROJECTED` pills; Hailo marked UNAVAILABLE. |
| **Air-Gap Security**| Zero Remote CDNs, Fonts, or External Trackers | **COMPLETE** | Verified 100% offline compliance. |
| **Hardware Deploy** | Raspberry Pi 5 & Hailo-8 Physical Testing | **BLOCKED** | Hardware pending physical procurement from user. |
| **Quantization** | 8da4w XNNPACK Quantization for Llama | **BLOCKED** | Requires Linux/WSL2 with `torchao` and GCC 11+. |
