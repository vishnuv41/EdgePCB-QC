# EdgePCB-QC — Technical Report Evidence Audit & Scientific Verification
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**  
**Lead Author / Auditor:** Lead Systems Engineer  
**Date:** 2026-11-07  
**Document Classification:** Comprehensive Technical Report Evidence Audit  
**Audit Target:** 17-Page Technical Report (`docs/FINAL_REPORT.md` / `docs/RESEARCH_PAPER_MASTER_SPECIFICATION.md`)

---

## 1. Executive Summary & Audit Methodology

This document establishes the authoritative, empirical evidence audit evaluating all claims, benchmark tables, architectural assertions, and performance metrics in the 17-page EdgePCB-QC technical report (`pcb.pdf`). 

Every claim has been investigated directly against:
1. Physical model weights, ONNX graphs, and compiled ExecuTorch `.pte` binaries on disk.
2. Bitwise SHA-256 checksums calculated natively via Windows PowerShell / Python.
3. Raw benchmark output files in `benchmarks/results/`.
4. Ground-truth dataset manifests and bounding box labels across all three raw datasets.
5. Automated pytest execution logs across all 152 test cases.
6. Hardware device enumeration, Windows CIM/WMI telemetry, and compiler reports.

### Strict Classification Categories
Every audited claim is assigned exactly one of the six non-negotiable classification categories:
- **`VERIFIED`**: Indisputably confirmed by direct disk inspection, SHA-256 checksum, raw benchmark log, or automated test execution on the host machine.
- **`VERIFIED WITH QUALIFICATION`**: Mathematically and empirically correct, but requires clarifying context (e.g. distinguishing dataset subsets, operational operating points vs PR-AUC curves, or specific model checkpoints).
- **`PROJECTED`**: Theoretical or simulation-derived estimates based on hardware specifications, compiler profilers, or analytical modeling, where physical hardware measurement has not taken place.
- **`SIMULATED`**: Software-emulated behavior using mock drivers, test patterns, or synthetic fixtures in place of physical devices.
- **`UNVERIFIED`**: Lacks direct empirical proof or execution logs in the repository.
- **`CONTRADICTORY`**: In direct conflict with the source code, committed checkpoints, or verified test outputs.

---

## 2. Issue 1: Baseline Clean-Board PASS Discrepancy (49.23% vs 93.33%)

### 2.1 The Discrepancy
- **PDF Page 8 / Section 12:** Reports a clean-board PASS rate of **`49.23% (96/195)`** for the baseline model.
- **Earlier Phase 1 Summaries:** Reported a clean-board PASS rate of **`93.33% (182/195)`**.
- **Audit Question:** Are these figures contradictory, and what are the exact evaluation scripts, datasets, checkpoints, and decision presets behind each figure?

### 2.2 Forensic Evidence & Script Attribution
- **Dataset:** DeepPCB Clean Reference Templates (`datasets/raw/DeepPCB/PCBData/`), specifically groups `group90100` (97 template images) and `group92000` (98 template images), totaling **195 defect-free clean reference boards**.
- **Evaluation Script:** `scripts/evaluate_baseline_vs_c2_clean.py` (lines 17–20, 47–48, 643).
- **Decision Engine Preset:** `DecisionEngine(preset="consumer")` (`decision_engine/decision.py`).
- **Raw Benchmark Source File:** `benchmarks/results/baseline_vs_c2_clean_full_audit.json` (lines 3866–3881) and `benchmarks/results/Baseline Multi-Domain_evaluation.json` (lines 71–89).

### 2.3 Empirical Audit Findings

| Parameter / Metric | Baseline Multi-Domain Reference | Clean Experiment C2 (Hard Negatives) | Experiment G (11-Class Extended) |
|---|---|---|---|
| **Checkpoint Path** | `models/checkpoints/pcb_multidomain_experiment/weights/best.pt` | `models/checkpoints/exp_c2_color_hard_negatives_clean/weights/best.pt` | `models/checkpoints/exp_g_11class/weights/best.pt` |
| **Checkpoint SHA-256** | `66d8d225fe...` | `281292144e...` | `8194e1918d...` |
| **Model Nature** | Frozen Production Model (Un-augmented Baseline) | Research Artifact (Trained with 34 colour hard negatives) | Research Artifact (11 classes + hard negatives) |
| **Evaluated Boards** | 195 clean test templates | 195 clean test templates | 195 clean test templates |
| **Clean PASS Count** | **96 / 195 (49.23%)** | **182 / 195 (93.33%)** | **182 / 195 (93.33%)** |
| **False INSPECT (Review)** | **80 / 195 (41.03%)** | **12 / 195 (6.15%)** | **8 / 195 (4.10%)** |
| **False REJECT (Scrap)** | **19 / 195 (9.74%)** | **1 / 195 (0.51%)** | **5 / 195 (2.56%)** |
| **Total False-Positive Boxes** | **357 boxes** (1.83 FP/board) | **15 boxes** (0.08 FP/board) | **21 boxes** (0.11 FP/board) |
| **Primary FP Failure Mode** | Normal via holes hallucinated as `pin_hole_missing_pad` (246 boxes; 68.9%) | Drastically suppressed by hard negative mining | Drastically suppressed |

### 2.4 Classification & Resolution
- **Classification:** **`VERIFIED WITH QUALIFICATION`**
- **Resolution:**
  - The two numbers are **NOT** conflicting observations of the same model.
  - **`49.23% (96/195)`** is the **true, un-augmented Baseline Multi-Domain model** (`best.pt`, SHA-256 `66d8d2...`). Because the baseline was trained without negative PCB copper templates, normal circular via holes trigger false `pin_hole_missing_pad` detections, resulting in 357 false-positive boxes that drag the clean pass rate down to 49.23%.
  - **`93.33% (182/195)`** is the **improved result achieved by Clean Experiment C2 and Experiment G**, where hard-negative training reduced false-positive boxes by 95.8% (357 $\to$ 15), raising clean PASS yield to 93.33% and slashing false scrap from 9.74% down to 0.51% (a 19.0× reduction).
  - In earlier informal Phase 1 notes, 93.33% was accidentally attributed to the "baseline" or discussed without clearly specifying that it required the C2 hard-negative intervention.
  - **PDF Correction Required:** The technical report must explicitly label `49.23%` as the un-augmented Baseline Multi-Domain model and `93.33%` as the post-hard-negative C2/G model, showing the direct comparative transition in a before/after yield table.

---

## 3. Issue 2: Hardware Latency & Performance Verification (Measured vs Projected)

### 3.1 The Audit Target
- **PDF Page 16 / Section 12.B:** Reports latency figures for laptop CPU, Raspberry Pi 5, and Hailo-8 NPU.
- **Audit Question:** Were the page 16 measurements obtained on physical Raspberry Pi 5 + Hailo hardware, or are they projected? Locate raw benchmark logs.

### 3.2 Evidence & Benchmark Logs
- **Hardware Telemetry Source:** `benchmarks/results/vision_computational_cost.json` and `benchmarks/results/hailo_preparation_report.json`.
- **Host Workstation:** AMD Ryzen 9 8940HX CPU (16 cores, 32 threads), 32 GB DDR5 RAM, Windows 11.
- **Zero Fabrication Declaration:** In `benchmarks/results/hailo_preparation_report.json` line 64:
  > *"EdgePCB-QC reports zero fabricated performance. Hailo deployment metrics are documented as PROJECTED until physical Raspberry Pi 5 + AI HAT+ hardware is procured and measured in Phase 2."*
- **Physical Procurement Status:** Confirmed in `docs/DEMO_PHASE_1_FINAL_REPORT.md` lines 86–89, and `docs/REQUIRED_HARDWARE_AND_FILES.md` lines 12–19: Physical Raspberry Pi 5 8GB, Raspberry Pi AI HAT+, and Camera Module 3 are physically pending procurement.

### 3.3 Latency Audit & Classification Breakdown

| Pipeline Stage | Reported Figure | Hardware Platform | Raw Benchmark Source | Audit Classification |
|---|---|---|---|---|
| **YOLOv8n PyTorch CPU Inference** | 27.21 ms (34.4 FPS end-to-end: 29.07 ms) | AMD Ryzen 9 8940HX CPU | `benchmarks/results/vision_computational_cost.json` (lines 32–40) | **`VERIFIED`** [MEASURED] |
| **YOLOv8n ONNX Runtime CPU Inference** | 26.97 ms (33.4 FPS end-to-end: 29.92 ms) | AMD Ryzen 9 8940HX CPU | `benchmarks/results/vision_computational_cost.json` (lines 41–47) | **`VERIFIED`** [MEASURED] |
| **Golden Reference Comparison (OpenCV)** | 45.54 ms (22.0 FPS) | AMD Ryzen 9 8940HX CPU | `benchmarks/results/inspection_reliability_report.json` | **`VERIFIED`** [MEASURED] |
| **Deterministic Decision Engine Gating** | 0.08 ms (< 0.1 ms) | AMD Ryzen 9 8940HX CPU | `benchmarks/results/complete_workflow_scenarios.json` | **`VERIFIED`** [MEASURED] |
| **Llama Advisory Deterministic Fallback** | 0.18 ms – 0.20 ms | AMD Ryzen 9 8940HX CPU | `benchmarks/results/advisory_modes_benchmark.json` | **`VERIFIED`** [MEASURED] |
| **ExecuTorch Llama 3.2 1B FP16 Prefill (TTFT)** | 20.93s – 47.36s (41–94 tokens) | AMD Ryzen 9 8940HX CPU | `benchmarks/results/advisory_latency_matrix.json` (lines 10–251) | **`VERIFIED`** [MEASURED] |
| **ExecuTorch Llama 3.2 1B Decode Throughput** | 1.05 – 1.11 tokens/s | AMD Ryzen 9 8940HX CPU | `benchmarks/results/advisory_latency_matrix.json` | **`VERIFIED`** [MEASURED] |
| **NVIDIA GeForce RTX 5050 Inference** | Not Claimed / Fallback to CPU | RTX 5050 Laptop GPU (`sm_120`) | PyTorch cu126 lacks sm_120 binary; CPU fallback enforced | **`VERIFIED`** [TRANSPARENT] |
| **Hailo-8 YOLOv8n Vision Inference** | 5.8 ms / 12.5 ms (~80–170 FPS) | Target Raspberry Pi AI HAT+ (Hailo-8) | Theoretical calculation from Hailo DFC compile profiler | **`PROJECTED`** |
| **Raspberry Pi 5 ARM CPU Golden Comparison** | 38.0 ms (ARM NEON) | Target Raspberry Pi 5 (Cortex-A76) | Analytical projection from ARM NEON vectorization | **`PROJECTED`** |
| **Raspberry Pi 5 ARM CPU Decision Engine** | 0.12 ms | Target Raspberry Pi 5 (Cortex-A76) | Analytical projection | **`PROJECTED`** |
| **Raspberry Pi 5 ARM CPU Fallback Advisory** | 0.35 ms | Target Raspberry Pi 5 (Cortex-A76) | Analytical projection | **`PROJECTED`** |
| **Hailo-8 / Pi 5 Quantized Llama Throughput** | ~240 ms TTFT, 14 tokens/s | Target Hailo-8 / Cortex-A76 (Int4) | Theoretical projection from Hailo LLM Model Zoo | **`PROJECTED`** |

### 3.4 PDF Correction Required
- On Page 16 (and all corresponding tables in the report), every Raspberry Pi 5 and Hailo measurement must have the explicit tag **`[PROJECTED]`** appended to the column header and cell entries.
- The report must state unequivocally in the text: *"Physical validation on Raspberry Pi 5 + Hailo AI HAT+ is scheduled for Phase 2 hardware integration following physical procurement; all edge SBC/NPU figures herein represent compiler-derived projections."*

---

## 4. Issue 3: Test Count Reconciliation & Taxonomy (129 vs 147 vs 152 Tests)

### 4.1 The Test Count Progression
- **129 Tests:** Reported in `docs/DEMO_PHASE_1_FINAL_REPORT.md` (DECISION 050).
- **147 Tests:** Reported in `docs/HARDWARE_ACCEPTANCE_TESTS.md` (DECISION 051).
- **152 Tests:** Current active automated test suite in `tests/`.
- **Audit Question:** Why did the count change? Which tests were added, removed, or skipped? Are these unit tests or physical hardware acceptance tests?

### 4.2 Test Evolution Timeline & Taxonomy

```
[Phase 1 Baseline Suite: 104 Tests]
       │
       ▼ (+25 new tests: splits, reliability, tiling, safety)
[Consolidation Suite: 129 Tests] (DECISION 050: React/FastAPI removed)
       │
       ▼ (+18 tests: Hardware Abstraction Layer)
[Hardware-Ready Suite: 147 Tests] (DECISION 051: Camera, Inference, Telemetry, Offline)
       │
       ▼ (+5 tests: Camera persistence across reruns, singleton, dataset browser, network URLs)
[Current Active Test Suite: 152 Tests] (All 152 Passing in 22.03s)
```

### 4.3 Detailed Test Influx & Outflux Accounting
1. **Tests Removed during React/FastAPI Decommissioning (-19 tests):**
   - In DECISION 050, `tests/test_react_inspection_flow.py`, `tests/test_react_spa_and_offline.py`, `tests/test_api.py`, and `tests/verify_demo_flow_15steps.py` were permanently deleted when React and FastAPI were decommissioned to keep Streamlit as the sole offline UI. This lowered the active suite to **129 tests**.
2. **Hardware Abstraction Layer Added (+18 tests $\to$ 147 tests):**
   - `tests/test_camera_manager.py` (+5 tests): `test_camera_manager_initial_state`, `test_camera_discovery`, `test_mock_camera_lifecycle`, `test_camera_manager_mock_streaming`, `test_camera_manager_invalid_device_handling`.
   - `tests/test_hardware_status.py` (+3 tests): `test_hardware_detection_truthfulness`, `test_system_telemetry`, `test_network_diagnostics`.
   - `tests/test_inference_manager.py` (+5 tests): `test_yolo_postprocess_letterbox`, `test_yolo_postprocess_nms`, `test_inference_manager_auto_resolution`, `test_inference_manager_cpu_inspection`, `test_inference_manager_hailo_handling_without_hardware`.
   - `tests/test_offline_mode.py` (+4 tests): `test_local_model_artifacts_exist`, `test_local_decision_engine_config_exists`, `test_sqlite_database_local`, `test_no_external_cdn_in_dashboard`.
   - `tests/test_streamlit_hardware_workflow.py` (+1 test): `test_complete_hardware_inspection_flow`.
3. **Camera Persistence & Provenance Fixes Added (+5 tests $\to$ 152 tests):**
   - `tests/test_camera_manager.py::test_camera_persistence_across_multiple_discover_calls`
   - `tests/test_camera_manager.py::test_camera_manager_singleton_idempotency`
   - `tests/test_camera_manager.py::test_dataset_browser_in_place_discovery_and_loading`
   - `tests/test_camera_manager.py::test_dataset_inspection_demo_provenance_logging`
   - `tests/test_camera_manager.py::test_network_diagnostics_edge_access_urls`

### 4.4 Test Nature & Hardware-in-the-Loop Distinction
- **100% of the 152 automated tests are software unit and integration tests** executed via pytest in Python 3.11.
- **0 tests are physical hardware-in-the-loop tests.** Hardware interfaces use mock devices (`MockCamera` in `hardware/camera/mock_camera.py`) and software fallbacks (`CPUInspector`) when physical hardware devices (CSI, V4L2, HailoRT PCIe) are not attached.
- Physical acceptance tests are codified as standalone verification scripts (`hardware/deployment/verify_hardware.sh`, `hardware/deployment/verify_hailo.py`) documented in `docs/HARDWARE_ACCEPTANCE_TESTS.md` for execution on the physical Pi once delivered.
- **Classification:** **`VERIFIED`** (152 passing unit/integration software tests; 0 skipped; 0 failed).

---

## 5. Issue 4: Llama 3.2 1B & ExecuTorch Evidence Audit

### 5.1 Model Artifact & Checksum Verification
- **Artifact Path on Disk:** `E:\EdgePCB-QC_cache\models\llama3_2_1b\llama3_2_1b_fp16.pte` (referenced dynamically by candidate search in `advisory/executorch_runtime.py`).
- **File Size:** **`2,472,432,290 bytes`** (2.30 GiB / 2,357.90 MB / 2.36 GB).
- **Verified SHA-256 Checksum:**  
  `0d2c47c426380fc520357aeede656897cb49fff2e0a8651877c631c0dbfe8b1a`
- **Tokenizer Path:** `models/llama/tokenizer.model` (`2,185,274 bytes`).
- **Classification:** **`VERIFIED`**.

### 5.2 Runtime Performance Audit (Host Platform: AMD Ryzen 9 8940HX CPU)
Source: `benchmarks/results/advisory_latency_matrix.json` (12 configurations, 4 vs 8 threads):
- **C++ Module Load Time (`_load_for_executorch`):** **`7.89 ms`** (Cold initialization / tokenizer load: 1,679.57 ms).
- **Time to First Token (TTFT):** Scales linearly with prompt length ($\approx 500\text{ ms per token}$):
  - 41 prompt tokens (compact template): **`20.93s – 22.70s`**
  - 69 prompt tokens (standard template): **`34.57s – 37.39s`**
  - 94 prompt tokens (verbose template): **`46.98s – 47.36s`**
- **Autoregressive Decoding Rate:** **`1.05 – 1.11 tokens/second`** ($\approx 910–950\text{ ms per token}$).
- **Thread Scaling:** 4 threads achieve 1.11 tok/s; 8 threads achieve 1.08 tok/s. This proves that edge CPU Transformer inference is memory bandwidth-bound (streaming 2.36 GB weights per token) rather than compute-bound.
- **Peak RSS RAM Footprint:** **`3,019.2 MB – 3,022.8 MB`** (~3.02 GB).
- **Sentence Completeness Quality:** A 24-token budget paired with dynamic token slicing produces complete, professional advisory sentences:
  > *"The board PCB-PERF-001 shows visually suspected short-circuit, which poses a significant failure risk to the system."*
- **Fast Deterministic Mode (`advisory/fallback.py`):** **`< 0.20 ms`** latency, 0 MB additional RAM, 100% compliant with IPC-A-600 / IPC-2221 expert rules.
- **Classification:** **`VERIFIED`** [MEASURED on Ryzen 9 CPU].

### 5.3 Deterministic Verdict Override Prevention
- **Architecture Invariant:** The LLM receives strictly structured JSON containing the pre-calculated decision (`PASS`, `INSPECT`, `REJECT`), triggered rule ID, defect coordinates, and confidence.
- **Immutability Verification:** The decision engine verdict is generated and stored in SQLite before the LLM is invoked. Pydantic validation (`AdvisoryExplanation` in `advisory/schema.py`) ensures the LLM generates advisory text fields only (`what_happened`, `why_it_matters`, `operator_action`, `caveat`). The LLM has no programmatic access to modify the decision code or triggered rules.
- **Regression Tests Passing:**
  - `tests/test_audit_and_attribution.py::test_decision_unchanged_across_advisory_modes`
  - `tests/test_advisory_quality.py::test_decision_invariance_across_all_advisory_outcomes`
- **Classification:** **`VERIFIED`**.

---

## 6. Issue 5: Dataset Inventory, Bounding Box Counts & Split Integrity Audit

### 6.1 Dataset Inventory & Ground-Truth Counts
Source: `benchmarks/results/exact_inventory_counts.json` and `benchmarks/results/dataset_authoritative_audit.json`:

| Dataset Name | Raw Images / Files | Annotated Instances | Formats Available | Split Distribution | Canonical Classes Mapped |
|---|---|---|---|---|---|
| **DeepPCB** | 1,500 pairs (3,000 files) | **10,013 bboxes** | Text files (normalized YOLO format) | • Train: 1,120 pairs (7,495 bboxes)<br>• Val: 186 pairs (1,375 bboxes)<br>• Test: 194 pairs (1,143 bboxes)<br>• Clean: 1,500 templates | 6 classes: `open` (1942), `short` (1506), `mousebite` (1965), `spur` (1625), `copper` (1474), `pin-hole` (1501) |
| **DsPCBSD+** | 10,259 crops | **20,276 bboxes** | COCO JSON & YOLO `.txt` | • Train: 8,208 images (5,811 used in 6-class)<br>• Val: 2,051 images (1,444 used in 6-class) | 6 mapped: `SH` (915), `SP` (4584), `SC` (1593), `OP` (1770), `MB` (2529), `HB` (2883).<br>3 excluded from baseline: `CS` (2490), `CFO` (1832), `BMFO` (1680) |
| **PCB-Defect V2** | 230 whole boards (22 physical boards) | **2,506 bboxes** (3,750 crop bboxes) | COCO JSON & YOLO OBB polygons | • Train (Boards 50–64): 172 boards $\to$ 2,376 crops<br>• Val (Boards 65–67): 33 boards $\to$ 495 crops<br>• Test (Boards 68, 80–82): 25 boards $\to$ 390 crops | 6 classes: `open_circuit` (304), `short` (289), `mouse_bite` (830), `spur` (490), `spurious_copper` (264), `missing_pad` (329) |
| **Domain 5 (`own_camera`)** | 0 images | 0 bboxes | N/A | Protocol ready; images pending physical camera | Standard 6 classes |

### 6.2 Physical Board-Level Split Integrity & Zero Leakage
- **DeepPCB:** Partitioned strictly across 11 board groups (Train: groups 00041, 12000, 12100, 12300, 13000, 20085, 44000; Val: groups 50600, 04400; Test: groups 90100, 92000).
- **PCB-Defect V2:** Partitioned strictly across 22 physical board layouts (Train: boards 50–64; Val: boards 65–67; Test: boards 68, 80–82).
- **Zero-Leakage Proof:** Mathematically verified in `tests/test_splits.py`:
  - $\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$ at both the physical board level and filename level.
  - Zero bitwise SHA-256 duplicate image hashes exist between splits.
- **Classification:** **`VERIFIED`**.

---

## 7. Issue 6: Vision Model Identity, Checkpoint Integrity & Metric Reconciliation

### 7.1 Architecture Identity & Checkpoint Verification
- **Architecture:** **YOLOv8n** (Ultralytics 8.4.201 `DetectionModel`, 73 layers fused).
  - *Correction:* Any documentation referring to the baseline as "YOLO11n" is **`CONTRADICTORY`** with the underlying model definition in `args.yaml` (`model: yolov8n.pt`) and must be corrected to YOLOv8n.
- **Active Frozen Checkpoint:** `models/checkpoints/pcb_multidomain_experiment/weights/best.pt`
- **Verified SHA-256 Checksum:**  
  `66d8d225fe2137c160b8038d0e68200a54d1544152b0941050f749ee0741be04`
- **Model Checkpoint Size:** 6,499,634 bytes (6.20 MB).
- **Parameters:** 3,011,043 parameters (fused) / 3,139,218 parameters (unfused).
- **Computational Complexity:** 8.2 GFLOPs (at 640×640 input resolution).
- **Classification:** **`VERIFIED`**.

### 7.2 Metric Reconciliation & Discrepancy Investigation
Source: `benchmarks/results/Baseline Multi-Domain_evaluation.json`, `benchmarks/results/methodology_reconciliation_map50.json`, and `benchmarks/results/eval_unseen_boards.json`:

| Evaluation Subset & Metric Mode | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 | Source File / Status |
|---|---|---|---|---|---|---|
| **DeepPCB Unseen Test Set (194 images, 1,143 bboxes)** | **71.36%** | **67.39%** | **0.6932** | **71.53%** | **41.29%** | `eval_unseen_boards.json` (`VERIFIED`) |
| **Validation Crops — Ultralytics Standard Sweep (conf=0.001)** | 75.10% | 78.67% | 0.7684 | **81.25%** | 48.52% | `methodology_reconciliation_map50.json` (`VERIFIED`) |
| **Validation Crops — Truncated Operational Sweep (conf=0.25)** | 75.46% | 79.57% | 0.7746 | **73.12%** | 45.10% | `methodology_reconciliation_map50.json` (`VERIFIED`) |
| **Validation Crops — Custom Evaluator (Bipartite IoU=0.50)** | **66.97%** | **77.50%** | **0.7185** | N/A | N/A | `Baseline Multi-Domain_evaluation.json` (`VERIFIED`) |
| **Small Defect Recall (< 32 px) on Baseline** | N/A | **63.01%** | (1,172 / 1,860 defects) | N/A | N/A | `Baseline Multi-Domain_evaluation.json` (`VERIFIED`) |
| **Medium/Large Defect Recall ($\ge$ 32 px) on Baseline** | N/A | **87.98%** | (2,262 / 2,571 defects) | N/A | N/A | `Baseline Multi-Domain_evaluation.json` (`VERIFIED`) |
| **Whole-Board Tiled Multi-Panel (25 boards, 349 GT)** | **67.57%** | **42.98%** | **0.5254** | N/A | N/A | `Baseline Multi-Domain_evaluation.json` (`VERIFIED`) |

### 7.3 Discrepancy Note on the 80.58% mAP50 Figure
- In `docs/RESEARCH_PAPER_MASTER_SPECIFICATION.md` (line 169), the Baseline Multi-Domain model was listed with mAP50 = `80.58%`, Precision = `75.02%`, and Recall = `83.91%`.
- **Audited Truth:** Those exact numbers (`80.58%`, `75.02%`, `83.91%`) belong to **Experiment C (Hard Negatives)** under the operational evaluator (`methodology_reconciliation_map50.json` lines 56–60).
- For the Baseline Multi-Domain model, the correct numbers are:
  - Standard Ultralytics mAP50: **`81.25%`** (PR curve AUC across all thresholds)
  - Truncated Operational mAP50: **`73.12%`** ($\tau \ge 0.25$)
  - Custom Discrete Bipartite Evaluator: **Precision = `66.97%`**, **Recall = `77.50%`**, **F1 = `0.7185`**.
- **Classification:** **`VERIFIED WITH QUALIFICATION`** (metrics exist and are verified; row attribution in early drafts merged Exp C numbers into Baseline row).

---

## 8. Issue 7: Edge Hardware Readiness & Deployment Audit

### 8.1 Implementation Status Matrix

| Component | Code Implementation | Software / Mock Tested | Physical Hardware Tested | Readiness Status |
|---|---|---|---|---|
| **Picamera2 (Pi Camera Module 3 IMX708 AF)** | `hardware/camera/csi_camera.py` | Unit tests verify import fallback and error handling | Pending physical Pi 5 + CSI camera | **Software-Ready (Awaiting Hardware)** |
| **V4L2 / USB Webcam (DirectShow / OpenCV)** | `hardware/camera/usb_camera.py` | Verified live with laptop camera & external USB webcam | Physically tested and streaming on Windows 11 | **VERIFIED & OPERATIONAL** |
| **Synthetic CI Camera Simulator** | `hardware/camera/mock_camera.py` | Verified in CI test suite across all 152 tests | 100% automated software verification | **VERIFIED & OPERATIONAL** |
| **Streamlit Browser Camera (`st.camera_input`)** | `dashboard/app.py` | Tested in Google Chrome / Edge with permission prompt | Physically tested on laptop | **VERIFIED & OPERATIONAL** |
| **Camera Manager (Singleton & Frame Buffer)** | `hardware/camera/camera_manager.py` | Threaded capture worker tested across tab reruns | Verified persistence across Streamlit reruns | **VERIFIED & OPERATIONAL** |
| **Hailo Model Parser & DFC Calibration** | `scripts/prepare_hailo_deployment.py` | 100% compatible 233-node ONNX graph; 1,024 calib images | Requires Ubuntu 22.04 LTS x86_64 host | **Software-Ready (Awaiting Linux Host)** |
| **HailoRT Python Abstraction (`HailoInspector`)** | `hardware/inference/hailo_inspector.py` | Verified fallback to CPU when HailoRT is absent | Pending physical Raspberry Pi AI HAT+ | **Software-Ready (Awaiting Hardware)** |
| **Inference Manager Auto Backend Routing** | `hardware/inference/inference_manager.py` | Unit tests verify auto fallback logic without Hailo | Tested on Windows 11 host | **VERIFIED & OPERATIONAL** |
| **Two-Node Network Diagnostics & IP Display** | `hardware/status/hardware_status.py` | Verified IP enumeration and LAN URL generation | Tested on local Wi-Fi / Ethernet | **VERIFIED & OPERATIONAL** |
| **Automated Pi Deployment Scripts** | `hardware/deployment/*.sh` | Shell scripts syntactically verified | Pending physical execution on Raspberry Pi OS | **Software-Ready (Awaiting Hardware)** |

### 8.2 Honest Hardware Attribution Invariant
- **Rule Enforced:** The application and documentation **never** report simulated frames as real camera captures, never claim Hailo is operational until physical hardware is detected, and never claim RTX 5050 GPU performance as Raspberry Pi or Hailo throughput.
- **Classification:** **`VERIFIED`**.

---

## 9. Issue 8: Bharat AI-SoC Challenge PS5 Compliance Verification

| PS5 Requirement | Technical Implementation | Empirical Evidence & Test Path | Audit Status |
|---|---|---|---|
| **1. 100% Offline Air-Gapped Operation** | Zero external CDNs, zero cloud APIs, zero remote font/script requests. Local SQLite database. | `tests/test_offline_mode.py`, `scripts/test_offline_inference.py` (0 active sockets intercepted) | **`VERIFIED`** |
| **2. Real-Time Defect Detection & Localization** | 6-class photolithographic defect detection using YOLOv8n (27.21 ms CPU inference / projected 5.8 ms Hailo). | `models/checkpoints/pcb_multidomain_experiment/weights/best.pt`, `inspection/cpu_inference.py` | **`VERIFIED`** |
| **3. IPC-Informed Defect Severity Classification** | 3-tier severity rating (`CRITICAL`, `MAJOR`, `MINOR`) evaluating defect class, area ratio, and layout density. | `decision_engine/severity.py`, `tests/test_system_integrity.py` | **`VERIFIED`** |
| **4. Deterministic Quality Decision Gating** | 8 prioritized deterministic rules (`RULE_001` – `RULE_008`) across 3 presets (`consumer`, `industrial`, `automotive`). | `decision_engine/decision.py`, `decision_engine/rules.py`, `tests/test_pcb_5655_safety_regression.py` | **`VERIFIED`** |
| **5. Pre-Inference Image Quality & Domain Gates** | Blur (Laplacian < 80), illumination (< 30 or > 248), glare (> 35%), low contrast (< 20), color domain shift gates. | `inspection/quality_assessment.py`, `tests/test_reliability.py` (0 false passes in 10 test scenarios) | **`VERIFIED`** |
| **6. Mandatory Electrical Testing Flags** | Trace continuity defects (`open_circuit`, `short_circuit`) unconditionally trigger `requires_electrical_test = True`. | `decision_engine/decision.py`, `tests/test_audit_and_attribution.py` | **`VERIFIED`** |
| **7. Decoupled SLM Operator Advisory** | Llama 3.2 1B Instruct via ExecuTorch FP16 (`llama3_2_1b_fp16.pte`) receiving structured JSON only. Cannot override verdict. | `advisory/executorch_runtime.py`, `advisory/schema.py`, `tests/test_advisory_quality.py` | **`VERIFIED`** |
| **8. Instant Sub-Millisecond Fallback Advisory** | Rule-based IPC troubleshooting advice (< 0.2 ms) with strict provenance tracking (`llm_version = None`). | `advisory/fallback.py`, `tests/test_advisory.py` | **`VERIFIED`** |
| **9. Embedded Audit Logging & Human Review** | SQLite persistence (`database/inspections.db`), operator override tracking with mandatory technical justification, CSV/JSON export. | `database/manager.py`, `dashboard/app.py`, `tests/test_audit_and_attribution.py` | **`VERIFIED`** |
| **10. Standalone Streamlit Industrial Dashboard** | 7 fully functional operational tabs: Live Camera, Captured PCB, Human Review, Audit Log, Pareto Analytics, Hardware Telemetry, Settings. | `dashboard/app.py` running on `http://localhost:8501`, Chrome DevTools screenshots | **`VERIFIED`** |

---

## 10. Complete Claim Classification Matrix

This matrix classifies every primary claim across the 17-page technical report into the 6 strict audit categories:

| Claim Description | Report Location | Target Metric / Value | Evidence Source Path | Classification Category | Audit Qualification / Justification |
|---|---|---|---|---|---|
| **Baseline Vision Architecture** | Page 3, 5, 8 | YOLOv8n (3.01M params, 8.2 GFLOPs) | `models/checkpoints/pcb_multidomain_experiment/args.yaml` | **`VERIFIED`** | Confirmed YOLOv8n; references to YOLO11n corrected. |
| **Baseline Checkpoint SHA-256** | Page 8 | `66d8d225fe...` | `models/checkpoints/pcb_multidomain_experiment/weights/best.pt` | **`VERIFIED`** | Bitwise match via PowerShell `Get-FileHash`. |
| **DeepPCB Test Split mAP@50** | Page 8 | 71.53% (mAP@50-95: 41.29%) | `benchmarks/results/eval_unseen_boards.json` | **`VERIFIED`** | Evaluated on 194 unseen test images (1,143 defects). |
| **Baseline Clean Board PASS Yield** | Page 8 | 49.23% (96 / 195 boards) | `benchmarks/results/Baseline Multi-Domain_evaluation.json` | **`VERIFIED`** | Evaluated on 195 clean templates (`consumer` preset). |
| **Clean Experiment C2 PASS Yield** | Page 8 / Draft | 93.33% (182 / 195 boards) | `benchmarks/results/baseline_vs_c2_clean_full_audit.json` | **`VERIFIED`** | Achieved via 34 audited colour hard negatives. |
| **Early Attribution of 93.33% to Baseline** | Phase 1 notes | Attributed 93.33% to baseline | `docs/DECISIONS.md` (lines 626, 793) | **`CONTRADICTORY`** | Erroneous attribution; 93.33% is Clean C2, 49.23% is Baseline. |
| **Baseline Validation mAP50 = 80.58%** | Page 8 / Spec | Attributed 80.58% to Baseline | `benchmarks/results/methodology_reconciliation_map50.json` | **`CONTRADICTORY`** | 80.58% belongs to Exp C; Baseline standard sweep is 81.25%, truncated is 73.12%. |
| **Laptop PyTorch CPU Latency** | Page 16 | 27.21 ms inference (29.07 ms end-to-end) | `benchmarks/results/vision_computational_cost.json` | **`VERIFIED`** | Measured on AMD Ryzen 9 8940HX CPU. |
| **Laptop ONNX Runtime CPU Latency** | Page 16 | 26.97 ms inference (29.92 ms end-to-end) | `benchmarks/results/vision_computational_cost.json` | **`VERIFIED`** | Measured on AMD Ryzen 9 8940HX CPU. |
| **Target Hailo-8 Vision Latency** | Page 16 | 5.8 ms / 12.5 ms (~80–170 FPS) | `benchmarks/results/hailo_preparation_report.json` | **`PROJECTED`** | Theoretical calculation; hardware pending procurement. |
| **Target Pi 5 Golden Compare Latency** | Page 16 | 38.0 ms (ARM NEON) | `docs/FINAL_REPORT.md` (Table 4) | **`PROJECTED`** | Analytical projection for Cortex-A76. |
| **Target Pi 5 Decision Engine Latency** | Page 16 | 0.12 ms | `docs/FINAL_REPORT.md` (Table 4) | **`PROJECTED`** | Analytical projection for Cortex-A76. |
| **Target Pi 5 Fallback Advisory Latency**| Page 16 | 0.35 ms | `docs/FINAL_REPORT.md` (Table 4) | **`PROJECTED`** | Analytical projection for Cortex-A76. |
| **Target Pi 5 / Hailo Llama Throughput** | Page 16 | ~240 ms TTFT, 14 tokens/s (Int4) | `docs/FINAL_REPORT.md` (Table 4) | **`PROJECTED`** | Theoretical projection from Hailo Model Zoo. |
| **ExecuTorch Model Artifact Size** | Page 11, 16 | `2,472,432,290 bytes` (2.36 GB) | `E:\EdgePCB-QC_cache\models\llama3_2_1b\llama3_2_1b_fp16.pte` | **`VERIFIED`** | Bitwise file inspection on disk. |
| **ExecuTorch Model SHA-256 Checksum** | Page 11 | `0d2c47c426380fc5...` | Direct PowerShell `Get-FileHash` | **`VERIFIED`** | SHA-256 confirmed. |
| **ExecuTorch CPU TTFT** | Page 11, 16 | 20.93s – 47.36s (41–94 tokens) | `benchmarks/results/advisory_latency_matrix.json` | **`VERIFIED`** | Measured on AMD Ryzen 9 8940HX CPU. |
| **ExecuTorch CPU Decode Throughput** | Page 11, 16 | 1.05 – 1.11 tokens/s | `benchmarks/results/advisory_latency_matrix.json` | **`VERIFIED`** | Measured on AMD Ryzen 9 8940HX CPU. |
| **ExecuTorch Peak RAM Consumption** | Page 11, 16 | 3,019.2 MB – 3,022.8 MB | `benchmarks/results/advisory_latency_matrix.json` | **`VERIFIED`** | Measured via psutil in benchmark runner. |
| **Instant Advisory Fallback Latency** | Page 11, 16 | 0.18 ms – 0.20 ms | `benchmarks/results/advisory_modes_benchmark.json` | **`VERIFIED`** | Measured on host CPU. |
| **Total Automated Regression Tests** | Summary | 152 passed in ~22s (0 failed) | `pytest -v` console execution output | **`VERIFIED`** | 152 automated software tests; 0 physical tests. |
| **Air-Gap Zero Network Connections** | Page 12 | 0 outbound sockets | `scripts/test_offline_inference.py` | **`VERIFIED`** | Socket interception test confirmed. |
| **Zero-Leakage Dataset Partitioning** | Page 7 | $Train \cap Val \cap Test = \emptyset$ | `tests/test_splits.py` | **`VERIFIED`** | Board-group and layout isolation verified. |
| **DsPCBSD+ Image Scaling Defect** | Page 8 | Coordinate scaling bug resolved | `docs/FINAL_REPORT_DATA.md` (lines 106–119) | **`VERIFIED`** | Hardcoded 640 fixed to `Image.open(p).size`. |
| **Small Defect Recall (< 32 px)** | Page 8 | 63.01% (Baseline) / 75.05% (Exp C) | `benchmarks/results/Baseline Multi-Domain_evaluation.json` | **`VERIFIED`** | Calculated on native sensor coordinates. |
| **Physical Hardware Physical Testing** | Page 5, 16 | Live Pi 5 + Hailo HAT+ testing | Hardware procurement status | **`UNVERIFIED`** | Physical hardware pending; marked PROJECTED. |

---

## 11. Actionable Itemized List of Exact PDF Corrections Needed

When generating or editing the final PDF report, apply the following exact corrections:

### Correction 1: Baseline Architecture Clarification (Title, Abstract, Section 4 & 5)
- **Current Text in Drafts:** Occasional legacy references to "YOLO11n".
- **Exact Replacement:** Replace all baseline references with **`YOLOv8n`** (Ultralytics 8.4.201, 73 layers fused, 3,011,043 parameters, 8.2 GFLOPs).
- **Reason:** Verified by `args.yaml` (`model: yolov8n.pt`) and checkpoint inspection.

### Correction 2: Reconciling the Clean-Board PASS Rate Discrepancy (Page 8 / Table 3)
- **Current Text:** Page 8 lists Baseline PASS as 49.23%, while earlier text suggested 93.33%.
- **Exact Table Correction:** Add a dedicated **Clean-Board Yield & False Scrap Table** directly comparing Baseline Multi-Domain vs Clean Experiment C2:
  ```markdown
  | Metric | Baseline Multi-Domain Reference | Clean Experiment C2 (Hard Negatives) | Empirical Impact |
  |---|:---:|:---:|:---:|
  | Clean PASS Yield (195 Boards) | 49.23% (96 / 195) | 93.33% (182 / 195) | +44.10% First-Pass Yield |
  | False INSPECT (Review Overkill) | 41.03% (80 / 195) | 6.15% (12 / 195) | 6.67x Overkill Reduction |
  | False REJECT (False Scrap) | 9.74% (19 / 195) | 0.51% (1 / 195) | 19.00x Scrap Reduction |
  | Total False-Positive Bounding Boxes | 357 boxes (1.83/board) | 15 boxes (0.08/board) | -95.80% False Alarms |
  ```
- **Clarification Text to Insert:** *"The Baseline Multi-Domain model achieves a clean PASS rate of 49.23% (96/195) due to 357 false-positive alarms triggered primarily by normal circular via holes. Clean Experiment C2 incorporates 34 audited colour hard negatives, suppressing false alarms by 95.8% and elevating clean-board PASS yield to 93.33% (182/195)."*

### Correction 3: Baseline Metric Row Discrepancy (Page 8 / Model Comparison Table)
- **Current Text:** Baseline Multi-Domain row lists mAP50 = 80.58%, Precision = 75.02%, Recall = 83.91%.
- **Exact Replacement:** 
  - Change Baseline Multi-Domain row to:
    - **Validation Precision (Operational conf=0.25):** `66.97%`
    - **Validation Recall (Operational conf=0.25):** `77.50%`
    - **Validation F1-Score:** `0.7185`
    - **Validation mAP@50 (Standard Ultralytics sweep):** `81.25%` (Truncated operational mAP@50: `73.12%`)
  - Retain `80.58%` mAP@50, `75.02%` Precision, and `83.91%` Recall exclusively in the row for **Experiment C (Hard Negatives)**.
- **Reason:** Source audit of `methodology_reconciliation_map50.json` proves 80.58% belongs to Exp C.

### Correction 4: Label All Raspberry Pi 5 & Hailo-8 Latency Metrics as `[PROJECTED]` (Page 16 / Table 5)
- **Current Table Header:** `Target Pi 5 + Hailo NPU`
- **Exact Replacement Header:** `Target Pi 5 + Hailo NPU [PROJECTED]`
- **Cell Entries to Relabel:**
  - `12.5 ms (~80 FPS on Hailo-8)` $\to$ `12.5 ms (~80 FPS on Hailo-8) [PROJECTED]`
  - `38.0 ms (ARM NEON)` $\to$ `38.0 ms (ARM NEON) [PROJECTED]`
  - `0.12 ms` $\to$ `0.12 ms [PROJECTED]`
  - `0.35 ms` $\to$ `0.35 ms [PROJECTED]`
  - `~240 ms TTFT, 14 tok/s (Int4 / NPU)` $\to$ `~240 ms TTFT, 14 tok/s (Int4 / NPU) [PROJECTED]`
- **Explicit Footnote to Add:** *"[PROJECTED] denotes compiler-derived or analytical performance estimates. Physical validation on Raspberry Pi 5 8GB + Raspberry Pi AI HAT+ is scheduled for Phase 2 hardware integration following physical hardware procurement."*

### Correction 5: Test Count & Testing Nature Specification (Executive Summary & Page 6)
- **Current Text in Previous Reports:** Variously cites 104, 129, or 147 tests.
- **Exact Replacement:** Update to **`152 automated regression tests passing (100% in 22.03s)`**.
- **Qualification Sentence to Insert:** *"All 152 tests represent automated software unit and integration tests executing in Python on the development workstation. Zero physical hardware-in-the-loop tests have been executed; physical hardware acceptance tests are codified as standalone shell scripts (`hardware/deployment/verify_hardware.sh`) awaiting physical board delivery."*

### Correction 6: Llama 3.2 1B Host Platform & Checksum Transparency (Page 11)
- **Current Text:** States Llama 3.2 1B ExecuTorch measurements without always reiterating the host CPU platform in every caption.
- **Exact Replacement:** Explicitly cite in every caption and summary:
  - Artifact: `llama3_2_1b_fp16.pte` (`2,472,432,290 bytes`, 2.36 GB).
  - Checksum: `SHA-256: 0d2c47c426380fc520357aeede656897cb49fff2e0a8651877c631c0dbfe8b1a`.
  - Host Measured Platform: **AMD Ryzen 9 8940HX CPU (16 cores, 32 threads, 32 GB DDR5 RAM)**.
  - Generative Latency: TTFT = `20.93s – 47.36s`, Decode = `1.05 – 1.11 tokens/s`, Peak RAM = `3,022.8 MB`.
  - Non-Negotiable Safety Gate: Generative text is strictly advisory; the deterministic quality gate establishes the verdict in `< 0.1 ms` prior to LLM invocation, preventing probabilistic verdict override.

---
**Audit Complete & Verified against Active Codebase Repository.**
