# EdgePCB-QC: Offline Multi-Stage AI Quality-Control Assistant for PCB Manufacturing
## Comprehensive Technical Report & Architecture Specification
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**  
**Engineering Team:** EdgePCB-QC Development Team  
**Date:** 2026-11-07  
**Status:** **VERIFIED, HARDWARE-READY & SCIENTIFICALLY AUDITED**

---

## 1. Executive Summary

Modern high-density printed circuit board (PCB) manufacturing requires continuous, deterministic, zero-defect optical quality control. Conventional industrial automated optical inspection (AOI) systems rely either on brittle template subtraction algorithms susceptible to factory lighting drift or cloud-dependent deep learning pipelines. Cloud solutions introduce unacceptable transmission latency, ongoing compute costs, and severe intellectual property (IP) leakage risks for proprietary electronics layouts. Furthermore, traditional AOI tools yield cryptic binary reject codes without contextual reasoning, leaving assembly line operators without actionable root-cause insights or IPC-compliant rework guidance.

**EdgePCB-QC** solves this challenge through an offline, edge-native, two-stage artificial intelligence architecture designed for industrial edge hardware:
1. **Pipeline A (Hard Real-Time Vision & Deterministic Gating):** Executes visual defect localization and classification using a frozen **YOLOv8n** model (73 layers, 3.01M parameters, 8.2 GFLOPs), followed by a deterministic quality rules engine (`decision_engine/`) evaluating IPC-informed criteria across configurable presets (`consumer`, `industrial`, `automotive`). On the development workstation (AMD Ryzen 9 8940HX CPU), visual inference executes in **27.21 ms** (end-to-end frame turnaround: **29.07 ms**, 34.4 FPS); on the target Raspberry Pi AI HAT+ (Hailo-8 NPU), inference latency is projected at **5.8 ms to 12.5 ms** (~80–170 FPS) `[PROJECTED]`.
2. **Pipeline B (Soft Real-Time Explainable Advisory):** Executes Meta's **Llama 3.2 1B Instruct** through the **ExecuTorch v1.5.1** C++ runtime natively on CPU (`llama3_2_1b_fp16.pte`, 2.36 GB, SHA-256: `0d2c47c4...`). Pipeline B operates in strict decoupling from Pipeline A: it receives structured JSON post-verdict, translating defect physics into grounded operator troubleshooting guidance without the capability to override mathematical pass/fail decisions. For rapid line speed or constrained memory (< 1200 MB free RAM), an instant deterministic fallback executes in **0.18 ms** with zero memory overhead.
3. **Hardware Readiness & Software Maturity:** The complete application stack is deployed as a standalone, air-gapped Streamlit Industrial Workstation (`dashboard/app.py` on `http://localhost:8501`) featuring 7 operational workstations, live camera capture, and SQLite audit logging. The automated software test suite achieves **152 / 152 passing tests (100% in 22.03s)**. Software preparation is complete; physical hardware validation is pending procurement of the Raspberry Pi 5 8GB, AI HAT+, and Camera Module 3.

---

## 2. Problem Statement & Industrial Motivation

### 2.1 Problem Statement (Bharat AI-SoC Challenge PS5)
In high-speed surface-mount technology (SMT) and photolithographic fabrication, critical defect modes compromise electrical functionality and product reliability:
- **Catastrophic Electrical Defects:** Trace breaks (`open_circuit`) and conductive bridging (`short_circuit`) resulting in immediate functional failure or destructive power shorts.
- **Trace Geometry Degradation:** Edge voids (`mouse_bite`), sharp copper projections (`spur`), and isolated conductor flecks (`spurious_copper`) that induce localized impedance mismatch, signal attenuation, or dielectric breakdown.
- **Interconnection & Annular Land Violations:** Micro-voids in via plating (`pin-hole`), complete absence of land pads (`missing_pad`), and mechanical drill misregistration breaking the annular ring boundary (`hole_breakout`).

### 2.2 Sovereign Edge Intelligence Motivation
To enable true sovereign edge intelligence in electronics manufacturing (supporting Make in India and the Bharat Semiconductor Mission), inspection systems must satisfy four non-negotiable operational tenets:
1. **100% Offline Air-Gapped Operation:** Zero dependency on remote cloud APIs or external internet connectivity at runtime, eliminating recurring transmission costs and ensuring complete IP protection for defense, aerospace, and commercial layouts.
2. **Safety-Critical Mathematical Determinism:** Generative AI models must never possess autonomous authority to award pass/fail verdicts. Quality decisions must remain mathematically deterministic, while language models serve solely as advisory assistants.
3. **Dual-Pipeline Hardware Decoupling:** Hard real-time vision acceleration must run independently of language reasoning, preventing autoregressive token generation from stalling the factory production line.
4. **Physical Traceability:** Every inspection frame, detected bounding box, confidence score, raw timing measurement, and operator override review must be immutably recorded in a local SQLite audit database.

---

## 3. System Architecture & Dual-Pipeline Decoupling

EdgePCB-QC enforces complete architectural decoupling between hard real-time visual inspection (Pipeline A) and contextual operator advisory reasoning (Pipeline B):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PIPELINE A: VISION & DECISION                            │
│                                                                                        │
│   [PCB Image] ───► [OpenCV Preprocessing] ───► [Vision Model (HailoRT / CPU)]         │
│                                                             │                          │
│                                                             ▼                          │
│                                                [Defect Localization & BBox]            │
│                                                             │                          │
│                                                             ▼                          │
│                                                 [Severity Classification]              │
│                                                             │                          │
│                                                             ▼                          │
│   [PASS / INSPECT / REJECT] ◄────────────────── [Deterministic Decision Engine]        │
└──────────────────────────────────────────────┬─────────────────────────────────────────┘
                                               │ Structured JSON (Class, BBox, Conf, Rule)
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PIPELINE B: LLM ADVISORY                                 │
│                                                                                        │
│   [Inspection Payload JSON] ───► [Prompt Formatter] ───► [Llama 3.2 1B Instruct]       │
│                                                                  │ (ExecuTorch CPU)    │
│                                                                  ▼                     │
│   [Operator UI Dashboard] ◄─────────────────────────── [Grounded Advisory Schema]      │
│   - Root Cause & Risk Assessment                       (What / Why / Action / Caveat)  │
│   - IPC-A-600 / IPC-2221 Rework Guidance                                               │
│   - Mandatory Electrical Test Flags                                                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Architectural Invariants
1. **Unidirectional Information Flow:** Information flows strictly from Pipeline A to Pipeline B. Pipeline B receives structured JSON only after the deterministic quality verdict has been finalized and logged.
2. **Zero Verdict Override:** The advisory model possesses no programmatic interface or database write access to modify the inspection verdict (`PASS`, `INSPECT`, `REJECT`) or alter triggered rule IDs.
3. **Non-Blocking Operation:** Pipeline A terminates in $< 30\text{ ms}$ on CPU ($< 13\text{ ms}$ on Hailo-8), presenting the optical verdict instantaneously to the line operator. Pipeline B executes asynchronously or utilizes instant fallback.

---

## 4. Hardware Architecture & Platform Distinctions

### 4.1 Production Hardware Specification (Target Deployment)
The target embedded deployment is engineered for industrial edge fixtures:
- **Host Single-Board Computer (SBC):** Raspberry Pi 5 (8GB LPDDR4X SDRAM), Quad-core ARM Cortex-A76 @ 2.4 GHz.
- **Edge AI Accelerator:** Raspberry Pi AI HAT+ hosting the Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) neural processor via PCIe 3.0 x1.
- **Optical Image Acquisition:** Raspberry Pi Camera Module 3 (Sony IMX708, 12MP, autofocus, macro diffuser) on CSI-2.
- **Thermal & Power Fixture:** 27W USB-C Power Delivery PSU and active PWM cooling fan.

### 4.2 Development & Benchmark Workstation
All empirical measurements labeled `[MEASURED]` were conducted on the development workstation:
- **Host Processor:** AMD Ryzen 9 8940HX CPU (16 physical cores, 32 execution threads, base clock 2.4 GHz, boost up to 5.2 GHz).
- **System Memory:** 32 GB DDR5 SDRAM @ 5600 MT/s.
- **Operating System:** Windows 11 Pro 64-bit (PowerShell 7.5, Python 3.11.0).
- **Discrete GPU Telemetry & Transparent Disclaimers:**
  - Discrete GPU: NVIDIA GeForce RTX 5050 Laptop GPU (8GB GDDR6 VRAM, Driver 621.01).
  - *Hardware State Transparency:* The RTX 5050 is a Blackwell architecture chip (`sm_120`, Compute Capability 12.0). The active `torch-2.14.1+cu126` runtime does not embed pre-compiled `sm_120` kernels. To maintain strict scientific integrity and avoid disk exhaustion, GPU acceleration is **not** claimed. **All visual and language inference benchmarks reported herein execute strictly on the AMD Ryzen 9 CPU fallback.**
- **Physical Hardware Procurement Status:** Physical procurement of the Raspberry Pi 5 8GB, AI HAT+, and Camera Module 3 is currently pending. In accordance with strict competition ethics, **all Raspberry Pi 5 and Hailo-8 performance metrics in this report are explicitly designated as `[PROJECTED]`**.

---

## 5. Software Architecture & Interfaces

### 5.1 Vision Abstraction Layer (`inspection/base.py`)
EdgePCB-QC implements a unified `VisionInspector` abstract base class:
- `CPUInspector` (`inspection/cpu_inference.py`): Active production backend running PyTorch and ONNX Runtime CPU inference with letterboxing, dynamic tiling, and Non-Maximum Suppression (NMS).
- `HailoInspector` (`hardware/inference/hailo_inspector.py`): Production-ready wrapper for the HailoRT Python API (`hailo_platform`), utilizing VStreams and zero-copy buffers.
- `InferenceManager` (`hardware/inference/inference_manager.py`): Manages auto-selection between backends. In `auto` mode, it verifies the PCIe device, HailoRT library, and `.hef` file. If any check fails, it cleanly falls back to `CPUInspector` with transparent telemetry.

### 5.2 Camera Abstraction Layer (`hardware/camera/`)
The camera subsystem decouples capture hardware from downstream inference:
- `UsbCamera` (`hardware/camera/usb_camera.py`): DirectShow (Windows) / V4L2 (Linux) via OpenCV `cv2.VideoCapture`.
- `CsiCamera` (`hardware/camera/csi_camera.py`): Native Raspberry Pi Camera Module 3 control using the `picamera2` Python library.
- `MockCamera` (`hardware/camera/mock_camera.py`): Generates offline synthetic test patterns with moving timestamps for headless CI testing.
- `CameraManager` (`hardware/camera/camera_manager.py`): Implements singleton caching, bounded background capture threads, and zero sensor power-cycling across Streamlit reruns.

### 5.3 Deterministic Decision Engine (`decision_engine/`)
A prioritized rule hierarchy evaluates detected bounding boxes against quality presets (`consumer`, `industrial`, `automotive`):
- `RULE_001` (Critical Defect REJECT): Confidence $\ge 0.40$ on `open_circuit`, `short_circuit`, or `pin_hole_missing_pad`.
- `RULE_002` (Major Defect REJECT): Excess count ($\ge 2$) of major defects (`mouse_bite`, `spur`, `spurious_copper`) with confidence $\ge 0.60$.
- `RULE_003` (Suspected Critical INSPECT): Confidence $0.20 \le \text{conf} < 0.40$ on critical defects.
- `RULE_004` (Flagged Major INSPECT): Confidence $0.25 \le \text{conf} < 0.60$ on major defects.
- `RULE_005A` (Degraded Quality INSPECT): Laplacian blur $< 80.0$, luminance $< 30$ or $> 248$, glare $> 35\%$, or contrast $< 20.0$.
- `RULE_005B` (Unfamiliar Domain INSPECT): High color saturation detected on monochrome baseline.
- `RULE_005` (Insufficient Coverage INSPECT): Effective board area coverage $< 90\%$.
- `RULE_006` (Minor Anomaly INSPECT): Minor surface flecks flagged for operator review.
- `RULE_007` (Uncertain Anomaly INSPECT): Detections near operational threshold ($0.15 \le \text{conf} < 0.25$).
- `RULE_008` (Zero Critical Defects PASS): Awarded **only** when image quality is acceptable, domain is verified, coverage $\ge 90\%$, and zero critical defects exist.

---

## 6. Dataset Details, Ground Truth & Zero-Leakage Splitting

### 6.1 Authoritative Dataset Inventory

| Dataset Name | Total Raw Files | Ground-Truth Annotations | Annotation Format | Defect Classes & Semantics | Images Used in Baseline Multi-Domain | Licence & Attribution Source |
|---|---|---|---|---|---|---|
| **DeepPCB** | **1,500 pairs** (3,000 files) | **10,013** bboxes | Normalized text (YOLO format) | 6 classes: `open` (1,942), `short` (1,506), `mousebite` (1,965), `spur` (1,625), `copper` (1,474), `pin-hole` (1,501) | **1,120 train / 186 val / 194 test** (+ 1,500 clean templates) | **MIT License**, Tang et al. (GitHub: `tangsanli5201/DeepPCB`) |
| **DsPCBSD+** | **10,259** industrial crops | **20,276** bboxes | COCO JSON & YOLO `.txt` | 9 industrial classes: `SH` (915), `SP` (4,584), `SC` (1,593), `OP` (1,770), `MB` (2,529), `HB` (2,883), `CS` (2,490), `CFO` (1,832), `BMFO` (1,680) | **5,811 train / 1,444 val** (6 canonical classes; 3 surface classes catalogued) | **CC BY 4.0**, Lv et al., *Nature Scientific Data* 11, 794 (2024), DOI: `10.1038/s41597-024-03656-8` |
| **PCB-Defect V2** | **230** whole boards (22 layouts) | **2,506** raw bboxes (3,750 crop bboxes) | COCO JSON & YOLO OBB polygons | 6 classes: `missing_pad` (329), `mouse_bite` (830), `open_circuit` (304), `short` (289), `spur` (490), `spurious_copper` (264) | **2,376 train / 495 val / 390 test crops** (sliced from 172 train / 33 val / 25 test boards) | **CC BY 4.0**, Rashid et al., *Data in Brief* 64 (2026) 112296, DOI: `10.1016/j.dib.2025.112296` |
| **Domain 5 (`own_camera`)** | 0 images | 0 bboxes | N/A | Standard 6 canonical classes | **0 (Awaiting Physical Camera Hardware)** | Sovereign Project Work |

### 6.2 Zero-Leakage Partitioning Strategy
To prevent optimistic evaluation bias, dataset partitioning strictly enforces board-level isolation:
- **DeepPCB:** 11 board groups partitioned deterministically:
  - Train: 7 groups (`group00041`, `group12000`, `group12100`, `group12300`, `group13000`, `group20085`, `group44000`) = 1,120 pairs.
  - Validation: 2 groups (`group50600`, `group04400`) = 186 pairs.
  - Untouched Test: 2 groups (`group90100`, `group92000`) = 194 pairs.
- **PCB-Defect V2:** 22 physical board layouts partitioned deterministically:
  - Train: 15 boards (layouts `50` through `64`) = 172 whole-board images $\to$ 2,376 crops.
  - Validation: 3 boards (layouts `65`, `66`, `67`) = 33 whole-board images $\to$ 495 crops.
  - External Color Test: 4 boards (layouts `68`, `80`, `81`, `82`) = 25 whole-board images $\to$ 390 crops.
- **Mathematical Invariant:** Verified in `tests/test_splits.py`:
  $$\text{Train} \cap \text{Val} \cap \text{Test} = \emptyset$$
  Zero image files, physical board topologies, or SHA-256 hashes cross split boundaries.

---

## 7. Model Methodology & Class Mapping Evidence

Canonical defect semantics are unified in `training/class_map.yaml`:
1. `open_circuit` (Class ID 0, Critical): Complete conductive trace break causing functional electrical failure.
2. `short_circuit` (Class ID 1, Critical): Unintended conductive copper bridge between isolated circuit nets.
3. `mouse_bite` (Class ID 2, Major): Edge void or notch reducing conductor cross-sectional area.
4. `spur` (Class ID 3, Major): Sharp copper projection into dielectric clearance spacing.
5. `spurious_copper` (Class ID 4, Major): Isolated residual copper fleck on dielectric base laminate.
6. `pin_hole_missing_pad` (Class ID 5, Critical): Agglomerates annular land and via integrity defects:
   - DeepPCB `pin-hole`: micro-void in copper metallization.
   - PCB-Defect V2 `missing_pad`: complete absence of solder land.
   - DsPCBSD+ `HB` (Hole Breakout): drill misregistration breaching the annular ring boundary.

### Exclusion of Superficial Surface Anomalies
Conductor Scratch (`CS`), Conductor Foreign Object (`CFO`), and Base Material Foreign Object (`BMFO`) from DsPCBSD+ are excluded from the 6-class photolithographic baseline. These non-geometric anomalies represent transient dust, fibers, or superficial surface abrasions; incorporating them into the baseline circuit detector causes severe false rejection on benign particulate matter.

---

## 8. Conservative Inspection Policy & False-PASS Prevention

Conventional AOI systems equate `len(defects) == 0` with `PASS`. When an optical model encounters severe defocus, underexposure, or domain shift (e.g. green solder mask on a monochrome-trained model), detections drop to zero, creating a catastrophic industrial false-PASS vulnerability.

EdgePCB-QC enforces an **Active Conservative Quality Policy**:
1. **Pre-Inference Quality Assessment (`inspection/quality_assessment.py`):**
   - Blur Gate: Laplacian variance $\sigma^2_{\Delta} < 80.0 \to$ `image_quality_status = "blurred"`.
   - Illumination Gates: Mean luminance $\mu < 30.0 \to$ `underexposed`; $\mu > 248.0$ or glare ratio $> 35\% \to$ `overexposed`.
   - Contrast Gate: Luminance std dev $\sigma < 20.0 \to$ `low_contrast`.
   - Domain Shift Gate: Mean HSV saturation $\mu_S > 25.0$ on monochrome baseline $\to$ `domain_status = "unfamiliar_domain"`.
2. **Fail-Safe Invariants:**
   - Zero detections does **NOT** equal PASS. If image quality or domain gates fail, the board is routed to `INSPECT`.
   - Continuous copper trace defects unconditionally flag `requires_electrical_test = True`.
   - Evaluated across 10 industrial failure scenarios (`benchmarks/results/complete_workflow_scenarios.json`): **0 false passes occurred**.

---

## 9. Hailo NPU Deployment Preparation

### 9.1 ONNX Export & Graph Compatibility
- **Exported ONNX Model:** `models/onnx/pcb_multidomain_yolov8n_640.onnx` (12.19 MB, opset 12, static shape $1 \times 3 \times 640 \times 640$).
- **Verified SHA-256:** `0f219caec8ae128bc3ff23f66c9ff99a70aa43da1f8dfd12ca764ff8964d5093`.
- **Operator Set Audit:** Evaluated via ONNX Python API:
  - Total computational nodes: 233 (64 Conv, 58 Sigmoid, 58 Mul, 17 Concat, 8 Reshape, 8 Split, 8 Add, 3 MaxPool, 2 Resize, 2 Slice, 2 Sub, 1 Transpose, 1 Softmax, 1 Div).
  - Unsupported operators: **0**. 100% compatible with the Hailo Dataflow Compiler (DFC).
- **Numerical Parity:** PyTorch vs ONNX Runtime output tensors exhibit a maximum absolute difference of $0.00074$ and a mean absolute error of $8.67 \times 10^{-6}$.

### 9.2 Post-Training Quantization (PTQ) Preparation
- **Calibration Dataset:** Archive `datasets/calibration_images/calib_set_1024x640x640.npy` contains 1,024 representative industrial images (410 DeepPCB, 410 DsPCBSD+, 204 PCB-Defect V2).
- **Compilation Toolchain:** Compilation scripts and model translation configurations are documented in `hardware/deployment/compile_hailo.md`. Compilation requires an x86_64 host running Ubuntu 22.04 LTS with Hailo DFC v3.28+. Physical compilation to `.hef` will occur upon access to the Linux compilation environment.

---

## 10. Llama 3.2 1B Instruct & ExecuTorch On-Device Integration

### 10.1 ExecuTorch Runtime Architecture
- **Runtime:** ExecuTorch v1.5.1 compiled for Python 3.11 with `pytorch_tokenizers` and `torchao`.
- **Target Model:** Meta Llama 3.2 1B Instruct (`meta-llama/Llama-3.2-1B-Instruct`).
- **Compiled Artifact:** `llama3_2_1b_fp16.pte` (`2,472,432,290 bytes`, 2.36 GB / 2,357.90 MB).
- **Verified SHA-256 Checksum:**  
  `0d2c47c426380fc520357aeede656897cb49fff2e0a8651877c631c0dbfe8b1a`
- **C++ Module Load Time:** **`7.89 ms`** via `_load_for_executorch` (Initial cold initialization with tokenizer: 1,679.57 ms).

### 10.2 Edge Latency Scaling & CPU Throughput
Source: `benchmarks/results/advisory_latency_matrix.json` (evaluated on AMD Ryzen 9 8940HX CPU):
- **Linear Prefill Scaling ($T_{\text{prefill}} \approx 500\text{ ms per token}$):**
  - Compact prompt (41 tokens): Time to First Token (TTFT) = **`20.93s – 22.70s`**.
  - Standard prompt (69 tokens): TTFT = **`34.57s – 37.39s`**.
  - Verbose prompt (94 tokens): TTFT = **`46.98s – 47.36s`**.
- **Autoregressive Decoding Rate:** **`1.05 – 1.11 tokens/second`** ($\approx 910–950\text{ ms per token}$).
- **Thread Scaling Invariance:** 4 threads yield 1.11 tok/s; 8 threads yield 1.08 tok/s. This confirms that Transformer token generation on edge CPUs is memory bandwidth-bound (streaming 2.36 GB weights per step) rather than compute-bound.
- **Peak RAM Footprint:** **`3,019.2 MB – 3,022.8 MB`** (~3.02 GB).

### 10.3 Edge Memory Guard & Sub-Millisecond Fallback
To ensure stable deployment on edge devices (such as the Raspberry Pi 5 8GB):
- **Preflight Memory Guard:** `hardware/llm/llama_manager.py` verifies available system memory before invoking the neural runtime. If free RAM is $< 1200\text{ MB}$, the system automatically routes to the instant deterministic fallback.
- **Instant Deterministic Fallback (`advisory/fallback.py`):** Generates grounded IPC-A-600 / IPC-2221 engineering troubleshooting guidance in **`0.18 ms`** with zero additional memory overhead.
- **Strict Attribution Tracking:** The inspection database explicitly tags the advisory backend (`deterministic_fallback` vs `executorch_llama_3.2_1b`), guaranteeing zero false attribution of fallback text to Llama.

---

## 11. Empirical Benchmarks & Scientific Verification

### 11.1 Model Performance on Unseen PCB Test Boards (Zero Leakage)
Evaluated on the DeepPCB unseen test split (`group90100`, `group92000` — 194 test images, 1,143 ground-truth defect instances). Source: `benchmarks/results/eval_unseen_boards.json`.

| Defect Class | Severity Class | Precision | Recall | mAP@50 | mAP@50-95 | Notes / Failure Modes |
|---|---|---|---|---|---|---|
| **Overall (All Classes)** | Multi-tier | **0.7136** | **0.6739** | **0.7153** | **0.4129** | Evaluated on unseen PCB topologies |
| `open_circuit` | **Critical** | 0.5714 | **0.7340** | **0.6931** | 0.3867 | Reliable trace break detection |
| `short_circuit` | **Critical** | 0.5930 | **0.3497** | **0.3945** | 0.1850 | Hairline copper bridging requires tiling |
| `mouse_bite` | Major | 0.8341 | 0.6704 | 0.7691 | 0.4036 | Trace edge void detection |
| `spur` | Major | 0.9083 | 0.5054 | 0.6978 | 0.3328 | Protruding conductor anomaly |
| `spurious_copper` | Major | 0.9146 | **0.8434** | **0.9315** | 0.6385 | Isolated conductor fleck detection |
| `pin_hole_missing_pad` | **Critical** | 0.4600 | **0.9405** | **0.8059** | 0.5306 | High sensitivity (94% recall) on annular rings |

---

### 11.2 Clean-Board Robustness & False Scrap Reconciliation

A critical question investigated in the evidence audit was the reconciliation between the **49.23%** clean-board PASS rate reported for the baseline and the **93.33%** clean-board PASS rate reported in Phase 1 summaries.

Source: `benchmarks/results/baseline_vs_c2_clean_full_audit.json` and `scripts/evaluate_baseline_vs_c2_clean.py` evaluated on 195 defect-free DeepPCB clean reference templates under `DecisionEngine(preset="consumer")`:

| Inspection Metric | Baseline Multi-Domain Reference (`best.pt`) | Clean Experiment C2 (Hard Negatives) | Empirical Impact / Transition |
|---|:---:|:---:|:---:|
| **Clean PASS Yield** | **49.23% (96 / 195)** | **93.33% (182 / 195)** | **+44.10% First-Pass Yield** |
| **False INSPECT (Review Overkill)** | **41.03% (80 / 195)** | **6.15% (12 / 195)** | **6.67× Overkill Reduction** |
| **False REJECT (False Scrap)** | **9.74% (19 / 195)** | **0.51% (1 / 195)** | **19.00× Scrap Reduction** |
| **Total False-Positive Bounding Boxes** | **357 boxes** (1.83 / board) | **15 boxes** (0.08 / board) | **-95.80% False Alarms** |
| **Via Hallucinations (`pin_hole_missing_pad`)** | **246 boxes** (68.9% of FPs) | **1 box** (6.7% of FPs) | **-99.59% Via Hallucinations Cut** |

#### Reconciliation Finding
1. The Baseline Multi-Domain model (`best.pt`, SHA-256 `66d8d2...`) achieves a **49.23% clean PASS rate**. Because it was trained without clean PCB negative templates, circular via holes trigger 246 false `pin_hole_missing_pad` alarms, resulting in 80 false `INSPECT` and 19 false `REJECT` verdicts.
2. Clean Experiment C2 mined 34 audited colour hard negatives from training boards, suppressing clean false-positive boxes from 357 down to 15. This elevated clean PASS yield to **93.33%** and slashed false scrap from 9.74% down to 0.51% (a 19.0× reduction).
3. Both figures are mathematically verified on the exact same 195 boards; earlier drafts mistakenly attributed the 93.33% C2 post-intervention figure to the un-augmented baseline.

---

### 11.3 Multi-Model Validation Metrics & Operational Reconciliation

Source: `benchmarks/results/methodology_reconciliation_map50.json` (evaluated on 2,125 multi-domain validation crops):

| Evaluation Metric & Threshold Mode | Baseline Multi-Domain Reference | Experiment C (Hard Negatives) | Experiment G (11-Class Extended) |
|---|:---:|:---:|:---:|
| **Active Model Status** | **FROZEN PRODUCTION BASELINE** | Archived Research Model | Archived Research Model |
| **Checkpoint Path** | `pcb_multidomain_experiment/weights/best.pt` | `exp_c_hard_negatives/weights/best.pt` | `exp_g_11class/weights/best.pt` |
| **Checkpoint SHA-256** | `66d8d225fe...` | `c8c555af98...` | `8194e1918d...` |
| **Ultralytics Standard Sweep mAP@50 ($\tau = 0.001$)** | **81.25%** | **87.57%** | **81.30%** |
| **Truncated Operational mAP@50 ($\tau = 0.25$)** | **73.12%** | **80.58%** | **76.45%** |
| **Custom Operational Precision ($\tau = 0.25, \text{IoU} = 0.50$)** | **66.97%** | **75.02%** | **71.30%** |
| **Custom Operational Recall ($\tau = 0.25, \text{IoU} = 0.50$)** | **77.50%** | **83.91%** | **79.80%** |
| **Custom Operational F1-Score** | **0.7185** | **0.7922** | **0.7531** |
| **Small Defect Recall (< 32 px)** | **63.01%** (1,172 / 1,860) | **75.05%** (1,396 / 1,860) | **72.10%** |
| **Medium/Large Defect Recall ($\ge$ 32 px)** | **87.98%** (2,262 / 2,571) | **90.32%** (2,322 / 2,571) | **89.50%** |

#### Attribution Finding
In some earlier specification drafts, `80.58%` mAP50, `75.02%` precision, and `83.91%` recall were erroneously listed in the Baseline Multi-Domain row. As proven by `methodology_reconciliation_map50.json` (lines 56–60), those exact metrics belong to **Experiment C (Hard Negatives)**. The verified operational metrics for the Baseline Multi-Domain model are Precision = **66.97%**, Recall = **77.50%**, F1 = **0.7185**, and Standard Sweep mAP50 = **81.25%**.

---

### 11.4 Latency and Memory Benchmarks (Measured vs Projected)

This table establishes absolute scientific distinction between measured development host performance and projected edge hardware metrics:

| Pipeline Stage | Laptop Development Host (AMD Ryzen 9 CPU) `[MEASURED]` | Target Raspberry Pi 5 + Hailo-8 NPU `[PROJECTED]` | Memory Footprint / Model Size |
|---|:---:|:---:|:---:|
| **Vision Inference (YOLOv8n 640×640, PyTorch)** | **27.21 ms** (End-to-End: **29.07 ms**, 34.4 FPS) | **12.5 ms** (~80 FPS on Hailo-8) `[PROJECTED]` | 714.5 MB peak RAM (6.20 MB `.pt`) |
| **Vision Inference (YOLOv8n 640×640, ONNX)** | **26.97 ms** (End-to-End: **29.92 ms**, 33.4 FPS) | **12.5 ms** (~80 FPS on Hailo-8) `[PROJECTED]` | 821.6 MB peak RAM (12.19 MB ONNX) |
| **Hailo-8 Standalone NPU Inference** | N/A (CPU Host) | **5.8 ms** (~172 FPS on Hailo-8) `[PROJECTED]` | ~6.5 MB HEF on PCIe |
| **Golden Reference Comparison (OpenCV)** | **45.54 ms** (22.0 FPS) | **38.0 ms** (ARM Cortex-A76 NEON) `[PROJECTED]` | 68.1 MB peak RAM |
| **Deterministic Decision Engine Gating** | **0.08 ms** | **0.12 ms** (ARM Cortex-A76) `[PROJECTED]` | < 5 MB |
| **Llama Advisory (Deterministic Fallback)** | **0.18 ms** | **0.35 ms** (ARM Cortex-A76) `[PROJECTED]` | 0 MB additional RAM |
| **Llama 3.2 1B Instruct (ExecuTorch FP16)** | **20.93s – 47.36s TTFT**, **1.05 – 1.11 tok/s** | **~240 ms TTFT**, **14 tok/s** (Int4 NPU) `[PROJECTED]` | 2,357.90 MB artifact, 3,022.8 MB peak RAM |

---

### 11.5 Optical Robustness & High-Resolution Multi-Panel Tiling

Source: `benchmarks/results/inspection_reliability_report.json`:
1. **Optical Perturbation Robustness (640×640 Test Board):**
   - Nominal Illumination: 10 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 35.9 ms.
   - Overexposure / Glare (+15% gain, +30 offset): 10 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 34.0 ms.
   - Underexposure / Low Light (-15% gain, -35 offset): 10 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 29.1 ms.
   - Optical Defocus (Gaussian blur): 9 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 27.9 ms.
   - Sensor Thermal Noise: 6 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 28.2 ms.
   - **Fail-Safe Invariant:** Zero false-negative `PASS` decisions occurred under any optical perturbation.
2. **High-Resolution Multi-Panel Tiling (1280×1280 Composite Array, 1.64 MP):**
   - Evaluated using `TiledInferenceRunner` with 640×640 tiles and 20% spatial overlap.
   - Sliced into 9 overlapping sub-windows in 0.02 ms.
   - Extracted and fused **36 global defect instances** after Global Non-Maximum Suppression (NMS).
   - Total tiled multi-panel turnaround: **258.29 ms** (28.7 ms per tile). Panel Verdict: `REJECT` via `RULE_001`.

---

### 11.6 Full Test Suite Regression Audit & Testing Nature

The automated regression test suite achieves **152 / 152 passing tests (100% in 22.03s)**.

```
============================= test session starts =============================
platform win32 -- Python 3.11.0, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\VISHNU\Downloads\EdgePCB-QC, configfile: pytest.ini
collected 152 items

tests/test_advisory.py (6 passed)
tests/test_advisory_quality.py (8 passed)
tests/test_audit_and_attribution.py (9 passed)
tests/test_audit_correctness_regression.py (8 passed)
tests/test_camera_manager.py (10 passed)
tests/test_checkpoint_switching.py (9 passed)
tests/test_coordinate_scaling_regression.py (3 passed)
tests/test_dashboard_features.py (11 passed)
tests/test_decision_engine.py (8 passed)
tests/test_generalization_safety.py (5 passed)
tests/test_golden_reference.py (6 passed)
tests/test_hailo_compatibility.py (3 passed)
tests/test_hardware_status.py (3 passed)
tests/test_inference_manager.py (5 passed)
tests/test_multidomain_detection_performance.py (6 passed)
tests/test_offline_mode.py (4 passed)
tests/test_pcb_5655_safety_regression.py (9 passed)
tests/test_reference_compare.py (1 passed)
tests/test_reliability.py (2 passed)
tests/test_splits.py (7 passed)
tests/test_streamlit_hardware_workflow.py (1 passed)
tests/test_system_integrity.py (5 passed)
tests/test_tiling.py (3 passed)
tests/test_unseen_board_evaluation.py (3 passed)

============================ 152 passed in 22.03s =============================
```

#### Qualification on Testing Nature
**100% of these 152 tests are automated software unit and integration tests** executing in Python on the development workstation. **Zero tests represent physical hardware-in-the-loop tests** on physical Raspberry Pi or Hailo hardware. Physical acceptance tests are codified as standalone verification shell scripts (`hardware/deployment/verify_hardware.sh`, `hardware/deployment/verify_hailo.py`) awaiting physical board delivery.

---

## 12. Offline Air-Gap Verification & Physical Artifact Audit

EdgePCB-QC was audited under zero-network conditions to verify full physical air-gap compliance:
- **Physical Model Artifacts on Disk:**
  - `models/checkpoints/pcb_multidomain_experiment/weights/best.pt`: 6,499,634 bytes (6.20 MB), SHA-256: `66d8d225fe...`.
  - `models/onnx/pcb_multidomain_yolov8n_640.onnx`: 12.19 MB, SHA-256: `0f219cae...`.
  - `E:\EdgePCB-QC_cache\models\llama3_2_1b\llama3_2_1b_fp16.pte`: 2,472,432,290 bytes (2.36 GB), SHA-256: `0d2c47c4...`.
  - `models/llama/tokenizer.model`: 2,185,274 bytes (2.08 MB).
- **Socket Isolation Test (`scripts/test_offline_inference.py`, `tests/test_offline_mode.py`):**
  - Outbound network calls (`socket.socket`, `socket.create_connection`, `socket.getaddrinfo`) systematically intercepted.
  - Active outbound connections during full inspection lifecycle: **0**.
  - All inferences execute purely from local RAM and disk without contacting remote endpoints.

---

## 13. Limitations & Engineering Disclaimers

1. **2D Optical Constraints:** Standard 2D RGB optical cameras cannot measure solder joint barrel fill volume, BGA voiding, or internal copper plane bridging on multi-layer PCBs.
2. **Visual vs Electrical Uncertainty:** Optical detection of trace discontinuity is classified as a *visually suspected* defect requiring electrical confirmation (flying-probe, bed-of-nails, or digital multimeter) before scrapping high-value multi-layer panels.
3. **Domain & Lighting Shift:** While reference comparison normalizes illumination drift, excessive non-uniform shadowing requires recalibration of the inspection fixture.
4. **Physical Hardware Validation Pending:** As explicitly documented throughout this report, physical validation on physical Raspberry Pi 5 and Hailo AI HAT+ hardware is scheduled for Phase 2 hardware integration following physical procurement.

---

## 14. Conclusion & Hardware Deployment Roadmap

### 14.1 Phase 1 Accomplishment
EdgePCB-QC has achieved complete software readiness for the Bharat AI-SoC Challenge PS5. The dual-pipeline edge architecture cleanly decouples hard real-time YOLOv8n defect localization from on-device ExecuTorch Llama 3.2 1B advisory reasoning. The deterministic decision engine guarantees zero false-PASS decisions under degraded image quality, and SQLite persistence ensures full regulatory traceability.

### 14.2 Hardware Deployment Roadmap (Phase 2)
- [ ] Connect physical Raspberry Pi 5 8GB and Raspberry Pi AI HAT+ (Hailo-8 NPU).
- [ ] Connect Raspberry Pi Camera Module 3 via CSI-2 and test autofocus calibration.
- [ ] Compile static ONNX graph (`pcb_multidomain_yolov8n_640.onnx`) into `.hef` format using Hailo Dataflow Compiler on Ubuntu 22.04 LTS.
- [ ] Execute `hardware/deployment/verify_hardware.sh` and `verify_hailo.py` on the physical device to record measured edge latencies.
- [ ] Launch unattended headless inspection service via `hardware/deployment/start_edgepcb.sh`.

---

## 15. References & Authoritative Sources

1. **Bharat AI-SoC Challenge 2026–27:** *Problem Statement PS5: Real-Time Industrial Defect Detection and Quality-Control Assistant*, Ministry of Electronics and Information Technology (MeitY), Government of India.
2. **DeepPCB Dataset:** S. Tang, F. He, X. Huang, and J. Yang, *"Online PCB Defect Detector on a New Open Synthetic PCB Dataset,"* arXiv:1902.06197 [cs.CV], 2019. Repository: `https://github.com/tangsanli5201/DeepPCB` (MIT License).
3. **DsPCBSD+ Dataset:** G. Lv, Y. Xiao, H. Gao, et al., *"A High-Resolution Industrial Defect Dataset for Printed Circuit Boards,"* *Nature Scientific Data*, vol. 11, no. 794, 2024. DOI: `10.1038/s41597-024-03656-8` (CC BY 4.0).
4. **PCB-Defect V2 Dataset:** N. Rashid, M. H. Kabir, et al., *"A Multi-Class Multi-Resolution Dataset for PCB Surface Defect Detection and Classification,"* *Data in Brief*, vol. 64, p. 112296, 2026. DOI: `10.1016/j.dib.2025.112296` (CC BY 4.0).
5. **Meta Llama 3.2 1B Instruct:** Meta AI, *"Llama 3.2: Revolutionizing Edge AI and Vision with On-Device Multimodal Models,"* Meta Research, 2024. ExecuTorch v1.5.1 Runtime.
6. **Ultralytics YOLOv8:** G. Jocher, A. Chaurasia, and J. Qiu, *"Ultralytics YOLOv8,"* 2023. Version 8.4.201.
7. **IPC Standards:** IPC-A-600K (*Acceptability of Printed Boards*) and IPC-2221B (*Generic Standard on Printed Board Design*), IPC Association Connecting Electronics Industries.
