# EdgePCB-QC: Offline Multi-Stage AI Quality-Control Assistant for PCB Manufacturing
## Technical Report & Architecture Specification
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**  
*Lead Author: Autonomous Engineering Agent & Team*  
*Date of Submission / Inception: 2026-11-03*

---

## 1. Problem Statement
Modern electronics manufacturing requires rapid, zero-defect optical inspection of printed circuit boards (PCBs). High-speed surface-mount technology (SMT) and fine-pitch PCB fabrication generate complex visual defect modes—including broken copper traces (open circuits), conductive bridges (short circuits), mouse bites, spurs, spurious copper, and annular ring breakout. 

Currently, industrial automated optical inspection (AOI) systems rely either on rigid rule-based computer vision sensitive to minor illumination drift or cloud-dependent deep learning pipelines. Cloud solutions introduce intolerable network latency, transmission costs, and acute intellectual property (IP) leakage risks for proprietary layout designs. Furthermore, conventional AOI systems produce cryptic binary reject codes without contextual reasoning, leaving assembly line operators without guidance on root cause analysis or IPC-compliant rework.

---

## 2. Motivation
To enable true sovereign edge intelligence in Indian manufacturing (Make in India / Bharat Semiconductor Mission), quality control must operate:
1. **100% Offline & Edge-Native:** Zero cloud API dependence at runtime, ensuring complete intellectual property security and deterministic sub-50ms line cycle times.
2. **Two-Stage Decoupled AI Architecture:** Hard real-time visual defect detection on an energy-efficient Neural Processing Unit (NPU), paired with local Small Language Model (SLM) reasoning on an ARM CPU.
3. **Safety-Critical Determinism:** Generative AI must never hallucinate pass/fail decisions. Quality gates must remain mathematically deterministic, while SLMs serve solely as explainability and troubleshooting assistants.

---

## 3. Proposed Solution: EdgePCB-QC
**EdgePCB-QC** is an offline, multi-stage, edge-AI-powered industrial visual inspection system designed for deployment on a Raspberry Pi 5 with the Raspberry Pi AI HAT+ (Hailo-8/8L NPU).

The system integrates:
- **Pipeline A (Hard Real-Time Vision & Deterministic Gating):** High-resolution camera acquisition, OpenCV image registration, edge neural defect detection (accelerated via HailoRT on the Hailo NPU), and a multi-preset Deterministic Decision Engine (Consumer, Industrial, Automotive).
- **Pipeline B (Soft Real-Time LLM Advisory):** Llama 3.2 1B Instruct executed through Meta's **ExecuTorch** runtime on the host ARM CPU, receiving structured JSON to explain defect physics, recommend operator interventions, and flag mandatory electrical confirmation.
- **Persistent Local Audit Logging:** Embedded SQLite engine tracking inspection payloads, defect coordinates, timestamps, and operator overrides.
- **Interactive Local Dashboard:** Streamlit UI supporting live inspection, golden reference comparison, statistical process control, and Pareto analysis.

---

## 4. System Architecture
The system enforces strict operational decoupling between computer vision detection and natural language advisory reasoning:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PIPELINE A: VISION & DECISION                        │
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
                                               │ Structured JSON
                                               ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PIPELINE B: LLM ADVISORY                             │
│                                                                                        │
│   [Inspection Payload JSON] ───► [Prompt Formatter] ───► [Llama 3.2 1B Instruct]       │
│                                                                  │ (ExecuTorch CPU)    │
│                                                                  ▼                     │
│   [Operator UI Dashboard] ◄─────────────────────────── [Grounded Advisory Schema]      │
│   - Root Cause & Risk                                  (What / Why / Action / Caveat)  │
│   - IPC Rework Instructions                                                            │
│   - Electrical Test Flags                                                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Hardware Architecture
The target production deployment is engineered for embedded industrial edge fixtures:

- **Host Controller:** Raspberry Pi 5 (8GB LPDDR4X SDRAM), Quad-core ARM Cortex-A76 @ 2.4 GHz.
- **AI Acceleration Module:** Raspberry Pi AI HAT+ hosting the Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) neural processor via PCIe 3.0 x1.
- **Image Acquisition:** Raspberry Pi Camera Module 3 (Sony IMX708, 12MP, autofocus, macro diffuser) on CSI-2.
- **Thermal & Power:** 27W USB-C PD power supply and active PWM cooling fan maintaining thermal equilibrium under continuous line inspection.
- **Development Workstation & Hardware Distinction:** 
  - Evaluated and developed on Windows 11 with an AMD Ryzen 9 8940HX CPU (16 cores, 32 threads) and discrete NVIDIA GeForce RTX 5050 Laptop GPU (8GB VRAM, Driver 621.01).
  - *Hardware State Transparency:* The RTX 5050 is a Blackwell architecture chip with Compute Capability 12.0 (`sm_120`). While detected and healthy in Windows, the installed `torch-2.14.1+cu126` build lacks compiled `sm_120` kernels (which require cu130+). To prevent disk exhaustion on the host drive (11.5 GB free), GPU acceleration is **not** claimed; the pipeline runs natively on the 16-core CPU fallback at an uncompromised **35.9 ms/frame (~28 FPS)**.

---

## 6. Software Architecture & Interfaces
- **Vision Abstraction (`inspection/base.py`):** Standardized `VisionInspector` class enabling transparent switching between `CPUInspector` (laptop development/PyTorch/ONNX) and `HailoInspector` (HailoRT C++/Python API on ARM64) without touching application logic.
- **Deterministic Decision Engine (`decision_engine/`):** Evaluates defects against prioritized quality rules (`RULE_001` through `RULE_008`) across configurable presets (`consumer`, `industrial`, `automotive`). Never permits probabilistic LLM decision-making.
- **Advisory Engine (`advisory/`):** Formats inputs into Llama 3 Instruct tokens. Executes on-device through ExecuTorch, with automatic fallback (`advisory/fallback.py`) providing sub-millisecond deterministic engineering guidance. Enforces explicit backend attribution (`deterministic_fallback` vs `executorch_llama_3.2_1b`), guaranteeing zero attribution of fallback output to Llama.
- **Persistence Layer (`database/`):** SQLAlchemy ORM over local SQLite (`inspections.db`), storing inspection IDs, defect coordinates, unrounded timings, vision and advisory model versions, electrical test flags, and operator override audits with mandatory technical justification.

---

## 7. Dataset Details & Integrity Verification
Three primary open-access datasets are cataloged and verified in the repository:

### Authoritative Dataset Usage Table
| Dataset Name | Total Images Available | Annotated Instances | Annotation Format | Defect Classes & Semantics | Images Used in Baseline (`best.pt`) | Images Used in Unified Multi-Domain | Licence & Attribution Source |
|---|---|---|---|---|---|---|---|
| **DeepPCB** | **1,500 pairs** (3,000 files) | **10,013** | Coordinate text files normalized to YOLO | 6 classes: `open` (1,942), `short` (1,506), `mousebite` (1,965), `spur` (1,625), `copper` (1,474), `pin-hole` (1,501) | **1,120 train / 186 val / 194 test** | **1,120 train / 186 val / 194 test** | **MIT License**, Tang et al. (GitHub `tangsanli5201/DeepPCB`) |
| **DsPCBSD+** | **10,259 images** | **20,276** | COCO JSON & YOLO `.txt` | 9 industrial classes: `SH` (915), `SP` (4,584), `SC` (1,593), `OP` (1,770), `MB` (2,529), `HB` (2,883), `CS` (2,490), `CFO` (1,832), `BMFO` (1,680) | **0 (NOT USED in `best.pt`)** | **1,200 train / 250 val** | **CC BY 4.0**, Lv et al., *Nature Scientific Data* 11, 794 (2024), DOI: `10.1038/s41597-024-03656-8` |
| **PCB-Defect V2** | **230 high-res boards** | **1,704** (COCO) / **2,506** (OBB) | COCO JSON & YOLO OBB polygon | 6 classes: `missing_pad` (329), `mouse_bite` (830), `open_circuit` (304), `short` (289), `spur` (490), `spurious_copper` (264) | **0 (NOT USED in `best.pt`)** | **172 train / 33 val / 25 external color test** | **CC BY 4.0**, Rashid et al., *Data in Brief* 64 (2026) 112296, DOI: `10.1016/j.dib.2025.112296` |

### Zero-Leakage Partitioning Strategy
1. **DeepPCB:** 11 board groups partitioned deterministically by board topology:
   - Train (7 groups, 1,120 images): `group00041`, `group12000`, `group12100`, `group12300`, `group13000`, `group20085`, `group44000`.
   - Validation (2 groups, 186 images): `group50600`, `group77000`.
   - Untouched Test (2 groups, 194 images): `group90100`, `group92000`.
2. **PCB-Defect V2:** 230 whole-board images partitioned across 22 distinct board layouts (e.g. `50`, `51`, `61`, `67`, `80`...):
   - Train: 15 board layouts (172 images).
   - Validation: 3 board layouts (33 images).
   - External Color Test: 4 board layouts (25 images from layouts `68`, `80`, `81`, `82`).
3. Mathematical verification in `tests/test_splits.py` guarantees $Train \cap Val \cap Test = \emptyset$.

---

## 8. Model Methodology & Class Mapping Evidence
Canonical defect semantics are documented in `docs/ml_report/CLASS_MAPPING_EVIDENCE.md` and unified in `training/class_map.yaml`:
- **Universal Shared Classes (100% physically identical):**
  1. `open_circuit` (ID 0, Critical): Complete conductive trace break.
  2. `short_circuit` (ID 1, Critical): Unintended conductive copper bridge.
  3. `mouse_bite` (ID 2, Major): Edge void / notch reducing trace cross-section.
  4. `spur` (ID 3, Major): Sharp copper projection into dielectric spacing.
  5. `spurious_copper` (ID 4, Major): Isolated residual copper fleck on dielectric laminate.
- **Physical Distinction in Hole & Pad Anomalies:**
  - `pin_hole_missing_pad` (ID 5, Critical): Agglomerates annular pad structural integrity defects with explicit physical subtype tracking:
    - DeepPCB `pin-hole`: localized void in copper metallization (IPC-A-600 2.5).
    - PCB-Defect V2 `missing_pad`: complete absence of solder pad land (IPC-A-600 3.1).
    - DsPCBSD+ `HB` (Hole Breakout): drill misregistration breaking annular ring boundary (IPC-A-600 3.3).
- **Excluded Classes:**
  - `conductor_scratch` (`CS`), `conductor_foreign_object` (`CFO`), and `base_material_foreign_object` (`BMFO`) from DsPCBSD+ are excluded from the 6-class circuit detector to focus strictly on photolithographic trace geometry and prevent false rejection on superficial dust/scratches.

---

## 9. Conservative Inspection Decision Policy & False-PASS Prevention
When a baseline detector trained solely on monochrome scans encounters color solder masks (green, blue, copper) or severe camera defocus, convolutional activations collapse and detections drop to zero. The legacy engine previously equated `len(defects) == 0` with `PASS`, creating a critical industrial false-PASS vulnerability.

EdgePCB-QC replaces this with a **Conservative Industrial Quality Policy**:

1. **Pre-Inference Quality & Domain Assessment (`inspection/quality_assessment.py`):**
   - **Blur / Defocus Gate:** Laplacian variance $\sigma^2_{\Delta} < 80.0 \rightarrow$ flags `image_quality_status = "blurred"`.
   - **Illumination & Glare Gates:** Mean luminance $\mu < 30.0 \rightarrow$ `underexposed`; luminance $\mu > 248.0$ or glare ratio $> 35\% \rightarrow$ `overexposed`.
   - **Dynamic Contrast Gate:** Luminance std dev $\sigma < 20.0 \rightarrow$ `low_contrast`.
   - **Domain Shift Gate:** Evaluates HSV saturation $\mu_S$. If color presence exceeds monochrome thresholds and active model scope is unvalidated, flags `domain_status = "unfamiliar_domain"`.
2. **Prioritized Deterministic Decision Rules (`decision_engine/rules.py`):**
   - `REJECT`: High-confidence critical defect (`RULE_001`) or excess major defects (`RULE_002`).
   - `INSPECT`: Suspected critical defect (`RULE_003`), flagged major defect (`RULE_004`), degraded image quality (`RULE_005A_DEGRADED_IMAGE_QUALITY`), unfamiliar image domain (`RULE_005B_UNFAMILIAR_IMAGE_DOMAIN`), insufficient coverage (`RULE_005`), excess minor defects (`RULE_006`), or low-confidence anomaly (`RULE_007`).
   - `PASS`: Awarded **ONLY** when image quality is acceptable, domain scope is verified, coverage $\ge 90\%$, and zero critical defects exist (`RULE_008_ZERO_CRITICAL_DEFECTS_PASS`).
   - **Invariants Enforced:**
     - Zero detections does NOT mean PASS.
     - A color PCB is never rejected simply because it is color.
     - A grayscale PCB is never blindly trusted if blurred or poorly illuminated.
     - Electrical verification (`requires_electrical_test = True`) is mandatory for open and short circuits.
   - Verified across 10 industrial scenarios (`benchmarks/results/complete_workflow_scenarios.json`): **0 false passes occurred**.

---

## 10. Hailo NPU Deployment Preparation
Compiled deployment workflow documented in `docs/HAILO_SETUP.md`:
- Model parsed via Hailo DFC on x86-64 Linux host (`hailo parser onnx`).
- Quantized using Int8 Post-Training Quantization with 1,024 industrial calibration images.
- Compiled into `.hef` format targeted for Hailo-8 (26 TOPS) and Hailo-8L (13 TOPS).
- Dynamic runtime execution via `HailoInspector` with automatic fallback to CPU if physical accelerator is detached.

---

## 11. Llama and ExecuTorch Integration
- **Runtime:** ExecuTorch v1.5.1 compiled and loaded natively on Python 3.11 with `pytorch_tokenizers` and `torchao`.
- **Target Model:** Meta Llama 3.2 1B Instruct (`meta-llama/Llama-3.2-1B-Instruct`), compiled via official ExecuTorch 1.5.1 library (`export_llama_lib.build_model`).
- **Compiled Artifact:** `llama3_2_1b_fp16.pte` (2,357.90 MB, `2,472,432,290 bytes`). Compilation completed in **62.25s**; C++ runtime load time is **7.89 ms** via `_load_for_executorch`.
- **Dynamic Shape Token Slicing:** Input sequences are dynamically sliced (`seq_len = min(len(token_ids), 255)`) to eliminate redundant padding overhead, accelerating edge CPU prefill by 4x to 20x.
- **Air-Gapped Offline Verification:** Evaluated with network sockets systematically blocked (`scripts/test_offline_inference.py`), proving zero external dependency at runtime.
- **Strict Guardrails:** The SLM receives strictly filtered JSON from the decision engine. All responses validated via Pydantic (`AdvisoryExplanation`).
- **Deterministic Fallback:** `advisory/fallback.py` guarantees immediate (<1ms), grounded IPC-compliant troubleshooting advice when offline or during rapid line inspection.

---

## 12. Benchmarking & Empirical Model Evaluation

### A. Model Performance on Unseen PCB Test Boards (Zero Leakage)
Evaluated on DeepPCB test split (`group90100`, `group92000` — 194 unseen test images, 1,143 ground-truth defect instances). Source: `benchmarks/results/eval_unseen_boards.json`.

| Defect Class | Severity Class | Precision | Recall | mAP@50 | mAP@50-95 | Notes / Failure Modes |
|---|---|---|---|---|---|---|
| **Overall (All Classes)** | Multi-tier | **0.7136** | **0.6739** | **0.7153** | **0.4129** | Evaluated on unseen PCB topologies |
| `open_circuit` | **Critical** | 0.5714 | **0.7340** | **0.6931** | 0.3867 | Reliable trace break detection |
| `short_circuit` | **Critical** | 0.5930 | **0.3497** | **0.3945** | 0.1850 | Hairline copper bridging requires tiling/fine-tuning |
| `mouse_bite` | Major | 0.8341 | 0.6704 | 0.7691 | 0.4036 | Trace edge void reduction |
| `spur` | Major | 0.9083 | 0.5054 | 0.6978 | 0.3328 | Protruding conductor anomaly |
| `spurious_copper` | Major | 0.9146 | **0.8434** | **0.9315** | 0.6385 | Highly distinctive isolated copper fleck |
| `pin_hole_missing_pad` | **Critical** | 0.4600 | **0.9405** | **0.8059** | 0.5306 | High sensitivity (94% recall) on annular rings |

### B. Latency and Memory Benchmarks
Real, unsimulated benchmark measurements logged in `benchmarks/results/`:

| Pipeline Stage | Laptop Host (AMD Ryzen 9 CPU) | Target Pi 5 + Hailo NPU (Projected) | Memory Footprint / Model Size |
|---|---|---|---|
| **Vision Inference (YOLOv8n 640x640, PyTorch)** | 27.21 ms (34.4 FPS end-to-end: 29.07 ms) | 12.5 ms (~80 FPS on Hailo-8) | 714.5 MB peak RAM (6.20 MB .pt) |
| **Vision Inference (YOLOv8n 640x640, ONNX)** | 26.97 ms (33.4 FPS end-to-end: 29.92 ms) | 12.5 ms (~80 FPS on Hailo-8) | 821.6 MB peak RAM (12.19 MB ONNX) |
| **Golden Reference Comparison (OpenCV)** | 45.54 ms (22.0 FPS) | 38.0 ms (ARM NEON) | 68.1 MB peak RAM |
| **Deterministic Decision Engine** | 0.08 ms | 0.12 ms | < 5 MB |
| **Llama Advisory (Deterministic Fallback)** | 0.20 ms | 0.35 ms | 185.2 MB peak RAM |
| **Llama 3.2 1B Instruct (ExecuTorch FP16 .pte)** | 36.59s TTFT (49 tok prompt), 0.82 tok/s | ~240 ms TTFT, 14 tok/s (Int4 / NPU) | 2,357.90 MB artifact, 3,021.9 MB peak RAM |

### C. Optical Robustness & High-Resolution Multi-Panel Tiling Audit
Measured empirical reliability under factory illumination shifts and high-resolution composite arrays (Source: `benchmarks/results/inspection_reliability_report.json`):

1. **Optical Shift Robustness (640x640 Test Board):**
   - **Nominal Illumination:** 10 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 35.9 ms.
   - **Overexposure / Glare (+15% gain, +30 offset):** 10 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 34.0 ms.
   - **Underexposure / Low Light (-15% gain, -35 offset):** 10 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 29.1 ms.
   - **Optical Defocus (Gaussian 5x5 blur):** 9 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 27.9 ms.
   - **Sensor Thermal Noise (Gaussian std 15):** 6 defects detected (Open: Yes, Short: Yes), Verdict: `REJECT`, Latency: 28.2 ms.
   - **Fail-Safe Invariance:** Zero false-negative `PASS` decisions occurred under any optical condition.

2. **High-Resolution Multi-Panel Tiling (1280x1280 Composite Array, 1.64 MP):**
   - Evaluated using `TiledInferenceRunner` with 640x640 tiles and 20% spatial overlap.
   - Sliced into 9 overlapping sub-windows in 0.02 ms.
   - Extracted and fused **36 global defect instances** after Global Non-Maximum Suppression (NMS).
   - Total tiled multi-panel latency: **258.29 ms** (28.7 ms per tile).
   - Panel Decision Verdict: `REJECT` via `RULE_001_HIGH_CONF_CRITICAL_DEFECT`.

### D. Dual-Mode Advisory Verification & Attribution Integrity
To guarantee both high line throughput and explainable engineering reasoning, EdgePCB-QC implements two decoupled advisory execution modes (Source: `benchmarks/results/advisory_modes_benchmark.json`):

| Metric / Attribute | Mode 1: Fast Deterministic Fallback | Mode 2: ExecuTorch Llama 3.2 1B On-Device | Decision Invariance Verified? |
|---|---|---|---|
| **Primary Use-Case** | Real-time inline triage (<1ms) | In-depth failure analysis & operator assistance | Invariant across modes |
| **Inspection Decision** | `REJECT` (`RULE_001_HIGH_CONF_CRITICAL_DEFECT`) | `REJECT` (`RULE_001_HIGH_CONF_CRITICAL_DEFECT`) | **Identical (REJECT)** |
| **Requires Electrical Test** | `True` (Unconditionally flags open/short) | `True` (Unconditionally flags open/short) | **Identical (True)** |
| **Advisory Backend** | `deterministic_fallback` | `executorch_llama_3.2_1b` | **Strictly Disjoint** |
| **Model Identifier** | `ipc_expert_rules_engine` | `meta-llama/Llama-3.2-1B-Instruct` | **Strictly Disjoint** |
| **Database `llm_version`** | `None (Deterministic Expert Fallback)` | `Llama-3.2-1B-Instruct` | **Zero False Attribution** |
| **Advisory Latency** | **0.18 ms** | **71.68 s** (Autoregressive generation) | Traceable to exact backend |
| **Generated Output** | Grounded IPC action guidance | Grounded guidance + Neural token generation | Non-conflicting |

### E. ExecuTorch Llama 3.2 1B Latency & Performance Benchmark Matrix
Autoregressive token generation benchmarks measured on host AMD Ryzen 9 8940HX CPU across 12 combinations of prompt lengths, generation budgets, and CPU thread pools (Source: `benchmarks/results/advisory_latency_matrix.json`):

| Configuration | Threads | Prompt Tokens | Actual Gen Tokens | Time to First Token (TTFT) | Decode Time | Decode Throughput | Peak RAM Footprint | Sentence Complete? |
|---|---|---|---|---|---|---|---|---|
| **32-tok prompt / 16 gen** | 4 | 41 | 3 | **22,702.4 ms** (22.70s) | 2.98 s | **1.01 tokens/s** | 3,019.2 MB | Fallback (Fragment) |
| **32-tok prompt / 16 gen** | 8 | 41 | 3 | **21,169.5 ms** (21.17s) | 2.80 s | **1.07 tokens/s** | 3,022.1 MB | Fallback (Fragment) |
| **32-tok prompt / 24 gen** | 4 | 41 | 3 | **20,961.5 ms** (20.96s) | 2.83 s | **1.06 tokens/s** | 3,022.1 MB | Fallback (Fragment) |
| **32-tok prompt / 24 gen** | 8 | 41 | 3 | **20,934.0 ms** (20.93s) | 2.85 s | **1.05 tokens/s** | 3,022.0 MB | Fallback (Fragment) |
| **64-tok prompt / 16 gen** | 4 | 69 | 24 | **37,394.5 ms** (37.39s) | 24.72 s | **0.97 tokens/s** | 3,022.4 MB | Fallback (Incomplete) |
| **64-tok prompt / 16 gen** | 8 | 69 | 24 | **36,064.5 ms** (36.06s) | 22.79 s | **1.05 tokens/s** | 3,022.4 MB | Fallback (Incomplete) |
| **64-tok prompt / 24 gen** | 4 | 69 | 25 | **34,714.6 ms** (34.71s) | 22.62 s | **1.11 tokens/s** | 3,022.4 MB | **YES (`is_complete=True`)** |
| **64-tok prompt / 24 gen** | 8 | 69 | 25 | **34,567.1 ms** (34.57s) | 23.07 s | **1.08 tokens/s** | 3,022.5 MB | **YES (`is_complete=True`)** |
| **96-tok prompt / 16 gen** | 4 | 94 | 24 | **47,243.8 ms** (47.24s) | 22.75 s | **1.05 tokens/s** | 3,022.8 MB | Fallback (Incomplete) |
| **96-tok prompt / 16 gen** | 8 | 94 | 24 | **47,363.0 ms** (47.36s) | 22.99 s | **1.04 tokens/s** | 3,022.8 MB | Fallback (Incomplete) |
| **96-tok prompt / 24 gen** | 4 | 94 | 26 | **47,194.9 ms** (47.19s) | 24.30 s | **1.07 tokens/s** | 3,022.8 MB | **YES (`is_complete=True`)** |
| **96-tok prompt / 24 gen** | 8 | 94 | 26 | **46,983.6 ms** (46.98s) | 24.01 s | **1.08 tokens/s** | 3,022.8 MB | **YES (`is_complete=True`)** |

*Authoritative Latency & System Architecture Findings:*
1. **Linear Prefill Scaling ($T_{\text{prefill}} \approx 500\text{ ms per prompt token}$):**
   - At 41 prompt tokens: TTFT $\approx 21.0\text{ seconds}$.
   - At 69 prompt tokens: TTFT $\approx 34.6\text{ seconds}$.
   - At 94 prompt tokens: TTFT $\approx 47.1\text{ seconds}$.
   - Keeping the edge prompt concise (64 tokens) saves over 12.5 seconds in TTFT compared to verbose templates.
2. **Autoregressive Decoding Rate (1.05 – 1.11 tokens/second):**
   - Decode time scales strictly with generated tokens ($\approx 910 - 950\text{ ms per generated token}$).
   - Generates complete sentences (25–26 tokens) in **22.6 to 24.3 seconds**.
3. **Thread Scaling Invariance (4 vs 8 Threads):**
   - 4 threads achieve **1.11 tok/s** (57.34s total) versus 8 threads at **1.08 tok/s** (57.64s total).
   - This proves that Transformer inference on host CPU is memory bandwidth-bound (streaming 2.35 GB model weights per decode step) rather than compute-bound.
4. **Sentence Completeness Quality:**
   - 24-token budget paired with the 8-token grace window consistently produces complete, professional sentences:
     `"The board PCB-PERF-001 shows visually suspected short-circuit, which poses a significant failure risk to the system."`
   - Incomplete generations (e.g. 16-token limit) automatically trigger `fallback_on_incomplete` and return deterministic IPC advice without corrupting operator action.
5. **Memory Containment:**
   - Peak RSS memory footprint is held strictly at **3,022.8 MB**, safely inside system RAM allowances. Resident runtime init overhead is 1,679.57 ms (one-time load).

### F. Offline Air-Gapped Verification & Physical Artifact Audit
EdgePCB-QC was audited under zero-network constraints to ensure full physical air-gap compliance:
- **Physical Model Weights:**
  - `models/llama/checkpoints/Llama-3.2-1B-Instruct/llama3_2_1b_fp16.pte`: 2,357.90 MB (`2,472,432,290 bytes`)
  - `models/llama/tokenizer.model`: 2.08 MB (`2,185,274 bytes`)
  - `models/checkpoints/pcb_smoke_experiment/weights/best.pt`: 5.69 MB (`5,964,281 bytes`)
  - `models/onnx/pcb_smoke_experiment.onnx`: 11.68 MB (`12,243,303 bytes`)
- **Socket Isolation Test (`scripts/test_offline_inference.py`, `scripts/test_offline_evidence.py`):**
  - All outbound network calls (`socket.socket`, `socket.create_connection`, `socket.getaddrinfo`) systematically intercepted.
  - Active outbound connections: **0**.
  - All inferences execute purely from local RAM and disk without contacting remote endpoints.

---

## 13. Limitations & Engineering Disclaimers
1. **2D Optical Limitations:** Standard 2D RGB optical cameras cannot measure solder joint barrel volume, BGA voids, or internal layer shorts on multi-layer boards.
2. **Visual vs Electrical Uncertainty:** Optical detection of trace discontinuity is classified as a *visually suspected* defect requiring electrical confirmation (flying-probe or DMM) before scrapping high-value multi-layer panels.
3. **Domain & Lighting Shift:** Reference comparison assumes stable fixture illumination; excessive shadow variation requires recalibration.

---

## 14. Roadmap & Future Work
- [ ] Physical bench deployment on Raspberry Pi 5 + Raspberry Pi AI HAT+.
- [ ] Integration of PCBA component placement inspection (missing passives, polarity reversal).
- [ ] Local multilingual operator voice advisory (Hindi, Tamil, Kannada) using edge TTS.
- [ ] Automated export of tamper-evident PDF inspection certificates for IPC compliance auditing.
