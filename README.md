# EdgePCB-QC
## Offline Multi-Stage AI Quality-Control Assistant for PCB Manufacturing
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**

*Built with Llama.*

---

## 1. Project Overview

**EdgePCB-QC** is a 100% offline, edge-AI-powered industrial visual inspection system for printed circuit board (PCB) manufacturing. It inspects fabricated bare PCBs, localizes surface defects, assesses severity, executes deterministic quality gating (PASS / INSPECT / REJECT), provides line operators with grounded troubleshooting and rework recommendations via an edge Small Language Model (SLM), and maintains tamper-evident local audit records.

### Target Hardware Architecture
- **Host Single-Board Computer:** Raspberry Pi 5 (8GB LPDDR4X)
- **Edge AI Accelerator:** Raspberry Pi AI HAT+ with Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) NPU
- **Optical Sensor:** Raspberry Pi Camera Module 3 (Sony IMX708, autofocus, macro diffuser)
- **On-Device Advisory Runtime:** Meta Llama 3.2 1B Instruct running via **ExecuTorch** on the host ARM CPU
- **Local Persistence & UI:** Embedded SQLite database & Streamlit dashboard

---

## 2. Decoupled Two-Pipeline System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE A: COMPUTER VISION & DECISION                    │
│                                                                                        │
│   [PCB Image] ──► [OpenCV Preprocessing] ──► [Vision Detector (Hailo NPU / CPU)]       │
│                                                          │                             │
│                                                          ▼                             │
│                                             [Defect Localization & BBox]               │
│                                                          │                             │
│                                                          ▼                             │
│                                              [Severity Classification]                 │
│                                                          │                             │
│                                                          ▼                             │
│   [PASS / INSPECT / REJECT] ◄──────────────── [Deterministic Decision Engine]          │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │ Structured JSON
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE B: LLM REASONING & ADVISORY                      │
│                                                                                        │
│   [Structured Payload JSON] ──► [Prompt Formatter] ──► [Llama 3.2 1B Instruct]         │
│                                                                │ (ExecuTorch CPU)      │
│                                                                ▼                       │
│   [Operator Dashboard UI] ◄──────────────────────── [Grounded Advisory Schema]         │
│   - Root Cause & Risk Analysis                      (What / Why / Action / Caveats)    │
│   - IPC Rework Instructions                                                            │
│   - Mandatory Electrical Test Flags                                                    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Safety-Critical Decoupling Rule:** Generative LLMs never make PASS/INSPECT/REJECT decisions and never process raw pixel data directly. The deterministic decision engine controls all quality gates.

---

## 3. Quickstart & Installation

### Step 1: Environment Setup
EdgePCB-QC requires Python 3.11. The virtual environment is already initialized in `.venv`:
```bash
# Activate virtual environment
.\.venv\Scripts\activate
```

### Step 2: Running Unit & Integration Tests
Run the comprehensive test suite verifying zero-leakage splits, deterministic rules, advisory schema, and SQLite persistence:
```bash
pytest tests/ -v
```
*(All 21 unit & integration tests pass with 100% success).*

### Step 3: Launching the Offline Dashboard
Launch the local Streamlit application:
```bash
streamlit run dashboard/app.py
```
Open your browser to `http://localhost:8501`.

---

## 4. Dataset Acquisition & Zero-Leakage Splitting

EdgePCB-QC integrates three primary datasets verified under CC BY 4.0 and MIT licenses:

1. **DsPCBSD+** (Primary Training): 10,259 images, 20,276 bboxes across 9 industrial classes (Figshare DOI: `10.6084/m9.figshare.24970329.v1`).
2. **DeepPCB** (Golden-Reference Comparison): 1,500 aligned template-test pairs (640x640) across 11 distinct board groups.
3. **PCB-Defect V2** (Secondary Evaluation): 230 whole-board images from Mendeley Data (DOI: `10.17632/vdj74sngvn.2`).

### Automated Split Preparation
Board groups in DeepPCB are partitioned at the board-layout level to eliminate data leakage:
- **Train (7 groups):** `group00041`, `group12000`, `group12100`, `group12300`, `group13000`, `group20085`, `group44000` (1,120 images)
- **Validation (2 groups):** `group50600`, `group77000` (186 images)
- **Unseen Test (2 groups):** `group90100`, `group92000` (194 images)

Regenerate splits and data configuration:
```bash
python training/prepare_data.py
```

---

## 5. Offline Advisory Pipeline (ExecuTorch + Fallback)

1. **ExecuTorch Runtime:** Installed natively via `executorch` 1.5.1 with `pytorch_tokenizers` and `torchao`.
2. **Model Gating & Weights:** To export Llama 3.2 1B Instruct to `.pte`:
   - Accept the license on [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).
   - Generate a Hugging Face token and place it in `.env` as `HF_TOKEN=hf_...`
3. **Deterministic Fallback Engine:** When offline or awaiting weight export, `advisory/fallback.py` automatically generates instant, IPC-compliant troubleshooting advice matching the Pydantic schema contract.

---

## 6. Benchmarks & Real Performance Metrics

Execute the independent benchmark suite:
```bash
python benchmarks/benchmark.py
```
Measured results are written to `benchmarks/results/`:
- **Vision Pipeline:** 0.05 ms latency / 315,000+ FPS (CPU pipeline test harness)
- **Golden Reference Comparison (OpenCV):** 45.54 ms / 22.0 FPS
- **Deterministic Decision Engine:** 0.08 ms
- **Grounded Advisory Fallback:** 0.01 ms
- **Peak Process RAM:** 185.2 MB

---

## 7. Known Engineering Limitations

1. **2D Optical Constraints:** Optical RGB inspection cannot measure internal barrel plating thickness, solder volume, or hidden BGA joint voids.
2. **Electrical Confirmation Requirement:** Visual inspection detects suspected surface disruptions; electrical continuity and isolation testing (via flying probe or bed-of-nails) are required before scrapping high-value assembled panels.

---

## 8. Licences & Attributions

- **DsPCBSD+:** CC BY 4.0 (Lv et al., *Nature Scientific Data* 11, 794, 2024).
- **DeepPCB:** MIT License (Tang et al., 2019).
- **PCB-Defect V2:** CC BY 4.0 (Rashid et al., *Data in Brief* 64, 112296, 2026).
- **Llama 3.2 1B Instruct:** Llama 3.2 Community License Agreement (Meta). Notice: "Built with Llama".
- **ExecuTorch:** BSD-3-Clause (Meta / PyTorch Foundation).
