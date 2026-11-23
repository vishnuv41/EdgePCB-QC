# EdgePCB-QC — Authoritative Research Evidence Audit
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**  
**Project:** EdgePCB-QC (Offline Edge-AI Industrial PCB Defect Detection & Explainable Quality Control)  
**Document Classification:** Scientific Evidence Audit & Provenance Verification  
**Date:** November 2026  

---

## 1. Audit Methodology & Verification Standards

To guarantee strict scientific integrity and defensibility, every technical claim, numerical metric, architectural assertion, and experimental finding presented in the IEEE research paper is cross-referenced against physical artifacts, codebases, and test executions.

### Classification Taxonomy
- **`[VERIFIED - MEASURED]`**: Empirically measured on hardware, backed by raw benchmark JSONs, automated pytest test suites, and direct runtime profiling.
- **`[VERIFIED - DISK ARTIFACT]`**: Verified by direct disk inspection, byte size measurement, and SHA-256 cryptographic hashing.
- **`[VERIFIED - CODEBASE INVARIANT]`**: Verified by inspectable source code implementation and deterministic logic.
- **`[PROJECTED - TARGET HARDWARE]`**: Derived from hardware vendor specifications, compiler profilers (Hailo DFC), or analytical projection for target edge hardware pending physical deployment.
- **`[TRANSPARENT NEGATIVE FINDING]`**: Rigorously documented negative results, failed experimental hypotheses (e.g., Experiment H1 Mosaic trace-severing), or resolved bugs.

---

## 2. Model Artifacts & Cryptographic Provenance

| Component / Artifact | File Path | File Size | SHA-256 Cryptographic Hash | Verification Evidence & Method | Status |
|:---|:---|:---:|:---|:---|:---:|
| **YOLOv8n PyTorch Weights** | `models/checkpoints/pcb_multidomain_experiment/weights/best.pt` | 6,505,130 bytes (6.20 MB) | `66d8d225fe2137c160b8038d0e68200a54d1544152b0941050f749ee0741be04` | Direct disk read & bitwise hash via Python `hashlib`. Frozen Phase 1 production model. | `[VERIFIED - DISK ARTIFACT]` |
| **YOLOv8n ONNX Export** | `models/onnx/pcb_multidomain_yolov8n_640.onnx` | 12,778,213 bytes (12.19 MB) | `0f219cae19807720f6528d635fc7168e1c067a91370e59d165a79c7536089662` | Opset 12 static export [1, 3, 640, 640] $\to$ [1, 10, 8400]. | `[VERIFIED - DISK ARTIFACT]` |
| **Hailo-8 Compiled HEF** | `models/hailo/pcb_yolov8n_raw_heads.hef` | 4,557,262 bytes (4.35 MB) | `053a1967ad9489847ee80e058a4608518713706baf2fa79915159d952df3dada` | Compiled via Hailo DFC 3.34.0 targeting Hailo-8 with 256 calibration images. | `[VERIFIED - DISK ARTIFACT]` |
| **Llama 3.2 1B 8da4w PTE** | `models/llama/llama3_2_1b_xnnpack_8da4w.pte` | 1,689,109,888 bytes (1.57 GiB) | `2de22a466571e7e0cdd0adf01d72bed5f89d53d14784f759c06fd597616a9791` | Quantized ExecuTorch XNNPACK binary (8-bit dynamic act, 4-bit weight). | `[VERIFIED - DISK ARTIFACT]` |
| **Llama Tokenizer** | `models/llama/tokenizer.model` | 2,183,982 bytes (2.08 MB) | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` | Official Meta Llama 3.2 BPE tokenizer model file. | `[VERIFIED - DISK ARTIFACT]` |
| **Exp C2 Clean Weights** | `models/checkpoints/exp_c2_color_hard_negatives_clean/weights/best.pt` | 6,511,338 bytes (6.21 MB) | `281292144e1d52ef72a9117621aa9e73fbf50bc594a123611e9f1a21f7a63d91` | Curated color + monochrome negative mining checkpoint. | `[VERIFIED - DISK ARTIFACT]` |
| **Exp G 11-Class Weights** | `models/checkpoints/exp_g_11class/weights/best.pt` | 6,513,994 bytes (6.21 MB) | `8194e1918d3600f6c243a42cae2d9b2e7ee42e5bf7e0ef733c7f998492061214` | Extended 11-class surface anomaly model. | `[VERIFIED - DISK ARTIFACT]` |

---

## 3. Dataset Composition & Zero-Leakage Splitting

| Dataset Benchmark | Raw Data Volume | Train Images (Boxes) | Validation Images (Boxes) | Held-Out Test Images (Boxes) | Partitioning Protocol & Evidence Source |
|:---|:---|:---:|:---:|:---:|:---|
| **DeepPCB** | 1,500 test images, 1,501 template images (3,001 total) | 1,120 (16,099 bboxes) | 186 (2,735 bboxes) | 194 (1,180 bboxes) | Group-isolated: 7 groups train (`group00041`..), 2 val (`group20010`..), 2 test (`group90100`, `group92000`). Zero board overlap. Source: `exact_inventory_counts.json`. |
| **DsPCBSD+** | 10,259 industrial color scanning images | 5,811 (20,283 bboxes) | 1,444 (3,990 bboxes) | 0 (All held-out evaluation on V2 & DeepPCB) | Bitwise hash & filename sequence-disjoint partition. Source: `dspcbsd_leakage_audit.json`. |
| **PCB-Defect V2** | 230 whole-board multi-megapixel images (22 physical boards) | 2,376 crops (3,750 bboxes) | 495 crops (792 bboxes) | 390 crops (497 bboxes) | Board serial number isolated: Boards 50–64 (train), 65–67 (val), 68, 80–82 (test). Source: `dataset_authoritative_audit.json`. |
| **Unified Total** | **12,016 images** | **9,307 images** (22,066 bboxes) | **2,125 images** (4,827 bboxes) | **584 images** (1,677 bboxes) | Automated verification in `tests/test_splits.py` (0 duplicate hashes, 0 board-level leakage). |

---

## 4. Vision Model Detection Benchmarks

| Metric / Experiment | Evaluated Checkpoint | Value | Protocol & Context | Source File & Audit Evidence | Classification |
|:---|:---|:---:|:---|:---|:---:|
| **Precision (at best epoch)** | `pcb_multidomain_experiment` | **76.00%** | Ultralytics validation protocol, conf $\ge 0.25$, IoU = 0.50 | `benchmarks/results/fair_training_comparison.json` | `[VERIFIED - MEASURED]` |
| **Recall (at best epoch)** | `pcb_multidomain_experiment` | **75.25%** | Ultralytics validation protocol, conf $\ge 0.25$, IoU = 0.50 | `benchmarks/results/fair_training_comparison.json` | `[VERIFIED - MEASURED]` |
| **mAP@0.50 (at best epoch)** | `pcb_multidomain_experiment` | **80.02%** | Standard COCO mAP@0.50 across 6 canonical defect classes | `benchmarks/results/fair_training_comparison.json` | `[VERIFIED - MEASURED]` |
| **mAP@0.50:0.95** | `pcb_multidomain_experiment` | **48.76%** | Standard COCO mAP across IoU thresholds 0.50 to 0.95 | `benchmarks/results/fair_training_comparison.json` | `[VERIFIED - MEASURED]` |
| **Full PR Integration mAP50** | `pcb_multidomain_experiment` | **81.25%** | Standard numerical integration across full confidence range (0.001..1.0) | `benchmarks/results/methodology_reconciliation_map50.json` | `[VERIFIED - MEASURED]` |
| **Truncated Operational mAP50** | `pcb_multidomain_experiment` | **73.12%** | PR integration truncated at industrial threshold (conf $\ge 0.25$) | `benchmarks/results/methodology_reconciliation_map50.json` | `[VERIFIED - MEASURED]` |
| **Custom Evaluator Precision** | `pcb_multidomain_experiment` | **66.97%** | Strict bipartite IoU matching with native sensor dimensions | `benchmarks/results/baseline_multidomain_reference_evaluation.json` | `[VERIFIED - MEASURED]` |
| **Custom Evaluator Recall** | `pcb_multidomain_experiment` | **77.50%** | Strict bipartite IoU matching with native sensor dimensions | `benchmarks/results/baseline_multidomain_reference_evaluation.json` | `[VERIFIED - MEASURED]` |
| **Exp C mAP@0.50** | `exp_c_hard_negatives` | **87.57%** | Monochrome negative mining, conf $\ge 0.001$ sweep | `benchmarks/results/methodology_reconciliation_map50.json` | `[VERIFIED - MEASURED]` |
| **Clean Board PASS (Baseline)** | `pcb_multidomain_experiment` | **49.23%** (96/195) | 195 clean DeepPCB reference boards (`group90100`, `group92000`) | `benchmarks/results/baseline_vs_c2_clean_full_audit.json` | `[VERIFIED - MEASURED]` |
| **Clean Board PASS (Exp C2)** | `exp_c2_color_hard_negatives_clean` | **93.33%** (182/195) | 195 clean DeepPCB reference boards after color negative mining | `benchmarks/results/baseline_vs_c2_clean_full_audit.json` | `[VERIFIED - MEASURED]` |
| **Clean False Scrap (Baseline $\to$ C2)** | Baseline vs C2 | **9.74% $\to$ 0.51%** | 19.0× reduction in false scrap (19 scrap boards $\to$ 1 scrap board) | `benchmarks/results/baseline_vs_c2_clean_full_audit.json` | `[VERIFIED - MEASURED]` |
| **FP Box Overkill (Baseline $\to$ C2)** | Baseline vs C2 | **357 $\to$ 15 boxes** | 95.8% reduction in spurious bounding box alarms | `benchmarks/results/baseline_vs_c2_clean_full_audit.json` | `[VERIFIED - MEASURED]` |

---

## 5. Vision Computational Performance & Latency Budget

| Pipeline Stage / Metric | Host Machine (AMD Ryzen 9 8940HX) | Raspberry Pi 5 ARM64 CPU | Hailo-8 NPU (26 TOPS) | Evidence Source & Measurement Protocol |
|:---|:---:|:---:|:---:|:---|
| **YOLOv8n PyTorch CPU Inference** | **27.21 ms** (34.4 FPS) | $42.3\text{ ms}$ (projected) | N/A | 100-image batch benchmark in `vision_computational_cost.json`. |
| **YOLOv8n ONNX Runtime Inference** | **26.97 ms** (33.4 FPS) | $39.5\text{ ms}$ (projected) | N/A | 100-image batch benchmark in `vision_computational_cost.json`. |
| **Hailo-8 NPU Inference** | N/A (Windows development host) | N/A | **$5.8\text{ ms}$** (projected) | Hailo Dataflow Compiler (DFC 3.34.0) profiler output. |
| **OpenCV Preprocessing (Letterbox)** | **0.99 ms** | $2.1\text{ ms}$ (projected) | $2.1\text{ ms}$ (projected) | Preprocessing benchmark in `vision_computational_cost.json`. |
| **Vectorized NMS Post-Processing** | **0.48 ms** | $1.4\text{ ms}$ (projected) | $1.4\text{ ms}$ (projected) | NumPy vectorized NMS at IoU = 0.45, conf = 0.25. |
| **End-to-End Vision Pipeline** | **29.07 ms** | $45.8\text{ ms}$ (projected) | **$9.3\text{ ms}$** (projected) | Capture + Preprocess + Inference + NMS. |
| **Peak Resident RAM (Vision)** | **714.5 MB** | $< 850\text{ MB}$ (projected) | $< 650\text{ MB}$ (projected) | Process memory profiling via `psutil`. |

---

## 6. Small Language Model (SLM) Advisory Audit (ExecuTorch Llama 3.2 1B)

| Parameter / Metric | Measured Value (Host CPU) | Test Condition & Protocol | Source File & Test Verification | Classification |
|:---|:---:|:---|:---|:---:|
| **Model Architecture** | Meta Llama 3.2 1B Instruct | 1.23B parameters, 16 layers, GQA | `models/llama/llama3_2_1b_xnnpack_8da4w.pte` | `[VERIFIED - DISK ARTIFACT]` |
| **Quantization Scheme** | 8da4w INT4 | 8-bit dynamic activation, 4-bit per-channel weight | `advisory/verify_quantization.py` | `[VERIFIED - DISK ARTIFACT]` |
| **Model Load / mmap Latency** | **1.54 s** | Cold file-system load to memory mapping | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - MEASURED]` |
| **Cold Time-to-First-Token (TTFT)** | **2,427.98 ms** | First un-cached prefill pass (58 tokens) | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - MEASURED]` |
| **Warmed-Up TTFT** | **853.21 ms** | Sub-second cached prefill pass | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - MEASURED]` |
| **Decode Throughput** | **10.73 tokens/s** | Greedy autoregressive generation (56 tokens) | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - MEASURED]` |
| **End-to-End Generation Latency** | **9,225.5 ms** (~9.2s) | Complete 56-token explanation generation | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - MEASURED]` |
| **Peak Working RAM (SLM)** | **1,322.6 MB** (~1.29 GB) | Peak memory during token generation | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - MEASURED]` |
| **Deterministic Fallback Latency** | **0.0996 ms** (~0.10 ms) | IPC-A-610 rule-based instant guidance | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - MEASURED]` |
| **Verdict Invariance Guarantee** | **100% Invariant** | SLM cannot modify PASS/INSPECT/REJECT | `tests/test_llama_engine_empirical_audit.py` | `[VERIFIED - CODEBASE INVARIANT]` |

---

## 7. Resolved Engineering Bugs & Negative Experimental Hypotheses

| Incident / Hypothesis | Nature of Problem | Root Cause Identified | Remediation Applied | Empirical Verification |
|:---|:---|:---|:---|:---|
| **Coordinate Scaling Bug** | Custom evaluator reported 28.2% Prec / 31.5% Rec, while Ultralytics reported 80.6% mAP50. | Evaluator assumed hardcoded 640×640 dimensions, while DsPCBSD+ native crops were ~226×226 px, causing 0.000 IoU on 2,875 ground-truth boxes. | Patched `evaluate_candidate_model.py` to dynamically query PIL image dimensions. | DsPCBSD+ Precision rose from 0.0% to 70.54%, reconciling custom evaluator with Ultralytics at 80.58% mAP50. |
| **Experiment H1 Regression** | Training with hard negatives to suppress annular ring false calls degraded clean PASS rate and surged false open circuits (+56). | Ultralytics `mosaic=1.0` 4-image stitching sliced continuous copper traces at quadrant boundaries, training the model that normal trace endings were open circuits. | Officially rejected Exp H1; frozen Baseline Multi-Domain model retained; disabled mosaic augmentation in fine-tuning. | Documented in `benchmarks/results/exp_h1_forensic_investigation_results.json`. |
| **Llama Preamble Cut-off Bug** | Llama generated only `"Based on the inspection results, the following conclusions can be made: 1."`. | Default `max_new_tokens` was 16, running out of tokens during preamble; period in `"1."` tripped sentence boundary. | Increased default `max_new_tokens` to 56; prompted strictly against numbered lists; added regex list-marker stripper. | Live API test verified 67 tokens generated as 3 complete sentences. |
| **Literal `"undefined"` DOM Bug** | `#workspace` rendered the literal word `"undefined"` below the neural advisory box. | `confidence_caveat` was omitted in backend dictionary return, causing template literal `${r.advisory.confidence_caveat}` to evaluate to `undefined`. | Added field to backend payload and wrapped in defensive null-check `${r.advisory?.confidence_caveat ? ... : ''}`. | Zero `undefined` text verified in live frontend testing. |
