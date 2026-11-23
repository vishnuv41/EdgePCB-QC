# EdgePCB-QC — Consolidated Benchmark & Experimental Evaluation Summary
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**  
**Document Type:** Empirical Benchmark Synthesis & Comparative Analysis  
**Date:** November 2026  

---

## 1. Vision Model Benchmark Matrix across Iterations

All candidate models share the lightweight YOLOv8 nano backbone (3.14M parameters, 8.8 GFLOPs at $640 \times 640$).

| Experiment Code | Checkpoint Path | Epochs | Clean Training Protocol | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | Clean Board PASS Yield | Clean Board False Scrap | FP Boxes / Board |
|:---|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **pcb_smoke_experiment** | `checkpoints/pcb_smoke_experiment` | 10 | DeepPCB-only sanity baseline | 93.77% | 67.31% | 94.45% | 86.21% | N/A (single-domain) | N/A | N/A |
| **pcb_multidomain_baseline** *(Frozen Production)* | `checkpoints/pcb_multidomain_experiment` | 10 | Unified 6-Class (DeepPCB + DsPCBSD+ + V2) | **80.02%** | **48.76%** | **76.00%** | **75.25%** | **49.23%** (96/195) | **9.74%** (19/195) | 1.83 (357 total) |
| **Exp A** | `checkpoints/exp_a_baseline_finetune` | 40 | Fine-tuned 40 epochs on unified data | 82.14% | 50.12% | 77.20% | 76.10% | 51.28% (100/195) | 8.71% (17/195) | 1.68 (328 total) |
| **Exp B** | `checkpoints/exp_b_taxonomy_split` | 20 | 8-Class split separating pinhole & spur | 79.40% | 47.90% | 74.10% | 73.80% | 54.35% (106/195) | 7.69% (15/195) | 1.51 (295 total) |
| **Exp C** | `checkpoints/exp_c_hard_negatives` | 20 | Monochrome negative template injection | 87.57% | 54.23% | 83.33% | 82.81% | 95.90% (187/195) | 1.54% (3/195) | 0.08 (15 total) |
| **Exp C2 Clean** *(Research Peak)* | `checkpoints/exp_c2_color_hard_negatives_clean` | 20 | Curated color + monochrome negative mining | **83.25%** | **52.10%** | **78.90%** | **81.12%** | **93.33%** (182/195) | **0.51%** (1/195) | **0.08** (15 total) |
| **Exp D** | `checkpoints/exp_d_domain_balanced` | 20 | Domain-balanced 1:1:1 resampled batches | 78.90% | 47.30% | 73.40% | 74.50% | 61.54% (120/195) | 5.64% (11/195) | 1.18 (231 total) |
| **Exp E** | `checkpoints/exp_e_small_defect_aug` | 20 | Small-defect scaling augmentation | 80.12% | 48.90% | 74.80% | 75.60% | 58.97% (115/195) | 6.15% (12/195) | 1.29 (252 total) |
| **Exp G** | `checkpoints/exp_g_11class` | 20 | Extended 11-class surface anomaly model | 76.45% | 46.10% | 71.30% | 72.80% | 93.33% (182/195) | 2.56% (5/195) | 0.11 (21 total) |
| **Exp H1** *(Rejected Regression)* | `checkpoints/exp_h1_hard_negatives` | 20 | Negative mining with `mosaic=1.0` | 77.80% | 45.80% | 69.40% | 71.20% | 90.77% (177/195) | 4.10% (8/195) | 0.32 (63 total) |

---

## 2. Per-Class Empirical Detection Metrics (Baseline Multi-Domain Model)

Evaluated at operational industrial confidence cutoff $\text{conf} \ge 0.25$ and $\text{IoU} = 0.50$ across the 2,125 multi-domain validation set:

| Class ID | Defect Class Name | Severity Category | Ground Truth Bounding Boxes | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision (%) | Recall (%) | F1-Score | Average Precision (AP@0.50) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | **Open Circuit** | **Critical Electrical** | 584 | 504 | 590 | 80 | 46.07% | 86.30% | 0.6007 | 78.42% |
| **1** | **Short Circuit** | **Critical Electrical** | 403 | 332 | 122 | 71 | 73.13% | 82.38% | 0.7748 | 84.15% |
| **2** | **Mouse Bite** | **Major Geometric** | 839 | 612 | 413 | 227 | 59.71% | 72.94% | 0.6567 | 74.88% |
| **3** | **Spur** | **Major Geometric** | 1,178 | 692 | 264 | 486 | 72.38% | 58.74% | 0.6485 | 69.95% |
| **4** | **Spurious Copper** | **Major Geometric** | 521 | 424 | 162 | 97 | 72.35% | 81.38% | 0.7660 | 82.90% |
| **5** | **Pin Hole / Missing Pad** | **Critical Mechanical** | 906 | 870 | 143 | 36 | 85.88% | 96.03% | 0.9067 | 89.81% |
| **Total / Mean** | **All 6 Canonical Defect Classes** | — | **4,431** | **3,434** | **1,694** | **997** | **66.97%** | **77.50%** | **0.7185** | **80.02%** |

---

## 3. Defect Area Sensitivity & Scale Granularity

Evaluated across the standardized COCO area definitions:
- **Small Defects:** Area $< 32 \times 32\text{ px}$ ($< 1,024\text{ px}^2$)
- **Medium Defects:** Area between $32 \times 32\text{ px}$ and $96 \times 96\text{ px}$ ($1,024\text{ to }9,216\text{ px}^2$)

| Defect Granularity Scale | Total Defect Instances | Detected True Positives (TP) | Detection Recall (%) | Primary Failure Mode |
|:---|:---:|:---:|:---:|:---|
| **Small Defect Instances ($< 32 \times 32\text{ px}$)** | 1,860 | 1,172 | **63.01%** | Feature map sub-sampling at P5 stride ($32\times$) |
| **Medium Defect Instances ($32 \times 32\text{ to }96 \times 96\text{ px}$)** | 2,571 | 2,262 | **87.98%** | High fidelity across P3 ($8\times$) and P4 ($16\times$) FPN heads |
| **Overall Aggregate** | **4,431** | **3,434** | **77.50%** | F1-Score: 0.7185 |

---

## 4. Industrial Clean Board Overkill & Yield Impact

Evaluated across 195 clean, defect-free DeepPCB reference template boards (`group90100` and `group92000`):

| Evaluation Metric | Baseline Multi-Domain Reference | Clean Exp C2 (Color Negative Mining) | Industrial Benefit / Delta |
|:---|:---:|:---:|:---:|
| **Clean Reference Boards Evaluated** | 195 boards | 195 boards | Fixed evaluation benchmark |
| **Clean PASS Verdicts** | **96 boards (49.23%)** | **182 boards (93.33%)** | **+44.10% absolute yield improvement** |
| **False INSPECT (Human Review)** | 80 boards (41.03%) | 12 boards (6.15%) | 6.67× reduction in unnecessary manual reviews |
| **False REJECT (Spurious Scrap)** | **19 boards (9.74%)** | **1 board (0.51%)** | **19.0× reduction in scrap waste** |
| **Total Spurious Defect BBoxes** | **357 bounding boxes** | **15 bounding boxes** | **95.80% reduction in false-positive overkill** |
| **Average False BBoxes per Board** | 1.83 boxes / board | 0.08 boxes / board | Sub-0.1 FP rate per board |
| **Dominant False Alarm Class** | Normal via holes $\to$ `pin_hole_missing_pad` (246 boxes; 68.9%) | Normal via holes suppressed by negative copper mining | Eliminates via-hole overkill |

---

## 5. End-to-End Latency Budget & Execution Timings

| Inspection Subsystem / Pipeline Stage | Hardware Component | Measured Latency (Host AMD Ryzen 9 8940HX) | Projected Latency (Raspberry Pi 5 ARM64 CPU) | Projected Latency (Hailo-8 NPU 26 TOPS) | Industrial Budget (30 FPS AOI) | Headroom / Margin |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Optical Capture** | Sony IMX708 AF Camera (1080p) | 0.00 ms (disk file) | 12.50 ms | 12.50 ms | 33.33 ms | +62.5% |
| **2. Preprocessing & Quality Gate** | OpenCV Letterbox (640x640x3) | **0.99 ms** | 2.10 ms | 2.10 ms | 5.00 ms | +58.0% |
| **3. Vision Inference** | YOLOv8n (FP32 ONNX / INT8 HEF) | **26.97 ms** | 39.50 ms | **5.80 ms** | 25.00 ms | **+76.8% (NPU)** |
| **4. Vectorized Post-Processing** | DFL Decoding & NMS (IoU 0.45) | **0.48 ms** | 1.40 ms | 1.40 ms | 5.00 ms | +72.0% |
| **5. Decision Rule Engine** | Deterministic IPC-A-610 Policy | **0.08 ms** | 0.12 ms | 0.12 ms | 2.00 ms | +94.0% |
| **6. SLM Fast Fallback Advisory** | Deterministic IPC Knowledge Base | **0.18 ms** | 0.35 ms | 0.35 ms | 5.00 ms | +93.0% |
| **7. Tamper-Evident SQLite Audit** | ACID WAL-Mode Database Log | **1.20 ms** | 2.50 ms | 2.50 ms | 10.00 ms | +75.0% |
| **Total Inspection Pipeline** | **End-to-End Frame Turnaround** | **29.90 ms (33.4 FPS)** | **58.47 ms (17.1 FPS)** | **24.77 ms (40.4 FPS)** | **100.00 ms** | **4.04× faster than budget** |

---

## 6. Small Language Model (SLM) Advisory Latency & Memory Footprint

Empirical benchmarking of Meta Llama 3.2 1B Instruct executed on host CPU:

| Metric | ExecuTorch FP16 Baseline | ExecuTorch 8da4w INT4 Model | Quantitative Comparison / Impact |
|:---|:---:|:---:|:---|
| **Model Artifact Size on Disk** | 2,472,432,290 bytes (2.30 GiB) | **1,689,109,888 bytes (1.57 GiB)** | **31.7% storage reduction (783.3 MB smaller)** |
| **Model Load / mmap Latency** | 2.10 s | **1.54 s** | 26.7% faster cold initialization |
| **Cold Time-to-First-Token (TTFT)** | 45,590.0 ms | **2,427.98 ms** | **18.78× faster cold prefill** |
| **Warmed-Up TTFT** | 20,930.0 ms | **853.21 ms** | **24.53× faster (Sub-second edge prefill)** |
| **Decode Throughput** | 1.05 – 1.36 tokens/s | **10.73 tokens/s** | **7.89× throughput acceleration** |
| **Total Generation Latency (56 tokens)** | 70,680.0 ms (~70.7s) | **9,225.5 ms (~9.2s)** | **7.66× faster explanation delivery** |
| **Peak Working RAM Utilization** | 3,030.0 MB (~2.96 GB) | **1,322.6 MB (~1.29 GB)** | **56.3% lower RAM footprint (Fits in Pi 5 8GB)** |
| **Quality Decision Overriding** | Strictly 0 instances | Strictly 0 instances | **100% Verdict Invariance Guaranteed** |

---

## 7. Model Optimization & Quantization Comparison

| Representation / Artifact | Format / Runtime | Precision | Disk Size (MB) | Compression vs FP32 | Execution Hardware Target | Measured / Projected Latency |
|:---|:---|:---:|:---:|:---:|:---|:---:|
| **YOLOv8n PyTorch Checkpoint** | PyTorch `.pt` | FP32 | 6.20 MB | 1.00× (Baseline) | Host CPU / CUDA GPU | 27.21 ms |
| **YOLOv8n ONNX Export** | ONNX Runtime | FP32 | 12.19 MB | 0.51× (Graph Expansion) | Host AMD64 / Pi 5 ARM64 CPU | 26.97 ms / 39.5 ms |
| **YOLOv8n Hailo-8 HEF** | HailoRT | INT8 Quantized | **4.35 MB** | **1.43× smaller than .pt** | Hailo-8 AI HAT+ (PCIe) | **5.80 ms (Projected)** |
| **Llama 3.2 1B FP16 PTE** | ExecuTorch XNNPACK | FP16 | 2,357.9 MB | 1.00× (Baseline) | Host CPU / Pi 5 ARM64 CPU | ~70.7 s total (1.1 tok/s) |
| **Llama 3.2 1B 8da4w PTE** | ExecuTorch XNNPACK | 8da4w INT4 | **1,610.9 MB** | **1.46× smaller than FP16** | Host CPU / Pi 5 ARM64 CPU | **9.2 s total (10.73 tok/s)** |
