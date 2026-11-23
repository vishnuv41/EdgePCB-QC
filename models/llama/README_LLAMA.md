# Meta Llama 3.2 1B Instruct ExecuTorch Deployment Directory
**EdgePCB-QC Industrial AOI System | Raspberry Pi 5 (8 GB)**

This directory holds the Small Language Model (SLM) artifacts for on-device IPC-A-610 engineering troubleshooting and disposition advice.

## Target ExecuTorch Model Binaries

### 1. Recommended Production Model: 8da4w Quantized (XNNPACK)
- **File**: `llama3_2_1b_xnnpack_8da4w.pte`
- **Path**: `models/llama/llama3_2_1b_xnnpack_8da4w.pte`
- **Quantization**: 8-bit dynamic activations + 4-bit weights (`8da4w` via `torchao`)
- **Backend Delegate**: XNNPACK CPU Delegate (optimized with ARM NEON / Dot Product for Cortex-A76)
- **Expected File Size**: ~600–750 MB (70% reduction vs FP16)
- **Runtime Memory (RAM)**: ~650–850 MB (ideal for Raspberry Pi 5 8 GB)
- **Tokenizer**: `models/llama/tokenizer.model` (2.08 MB, bundled)

### 2. Full Precision Baseline: FP16 (Half Precision)
- **File**: `llama3_2_1b_fp16.pte`
- **Path**: `models/llama/llama3_2_1b_fp16.pte`
- **Quantization**: Unquantized FP16
- **File Size**: ~2.30 GB (2,472,432,290 bytes)
- **Runtime Memory (RAM)**: ~3.03 GB peak RSS

---

## Dual-Pipeline Architecture & Safety Governance

1. **Pipeline A (Real-Time Vision & Deterministic Gating)**:
   - YOLOv8n running on Hailo-8 NPU (<15 ms) or ONNX CPU baseline (<40 ms).
   - High-speed conveyor camera inspection with strict deterministic IPC-A-610 Class 2/3 rules.
   - **Quality verdicts (PASS / INSPECT / REJECT) are 100% deterministic and hardware-gated.**

2. **Pipeline B (Engineering Advisory & Root Cause)**:
   - On-demand / asynchronous Small Language Model (SLM) inference.
   - Llama receives structured inspection findings (defect classes, bounding boxes, rule triggers, quality verdict).
   - Generates natural language root-cause hypotheses, tooling verification steps, and IPC standard citations.
   - **Safety Invariance**: Llama **NEVER** decides or overrides the inspection verdict.

3. **Deterministic Fallback Engine**:
   - If ExecuTorch `.pte` is missing, uninitialized, or exceeds latency limits, the system instantly (<1 ms) falls back to the deterministic IPC-A-610 expert rules engine.
   - Zero crash risk: 100% offline, air-gapped, and resilient.

