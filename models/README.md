# EdgePCB-QC — Production Model Artifacts & Deployment Manifest
**Bharat AI-SoC Challenge 2026–27 | Problem Statement 5**

This directory defines the verified model artifacts for offline edge AI defect detection and operator advisory reasoning.

---

## 1. Artifact Verification & Cryptographic Provenance

| Artifact Role | File Path | File Size | SHA-256 Cryptographic Hash | Distribution Status | Target Runtime |
|:---|:---|:---:|:---|:---:|:---|
| **Hailo-8 NPU Vision Model** | `models/hailo/pcb_yolov8n_raw_heads.hef` | 4,557,262 bytes (4.35 MB) | `053a1967ad9489847ee80e058a4608518713706baf2fa79915159d952df3dada` | **Bundled in Repo** | Raspberry Pi AI HAT+ (HailoRT 4.24.0) |
| **CPU Fallback Vision Model** | `models/onnx/pcb_multidomain_yolov8n_640.onnx` | 12,778,213 bytes (12.19 MB) | `0f219cae19807720f6528d635fc7168e1c067a91370e59d165a79c7536089662` | **Bundled in Repo** | Host ARM64 / x86_64 CPU (ONNX Runtime 1.20) |
| **Llama 3.2 BPE Tokenizer** | `models/llama/tokenizer.model` | 2,183,982 bytes (2.08 MB) | `82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55` | **Bundled in Repo** | Tiktoken BPE / ExecuTorch |
| **Llama 3.2 1B ExecuTorch Binary** | `models/llama/llama3_2_1b_xnnpack_8da4w.pte` | 1,689,109,888 bytes (1.57 GiB) | `2de22a466571e7e0cdd0adf01d72bed5f89d53d14784f759c06fd597616a9791` | **Git-Excluded (1.57 GiB)** | Host ARM Cortex-A76 CPU (ExecuTorch 1.6.0 XNNPACK) |

---

## 2. Bundled Models vs. Git File Size Limitations

- **Vision Models (`.hef` and `.onnx`):** Both production vision models are bundled directly in this repository. Anyone cloning the repository can immediately execute YOLOv8n defect inspection out-of-the-box on either the Hailo-8 NPU or host CPU.
- **Small Language Model (`.pte`):** GitHub enforces a strict 100 MB per-file upload limit. The 1.57 GiB compiled ExecuTorch binary (`llama3_2_1b_xnnpack_8da4w.pte`) is intentionally excluded from Git commits via `.gitignore`.

---

## 3. How to Place or Export the Llama ExecuTorch Model

### Option A: Use the Pre-Packaged Release Asset
Download `llama3_2_1b_xnnpack_8da4w.pte` from the official Challenge Phase 1 release bundle and place it at:
```bash
models/llama/llama3_2_1b_xnnpack_8da4w.pte
```
Verify the integrity hash:
```bash
sha256sum models/llama/llama3_2_1b_xnnpack_8da4w.pte
# Expected output: 2de22a466571e7e0cdd0adf01d72bed5f89d53d14784f759c06fd597616a9791
```

### Option B: Export On-Device from Meta Weights
If you have access to Hugging Face or Meta Llama 3.2 1B Instruct checkpoints:
```bash
# 1. Set your Hugging Face authentication token
export HF_TOKEN="your_huggingface_token"

# 2. Run the audited ExecuTorch export script
bash advisory/export_xnnpack_8da4w.sh
```
The export pipeline will:
1. Load `meta-llama/Llama-3.2-1B-Instruct` in PyTorch.
2. Apply `torchao` 8-bit dynamic activation and 4-bit per-channel weight quantization (`8da4w`).
3. Compile with the ExecuTorch XNNPACK delegate targeting ARM NEON / Cortex-A76.
4. Output the `.pte` artifact directly into `models/llama/`.

---

## 4. Deterministic Safety Fallback Guarantee

If `llama3_2_1b_xnnpack_8da4w.pte` is not placed locally:
- The EdgePCB-QC station **remains 100% operational**.
- Primary vision inspection (**Pipeline A**) executes without interruption.
- Operator guidance automatically activates the **Deterministic IPC-A-610 Expert Fallback Engine** (`advisory/fallback.py`), generating instant (<0.1 ms) grounded troubleshooting advice based strictly on localized defect classes and severities.
- The inspection verdict (`PASS` / `REJECT` / `INSPECT`) is governed entirely by the deterministic decision engine and is **100% invariant** to the presence or absence of the language model.
