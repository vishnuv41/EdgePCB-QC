#!/usr/bin/env bash
# ==============================================================================
# ExecuTorch Llama 3.2 1B Instruct XNNPACK 8da4w Export Script (WSL2 / Linux)
# EdgePCB-QC (Bharat AI-SoC Challenge 2026-27 | PS5)
#
# Target: ARM Cortex-A76 (Raspberry Pi 5) / x86-64 CPU
# Quantization: 8da4w (8-bit Dynamic Activation + 4-bit Groupwise Weights via torchao)
# Footprint Reduction: ~2.4 GB FP16 -> ~650 MB Int4
# ==============================================================================

set -euo pipefail

echo "======================================================================="
echo "EXECUTORCH LLAMA 3.2 1B XNNPACK 8DA4W EXPORT PIPELINE"
echo "======================================================================="

# 1. Environment & Path Resolution
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT_DIR="${WORKSPACE_DIR}/models/llama/checkpoints/Llama-3.2-1B-Instruct"
CHECKPOINT_FILE="${CHECKPOINT_DIR}/original/consolidated.00.pth"
PARAMS_FILE="${CHECKPOINT_DIR}/original/params.json"
OUTPUT_DIR="${WORKSPACE_DIR}/models/llama/checkpoints/Llama-3.2-1B-Instruct"
OUTPUT_NAME="llama3_2_1b_xnnpack_8da4w.pte"

if [ ! -f "${CHECKPOINT_FILE}" ]; then
    echo "[!] Checkpoint file not found: ${CHECKPOINT_FILE}"
    echo "    Please verify model weights are present."
    exit 1
fi

if [ ! -f "${PARAMS_FILE}" ]; then
    echo "[!] Params file not found: ${PARAMS_FILE}"
    exit 1
fi

echo "[*] Workspace:    ${WORKSPACE_DIR}"
echo "[*] Checkpoint:   ${CHECKPOINT_FILE}"
echo "[*] Params:       ${PARAMS_FILE}"
echo "[*] Target PTE:   ${OUTPUT_DIR}/${OUTPUT_NAME}"

# 2. Verify Python dependencies
python3 -c "import executorch, torch, torchao; print('[+] ExecuTorch:', executorch.__version__, '| PyTorch:', torch.__version__, '| TorchAO:', torchao.__version__)"

# 3. Compile via ExecuTorch export_llama_lib with XNNPACK and 8da4w quantization
echo "[*] Compiling with XNNPACK 8da4w quantization..."
python3 -m executorch.examples.models.llama.export_llama_lib \
    --model llama3_2 \
    --checkpoint "${CHECKPOINT_FILE}" \
    --params "${PARAMS_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    -kv \
    --max_seq_length 256 \
    --max_context_length 512 \
    -X \
    -qmode 8da4w \
    -n "${OUTPUT_NAME}"

echo "======================================================================="
echo "[+] SUCCESS: XNNPACK 8da4w artifact compiled at:"
echo "    ${OUTPUT_DIR}/${OUTPUT_NAME}"
ls -lh "${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "======================================================================="
