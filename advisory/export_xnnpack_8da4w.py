"""
ExecuTorch Llama 3.2 1B Instruct XNNPACK 8da4w Python Export Utility
EdgePCB-QC (Bharat AI-SoC Challenge 2026-27 | PS5)

Quantizes Llama 3.2 1B Instruct using torchao 8da4w (8-bit dynamic activation + 4-bit weights)
with XNNPACK backend delegate for ARM Cortex-A76 (Raspberry Pi 5) and x86 CPU.
"""

import os
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = WORKSPACE_ROOT / "models" / "llama" / "checkpoints" / "Llama-3.2-1B-Instruct"
CHECKPOINT_PATH = LLAMA_DIR / "original" / "consolidated.00.pth"
PARAMS_PATH = LLAMA_DIR / "original" / "params.json"
OUTPUT_NAME = "llama3_2_1b_xnnpack_8da4w.pte"


def export_xnnpack_8da4w():
    print("=" * 70)
    print("EXECUTORCH LLAMA 3.2 1B XNNPACK 8DA4W EXPORT PIPELINE")
    print("=" * 70)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Params not found: {PARAMS_PATH}")

    import torch
    import executorch
    from executorch.examples.models.llama import export_llama_lib

    print(f"[*] Checkpoint:   {CHECKPOINT_PATH}")
    print(f"[*] Params:       {PARAMS_PATH}")
    print(f"[*] Backend:      XNNPACK (-X)")
    print(f"[*] Quantization: 8da4w (-qmode 8da4w)")

    extra_opts = f"-kv --max_seq_length 256 --max_context_length 512 -X -qmode 8da4w -n {OUTPUT_NAME}"
    t0 = time.perf_counter()

    export_llama_lib.build_model(
        model="llama3_2",
        checkpoint=CHECKPOINT_PATH.resolve().as_posix(),
        params=PARAMS_PATH.resolve().as_posix(),
        output_dir=LLAMA_DIR.resolve().as_posix(),
        extra_opts=extra_opts
    )

    t_export = time.perf_counter() - t0
    out_file = LLAMA_DIR / OUTPUT_NAME
    if out_file.exists():
        size_mb = out_file.stat().st_size / (1024 * 1024)
        print(f"[+] Successfully compiled {OUTPUT_NAME} in {t_export:.2f}s ({size_mb:.1f} MB)!")
    else:
        print(f"[+] Compilation completed in {t_export:.2f}s.")


if __name__ == "__main__":
    export_xnnpack_8da4w()
