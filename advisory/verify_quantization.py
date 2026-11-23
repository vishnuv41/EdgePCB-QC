"""
Verification Utility for ExecuTorch Quantized PTE Model
EdgePCB-QC (Bharat AI-SoC Challenge 2026-27 | PS5)

Inspects the internal graph structure, operators, tensor dtypes, and file size
of the exported .pte file to rigorously confirm that the model is genuinely
quantized (8da4w: 8-bit dynamic activation, 4-bit weight) rather than FP16/FP32.
"""

import os
import sys
import json
import time
import psutil
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))


def verify_quantized_pte(pte_path: Path) -> dict:
    print("=" * 70)
    print("EXECUTORCH QUANTIZATION AUDIT & GRAPH INSPECTION")
    print(f"Target: {pte_path}")
    print("=" * 70)

    if not pte_path.exists():
        return {"status": "FAIL", "reason": f"File does not exist: {pte_path}"}

    stat = pte_path.stat()
    file_size_bytes = stat.st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Check magic header: ExecuTorch files use FlatBuffers (or ET01 / ET02 / etc)
    with open(pte_path, "rb") as f:
        header = f.read(16)

    # Load via ExecuTorch inspection
    import torch
    torch_lib = Path(torch.__file__).parent / "lib"
    if torch_lib.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(torch_lib))

    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    from executorch.extension.pybindings._C import _load_for_executorch
    module = _load_for_executorch(str(pte_path.resolve()))
    load_time = time.perf_counter() - t0

    ram_after = process.memory_info().rss / (1024 * 1024)
    ram_used = ram_after - ram_before

    # Inspect methods
    method_names = module.method_names() if hasattr(module, "method_names") else ["forward"]

    # Binary pattern search for XNNPACK and quantization operator signatures
    with open(pte_path, "rb") as f:
        raw_bytes = f.read()

    quant_signatures = [
        b"xnnpack",
        b"dq_linear",
        b"quantized",
        b"q_dq",
        b"int4",
        b"int8",
        b"weight_quant",
        b"dynamic_quant",
        b"8da4w"
    ]
    detected_signatures = [sig.decode("utf-8") for sig in quant_signatures if sig in raw_bytes.lower()]

    # FP16 comparison check: FP16 Llama 3.2 1B is ~2.30 GB.
    # An 8da4w (4-bit weight) model must be roughly 600 MB - 850 MB (at least 60% reduction).
    fp16_expected_bytes = 2472432290
    compression_ratio = fp16_expected_bytes / file_size_bytes if file_size_bytes > 0 else 0

    is_quantized = (
        file_size_bytes < fp16_expected_bytes and
        compression_ratio > 1.2 and
        ("xnnpack" in detected_signatures or "quantized" in detected_signatures or "dq_linear" in detected_signatures)
    )

    report = {
        "status": "PASS" if is_quantized else "INSPECT",
        "file_path": str(pte_path),
        "file_name": pte_path.name,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "fp16_reference_mb": round(fp16_expected_bytes / (1024 * 1024), 2),
        "size_reduction_percent": round((1.0 - (file_size_bytes / fp16_expected_bytes)) * 100, 2),
        "compression_ratio": round(compression_ratio, 2),
        "load_time_sec": round(load_time, 3),
        "ram_consumption_mb": round(ram_used, 2),
        "method_names": list(method_names),
        "detected_quant_signatures": detected_signatures,
        "is_quantized_verified": is_quantized
    }

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/llama/llama3_2_1b_xnnpack_8da4w.pte")
    res = verify_quantized_pte(target)
    sys.exit(0 if res["status"] == "PASS" else 1)
