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
OUTPUT_DIR = WORKSPACE_ROOT / "models" / "llama"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_NAME = "llama3_2_1b_xnnpack_8da4w.pte"

CANDIDATE_CHECKPOINTS = [
    OUTPUT_DIR / "checkpoints" / "Llama-3.2-1B-Instruct" / "original" / "consolidated.00.pth",
    Path(r"E:\EdgePCB-QC_cache\models\llama3_2_1b\original\consolidated.00.pth"),
    OUTPUT_DIR / "original" / "consolidated.00.pth",
]

CANDIDATE_PARAMS = [
    OUTPUT_DIR / "checkpoints" / "Llama-3.2-1B-Instruct" / "original" / "params.json",
    Path(r"E:\EdgePCB-QC_cache\models\llama3_2_1b\original\params.json"),
    OUTPUT_DIR / "original" / "params.json",
]


def find_file(candidates, label):
    for c in candidates:
        if c.exists():
            print(f"[+] Found {label}: {c} ({c.stat().st_size / (1024*1024):.1f} MB)")
            return c
    raise FileNotFoundError(f"Could not locate {label} in candidate paths: {candidates}")


def export_xnnpack_8da4w():
    print("=" * 70)
    print("EXECUTORCH LLAMA 3.2 1B XNNPACK 8DA4W EXPORT PIPELINE")
    print("=" * 70)

    checkpoint_path = find_file(CANDIDATE_CHECKPOINTS, "PyTorch checkpoint")
    params_path = find_file(CANDIDATE_PARAMS, "params.json")

    # Set FLATC_EXECUTABLE so flatbuffer compilation succeeds on Windows
    scripts_flatc = Path(sys.prefix) / "Scripts" / "flatc.exe"
    bin_flatc = Path(sys.prefix) / "Lib" / "site-packages" / "executorch" / "data" / "bin" / "flatc.exe"
    if "FLATC_EXECUTABLE" not in os.environ:
        if scripts_flatc.exists():
            os.environ["FLATC_EXECUTABLE"] = str(scripts_flatc)
        elif bin_flatc.exists():
            os.environ["FLATC_EXECUTABLE"] = str(bin_flatc)
    os.environ["PATH"] = str(Path(sys.prefix) / "Scripts") + os.pathsep + os.environ.get("PATH", "")
    print(f"[*] FLATC_EXECUTABLE: {os.environ.get('FLATC_EXECUTABLE')}")

    import torch
    import executorch
    from executorch.examples.models.llama import export_llama_lib

    print(f"[*] Checkpoint:   {checkpoint_path}")
    print(f"[*] Params:       {params_path}")
    print(f"[*] Output Dir:   {OUTPUT_DIR}")
    print(f"[*] Output File:  {OUTPUT_NAME}")
    print(f"[*] Backend:      XNNPACK (-X)")
    print(f"[*] Quantization: 8da4w (-qmode 8da4w: 8-bit act, 4-bit weight)")

    # extra_opts string passed to argument parser
    extra_opts = f"-kv --max_seq_length 256 --max_context_length 512 -X -qmode 8da4w -n {OUTPUT_NAME}"
    print(f"[*] Extra options: {extra_opts}")
    t0 = time.perf_counter()

    out_path = export_llama_lib.build_model(
        model="llama3_2",
        checkpoint=checkpoint_path.resolve().as_posix(),
        params=params_path.resolve().as_posix(),
        output_dir=OUTPUT_DIR.resolve().as_posix(),
        extra_opts=extra_opts
    )

    t_export = time.perf_counter() - t0
    final_pte = OUTPUT_DIR / OUTPUT_NAME
    if final_pte.exists():
        size_mb = final_pte.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 70)
        print(f"[✓] EXPORT SUCCESSFUL: {final_pte.name}")
        print(f"    Path:    {final_pte}")
        print(f"    Size:    {size_mb:.2f} MB ({final_pte.stat().st_size:,} bytes)")
        print(f"    Elapsed: {t_export:.2f} s")
        print("=" * 70)
        return final_pte
    else:
        print(f"[!] Compilation finished in {t_export:.2f}s, returned: {out_path}")
        return Path(out_path) if out_path else None


if __name__ == "__main__":
    export_xnnpack_8da4w()
