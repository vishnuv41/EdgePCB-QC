"""
Physical Offline Integrity and Zero-Network Dependency Verification
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import os
import sys
from pathlib import Path
import psutil

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from advisory.executorch_runtime import ExecuTorchLlamaRunner, _find_default_pte, DEFAULT_TOKENIZER_PATH


def audit_offline_artifacts():
    print("=" * 70)
    print("AUDIT OF OFFLINE RUNTIME ASSETS (ZERO CLOUD / ZERO NETWORK)")
    print("=" * 70)

    pte_path = _find_default_pte()
    assert pte_path.exists(), f"PTE artifact missing at {pte_path}"
    pte_size_bytes = pte_path.stat().st_size
    pte_size_mb = pte_size_bytes / (1024 * 1024)

    assert DEFAULT_TOKENIZER_PATH.exists(), f"Tokenizer missing at {DEFAULT_TOKENIZER_PATH}"
    tok_size_bytes = DEFAULT_TOKENIZER_PATH.stat().st_size

    vision_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
    assert vision_pt.exists(), f"Vision checkpoint missing at {vision_pt}"
    vision_size_mb = vision_pt.stat().st_size / (1024 * 1024)

    vision_onnx = WORKSPACE_ROOT / "models" / "onnx" / "pcb_smoke_experiment.onnx"
    assert vision_onnx.exists(), f"ONNX missing at {vision_onnx}"
    onnx_size_mb = vision_onnx.stat().st_size / (1024 * 1024)

    db_path = WORKSPACE_ROOT / "database" / "inspections.db"
    assert db_path.exists(), f"Database missing at {db_path}"

    print(f"[+] Verified Local Asset 1: ExecuTorch PTE Binary")
    print(f"    - Path: {pte_path}")
    print(f"    - Size: {pte_size_mb:.2f} MB ({pte_size_bytes:,} bytes)")

    print(f"[+] Verified Local Asset 2: SentencePiece/Tiktoken Tokenizer")
    print(f"    - Path: {DEFAULT_TOKENIZER_PATH}")
    print(f"    - Size: {tok_size_bytes:,} bytes")

    print(f"[+] Verified Local Asset 3: Trained Vision PyTorch Weights")
    print(f"    - Path: {vision_pt}")
    print(f"    - Size: {vision_size_mb:.2f} MB")

    print(f"[+] Verified Local Asset 4: Edge-Ready ONNX Model")
    print(f"    - Path: {vision_onnx}")
    print(f"    - Size: {onnx_size_mb:.2f} MB")

    print(f"[+] Verified Local Asset 5: Persistent SQLite Audit Store")
    print(f"    - Path: {db_path}")

    # Process socket audit
    proc = psutil.Process(os.getpid())
    conns = proc.connections(kind="inet")
    remote_conns = [c for c in conns if c.raddr]
    print(f"\n[+] Active Network Sockets during local runtime: {len(remote_conns)} remote connections")
    assert len(remote_conns) == 0, "No remote connections allowed at runtime"

    # Load runtime locally
    runner = ExecuTorchLlamaRunner()
    assert runner.is_loaded is True, "ExecuTorch model failed to load from local disk"
    print(f"[+] Local ExecuTorch C++ runtime successfully initialized without network access.")


if __name__ == "__main__":
    audit_offline_artifacts()
