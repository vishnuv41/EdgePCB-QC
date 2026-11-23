"""
Frozen YOLOv8n Checkpoint ONNX Exporter & Validator for Hailo Compilation
Bharat AI-SoC Challenge 2026-27 | PS5

Prepares the frozen Phase 1 baseline checkpoint:
models/checkpoints/pcb_multidomain_experiment/weights/best.pt
(Expected SHA-256: 66d8d225fe2137c160b8038d0e68200a54d1544152b0941050f749ee0741be04)

Exports to:
models/onnx/pcb_multidomain_yolov8n_640.onnx
with static 640x640 input shape, ONNX opset 12 (Hailo-compatible),
and validates inference via onnxruntime.
"""

import sys
import hashlib
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_WEIGHTS = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
OUTPUT_DIR = WORKSPACE_ROOT / "models" / "onnx"
OUTPUT_ONNX = OUTPUT_DIR / "pcb_multidomain_yolov8n_640.onnx"
EXPECTED_SHA256 = "66d8d225fe2137c160b8038d0e68200a54d1544152b0941050f749ee0741be04"


def calculate_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_and_validate():
    print("=" * 75)
    print("EDGEPCB-QC: FROZEN BASELINE TO ONNX EXPORTER")
    print("=" * 75)

    if not BASELINE_WEIGHTS.exists():
        print(f"[ERROR] Baseline checkpoint not found: {BASELINE_WEIGHTS}")
        sys.exit(1)

    # 1. Verify frozen checkpoint checksum
    actual_sha256 = calculate_sha256(BASELINE_WEIGHTS)
    print(f"[*] Baseline Checkpoint : {BASELINE_WEIGHTS.name}")
    print(f"[*] Verified SHA-256    : {actual_sha256}")

    if actual_sha256.lower() != EXPECTED_SHA256.lower():
        print(f"[ERROR] SHA-256 mismatch! Expected: {EXPECTED_SHA256}, Got: {actual_sha256}")
        print("Export aborted to preserve baseline model integrity.")
        sys.exit(1)
    print("[PASS] Frozen checkpoint SHA-256 verification successful.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Export using Ultralytics
    print("\n[*] Exporting to ONNX (imgsz=640, opset=12, dynamic=False)...")
    from ultralytics import YOLO
    model = YOLO(str(BASELINE_WEIGHTS))

    exported_path_str = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        dynamic=False,
        simplify=True
    )
    exported_path = Path(exported_path_str)

    # Move or copy to target location
    if exported_path != OUTPUT_ONNX:
        import shutil
        shutil.copy2(exported_path, OUTPUT_ONNX)
        print(f"[+] Saved ONNX model to: {OUTPUT_ONNX}")

    # 3. Validate ONNX structure
    print("\n[*] Validating ONNX structure...")
    import onnx
    onnx_model = onnx.load(str(OUTPUT_ONNX))
    onnx.checker.check_model(onnx_model)
    print("[PASS] ONNX checker passed cleanly.")

    # 4. Validate runtime with onnxruntime
    print("[*] Testing inference session with onnxruntime...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(OUTPUT_ONNX), providers=["CPUExecutionProvider"])
    
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    print(f"    Input Name : {inputs[0].name}, Shape: {inputs[0].shape}, Type: {inputs[0].type}")
    print(f"    Output Name: {outputs[0].name}, Shape: {outputs[0].shape}, Type: {outputs[0].type}")

    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    output_tensors = session.run(None, {inputs[0].name: dummy_input})

    print(f"[PASS] ONNX test inference passed. Output tensor shape: {output_tensors[0].shape}")
    onnx_sha = calculate_sha256(OUTPUT_ONNX)
    print(f"[*] Exported ONNX SHA-256: {onnx_sha}")
    print("\n[SUCCESS] Model ready for Hailo Dataflow Compiler (DFC) compilation to HEF.")


if __name__ == "__main__":
    export_and_validate()
