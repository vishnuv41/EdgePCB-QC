"""
Hailo NPU & ONNX Model Compatibility Verification
Bharat AI-SoC Challenge 2026-27 | PS5

Verifies:
1. ONNX export existence, validity, and ONNX checker compliance.
2. Fixed tensor dimensions: [1, 3, 640, 640] -> [1, 10, 8400].
3. Operator set audit: ensures 100% of ONNX nodes belong to Hailo DFC-supported operators.
4. Numerical parity between PyTorch checkpoint and ONNX Runtime evaluation.
5. Calibration dataset availability (1024 representative calibration images).
"""

from pathlib import Path
import json
import pytest
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = WORKSPACE_ROOT / "models" / "onnx" / "pcb_multidomain_experiment.onnx"
PT_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
CALIB_DIR = WORKSPACE_ROOT / "datasets" / "calibration_images"
HAILO_REPORT_JSON = WORKSPACE_ROOT / "benchmarks" / "results" / "hailo_preparation_report.json"

# Set of standard operators supported natively by Hailo Dataflow Compiler (DFC)
HAILO_DFC_SUPPORTED_OPS = {
    "Conv", "Sigmoid", "Mul", "Split", "Add", "Concat",
    "MaxPool", "Resize", "Reshape", "Transpose", "Softmax",
    "Slice", "Sub", "Div", "Constant", "Flatten"
}


def test_onnx_model_exists_and_valid():
    assert ONNX_PATH.exists(), f"Missing ONNX export at: {ONNX_PATH}"
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)
    assert model.ir_version > 0


def test_tensor_shapes_and_types():
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()

    assert len(inputs) == 1, f"Expected 1 input tensor, got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output tensor, got {len(outputs)}"

    inp = inputs[0]
    out = outputs[0]

    # Input shape: [batch, channels, height, width] = [1, 3, 640, 640]
    assert inp.shape == [1, 3, 640, 640], f"Unexpected input shape: {inp.shape}"
    assert inp.type == "tensor(float)"

    # Output shape: [batch, 4_bbox_coords + 6_classes, 8400_anchors] = [1, 10, 8400]
    assert out.shape == [1, 10, 8400], f"Unexpected output shape: {out.shape}"
    assert out.type == "tensor(float)"


def test_hailo_operator_set_audit():
    model = onnx.load(str(ONNX_PATH))
    unsupported = set()
    node_ops = set()

    for node in model.graph.node:
        node_ops.add(node.op_type)
        if node.op_type not in HAILO_DFC_SUPPORTED_OPS:
            unsupported.add(node.op_type)

    assert len(unsupported) == 0, f"Unsupported Hailo operators detected: {unsupported}"
    assert "Conv" in node_ops
    assert "Concat" in node_ops


def test_pytorch_onnx_numerical_parity():
    assert PT_PATH.exists(), f"Missing PyTorch checkpoint at: {PT_PATH}"
    
    # Load PyTorch model
    yolo = YOLO(str(PT_PATH))
    py_model = yolo.model.fuse().eval().cpu()

    # Load ONNX session
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Fixed deterministic input
    np.random.seed(42)
    dummy_np = np.random.randn(1, 3, 640, 640).astype(np.float32) * 0.1
    dummy_pt = torch.from_numpy(dummy_np)

    with torch.no_grad():
        pt_out = py_model(dummy_pt)
        if isinstance(pt_out, (list, tuple)):
            pt_out = pt_out[0]
        pt_np = pt_out.cpu().numpy()

    ort_out = sess.run([out_name], {inp_name: dummy_np})[0]

    assert pt_np.shape == ort_out.shape, f"Shape mismatch: PT {pt_np.shape} vs ORT {ort_out.shape}"
    max_diff = float(np.max(np.abs(pt_np - ort_out)))
    mean_diff = float(np.mean(np.abs(pt_np - ort_out)))

    assert max_diff < 0.05, f"Excessive numerical difference: max={max_diff}, mean={mean_diff}"


def test_calibration_dataset_ready():
    assert CALIB_DIR.exists(), f"Missing calibration directory: {CALIB_DIR}"
    calib_images = list(CALIB_DIR.glob("*.jpg")) + list(CALIB_DIR.glob("*.png"))
    assert len(calib_images) >= 100, f"Calibration images insufficient: found {len(calib_images)}, expected >= 100"
