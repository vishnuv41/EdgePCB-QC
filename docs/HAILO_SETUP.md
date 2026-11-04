# Hailo NPU Deployment Guide — EdgePCB-QC
*Bharat AI-SoC Challenge 2026–27 | PS5*

This document specifies the deployment architecture, compilation pipeline, and runtime integration for running **EdgePCB-QC** on a Raspberry Pi 5 equipped with the Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L accelerator).

---

## 1. Target Hardware Specifications

| Component | Target Specification | Interface / Architecture |
|---|---|---|
| **Host Single-Board Computer** | Raspberry Pi 5 (8GB LPDDR4X) | Broadcom BCM2712, Quad-core ARM Cortex-A76 @ 2.4 GHz |
| **Edge AI Accelerator** | Raspberry Pi AI HAT+ | M.2 M-key PCIe Gen2/Gen3 (x1 lane) |
| **NPU Architecture** | Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) | Structure-defined dataflow architecture |
| **Camera Sensor** | Raspberry Pi Camera Module 3 (Wide / Macro) | Sony IMX708 (11.9 MP, autofocus, I2C + CSI-2 2-lane) |
| **Lighting** | Ring / Diffused LED illuminator | High CRI (>= 90), non-flicker DC supply |
| **Power Supply** | Official 27W USB-C Power Supply | 5.1V / 5.0A with PD capability |
| **Cooling** | Raspberry Pi Active Cooler | Aluminum heatsink + PWM blower fan |

---

## 2. Software Architecture & Operating Environment

### Host Operating System (Target Pi 5)
- **OS:** Raspberry Pi OS 64-bit (Debian 12 Bookworm, Kernel 6.6+).
- **PCIe Enablement:** Ensure PCIe Gen3 is enabled in `/boot/firmware/config.txt`:
  ```ini
  dtparam=pciex1
  dtparam=pciex1_gen=3
  ```

### NPU Software Stack
- **HailoRT (Runtime):** v4.18.0 or later (ARM64 prebuilt Debian packages).
  - Driver: `hailo-all` / `hailort-pcie-driver`
  - Python Bindings: `python3-hailort`
- **Compiler Host (Development Workstation):**
  - **Hailo Dataflow Compiler (DFC):** v3.28.0+
  - **Environment Requirement:** x86-64 Linux (Ubuntu 22.04 LTS native or Docker container).
  - *Note:* DFC model quantization and `.hef` compilation cannot be performed on ARM SBCs or native Windows without a Linux container/host.

---

## 3. End-to-End Hailo Compilation Workflow (From PyTorch/ONNX to HEF)

```
[PyTorch Model / ONNX]
        │
        ▼ (hailo parser onnx)
[Hailo HAR: Model Translation]
        │
        ▼ (hailo optimize --calib-set images/)
[Quantized Int8 HAR (Post-Training Quantization)]
        │
        ▼ (hailo compiler --target hailo8 / hailo8l)
[Compiled .hef Binary Artifact]
        │
        ▼ (SCP to Raspberry Pi 5)
[Inference via HailoRT on Pi AI HAT+]
```

### Step 1: Export Clean ONNX Model
Export the trained detector with dynamic batch or static 640x640 shape:
```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model,
    dummy_input,
    "models/onnx/pcb_defect_detector.onnx",
    input_names=["images"],
    output_names=["output"],
    opset_version=17
)
```

### Step 2: Model Translation (x86-64 Linux Host)
```bash
# Activate Hailo virtual environment on Ubuntu 22.04
source hailo_env/bin/activate

# Parse ONNX model into Hailo Archive (HAR)
hailo parser onnx models/onnx/pcb_defect_detector.onnx \
    --har-path models/hailo/pcb_detector.har \
    --start-node-names images \
    --end-node-names output
```

### Step 3: Model Optimization & Quantization (PTQ)
Hailo uses integer 8-bit quantization (Int8) requiring a representative calibration set of 1,024 unlabelled industrial PCB crops:
- **Verified Multi-Domain Calibration Set:** Constructed at `datasets/calibration_images/` (1,024 JPEG frames, 57.6 MB) sampled proportionally from:
  - DeepPCB monochrome high-contrast scans: 410 images (40.0%)
  - DsPCBSD+ industrial color patches: 410 images (40.0%)
  - PCB-Defect V2 native-resolution 640x640 tiled crops: 204 images (20.0%)
- Also saved as normalized NumPy tensor batch: `E:/calib_set_1024x640x640.npy` (1024, 3, 640, 640).

```bash
# Run Hailo model optimizer with multi-domain calibration dataset
hailo optimize \
    --har-path models/hailo/pcb_detector.har \
    --calib-set-path datasets/calibration_images/ \
    --target hailo8 \
    --output-har-path models/hailo/pcb_detector_quantized.har
```

### Step 4: ONNX Verification & Operator Graph Audit (Verified in Phase 1)
As verified in `benchmarks/results/hailo_preparation_report.json`:
- **PyTorch vs ONNX Runtime Numerical Parity:** **PASSED** (Maximum absolute difference: $7.40 \times 10^{-4} < 10^{-3}$, Mean absolute error: $8.67 \times 10^{-6}$).
- **Graph Complexity:** 233 nodes across 14 unique operator types (`Conv`: 64, `Sigmoid`: 58, `Mul`: 58, `Split`: 8, `Add`: 8, `Concat`: 17, `MaxPool`: 3, `Resize`: 2, `Reshape`: 8, `Transpose`: 1, `Softmax`: 1, `Slice`: 2, `Sub`: 2, `Div`: 1).
- **Hailo DFC v3.28+ Compatibility:** **100% of operators are natively supported**. Zero custom or unsupported ops detected.
- **Zero Fabrication Statement:** All deployment latencies, frame rates, and power consumption figures on Hailo-8 are marked strictly as **PROJECTED** pending receipt of physical hardware in Phase 2. Zero synthetic or extrapolated edge metrics are claimed as measured.

### Step 5: HEF Compilation
Compile the quantized graph into a Hailo Executable Format (`.hef`):
```bash
hailo compiler \
    --har-path models/hailo/pcb_detector_quantized.har \
    --target hailo8 \
    --output-hef-path models/hailo/pcb_detector.hef
```

---

## 4. Raspberry Pi 5 Hardware Setup & Installation

On the physical Raspberry Pi 5:

```bash
# 1. Update system packages
sudo apt update && sudo apt full-upgrade -y

# 2. Install Raspberry Pi kernel headers and Hailo packages
sudo apt install -y raspberrypi-kernel-headers hailo-all

# 3. Verify PCIe device enumeration
lspci -nn | grep -i hailo
# Expected output: 
# 0000:01:00.0 Co-processor: Hailo Technologies Ltd. Hailo-8 AI Processor [1e60:2864]

# 4. Verify HailoRT command-line tool
hailortcli fw-control identify
# Expected: Displays Device Architecture (HAILO8 or HAILO8L), Serial number, Firmware version.
```

---

## 5. Software Runtime Integration (`HailoInspector`)

The EdgePCB-QC application integrates HailoRT using our abstract interface in `inspection/hailo_inference.py`:

```python
import numpy as np
from hailo_platform import VDevice, InferModel, ConfigureParams

class HailoInspector(VisionInspector):
    def __init__(self, hef_path="models/hailo/pcb_detector.hef"):
        self.hef_path = hef_path
        self.target = None

    def is_hardware_available(self) -> bool:
        try:
            from hailo_platform import Device
            return len(Device.scan()) > 0
        except Exception:
            return False

    def load_model(self):
        self.vdevice = VDevice()
        self.infer_model = self.vdevice.create_infer_model(self.hef_path)
        self.configured_model = self.infer_model.configure()

    def inspect(self, image: np.ndarray) -> list[dict]:
        # Preprocess 640x640 Int8 / Float32
        # Execute synchronous infer bindings
        # Post-process NMS bboxes and map to canonical classes
        pass
```

---

## 6. Benchmarking Protocol for Physical Deployment

When the physical Raspberry Pi 5 and AI HAT+ arrive, execute `benchmarks/benchmark_edge.py` to record:
1. **NPU Latency:** Time spent strictly inside HailoRT inference stream (target: `< 15 ms`).
2. **End-to-End Latency:** Camera capture -> Preprocessing -> Hailo NPU -> Decision Engine -> SQLite (target: `< 40 ms` / `> 25 FPS`).
3. **CPU Utilization:** Host Broadcom BCM2712 4-core utilization during live stream.
4. **NPU Utilization:** Hardware duty cycle percentage via `hailortcli monitor`.
5. **Thermals:** SoC and NPU temperatures under 10-minute continuous stream via `vcgencmd measure_temp`.
6. **Power Draw:** System current at 5.1V rail via USB-C power meter.
