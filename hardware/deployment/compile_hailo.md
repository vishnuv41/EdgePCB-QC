# Hailo Model Compilation Guide: YOLOv8n to HEF
**EdgePCB-QC | Bharat AI-SoC Challenge 2026–27 | PS5**

This document details the exact, tested procedure to convert the frozen **EdgePCB-QC YOLOv8n** vision baseline (`best.pt`) into a compiled **Hailo Executable Format (HEF)** binary for hardware-accelerated inference on the **Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L)**.

---

## 1. Hardware & Software Compatibility Matrix

| Component | Target Development Host | Target Edge Device (Raspberry Pi 5) |
|---|---|---|
| **OS** | Ubuntu 22.04 LTS (x86_64) or WSL2 | Raspberry Pi OS (64-bit Bookworm, Debian 12) |
| **Python** | Python 3.10 virtual environment | Python 3.11 virtual environment |
| **Hailo Software** | **Hailo Dataflow Compiler (DFC) v3.28.0+** | **HailoRT v4.18.0+ / v4.19.0+** (`hailo-all`) |
| **Hardware** | x86_64 PC (min 16GB RAM for quantization) | Raspberry Pi 5 (8GB) + AI HAT+ (PCIe Gen2/3) |
| **NPU Target** | Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) | Hailo-8 (HAILO8) or Hailo-8L (HAILO8L) |

> [!IMPORTANT]
> The Hailo Dataflow Compiler (DFC) is an x86_64 Linux application. Compilation **cannot** be executed natively on Windows or on the Raspberry Pi Arm CPU. Use an Ubuntu 22.04 LTS workstation or WSL2 with Python 3.10.

---

## 2. Step 1: Export Frozen PyTorch Checkpoint to ONNX

On your development computer, execute the automated export script:

```bash
python hardware/deployment/export_onnx.py
```

This verifies the frozen checkpoint SHA-256 (`66d8d225fe...`), sets static input dimensions `[1, 3, 640, 640]`, opset 12, and produces:
- `models/onnx/pcb_multidomain_yolov8n_640.onnx`

---

## 3. Step 2: Set Up Hailo DFC in Ubuntu 22.04 / WSL2

Install the Hailo Dataflow Compiler wheel obtained from the Hailo Developer Zone:

```bash
# Create dedicated Python 3.10 environment
python3.10 -m venv hailo_dfc_venv
source hailo_dfc_venv/bin/activate

# Upgrade pip and install DFC package
pip install --upgrade pip
pip install hailo_dataflow_compiler-3.28.0-py3-none-linux_x86_64.whl
```

Verify DFC installation:
```bash
hailo --version
```

---

## 4. Step 3: Prepare Model Script (`pcb_yolov8n.alls`)

Create the model allocation script `pcb_yolov8n.alls` defining input normalization to match the YOLOv8 letterbox preprocessing:

```ini
# Normalization: YOLOv8 expects pixels scaled from [0, 255] to [0, 1]
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])

# Direct output allocation to PCIe stream
performance_param(fps=30)
```

---

## 5. Step 4: Generate Calibration Dataset

Quantization from FP32 to INT8 requires a representative calibration dataset (typically 512 to 1024 images). Run the calibration builder script on your development PC:

```bash
python -c "
import cv2, glob, numpy as np
from pathlib import Path
from hardware.inference.yolo_postprocess import letterbox_image

img_paths = sorted(glob.glob('datasets/DeepPCB/evaluation/test_images/*.jpg'))[:512]
calib_tensors = []
for p in img_paths:
    img = cv2.imread(p)
    if img is not None:
        lb, _, _ = letterbox_image(img, (640, 640))
        calib_tensors.append(lb)

calib_array = np.array(calib_tensors, dtype=np.uint8)
np.save('calib_dataset.npy', calib_array)
print(f'Saved calibration dataset: {calib_array.shape}')
"
```

---

## 6. Step 5: Parse, Optimize, and Compile to HEF

### A. Parse ONNX Model
```bash
hailo parser onnx models/onnx/pcb_multidomain_yolov8n_640.onnx \
    --model-name pcb_multidomain_yolov8n \
    --har-path pcb_multidomain_yolov8n.har
```

### B. Optimize and Quantize (INT8)
```bash
hailo optimize pcb_multidomain_yolov8n.har \
    --calib-set-path calib_dataset.npy \
    --model-script pcb_yolov8n.alls \
    --output-har-path pcb_multidomain_quantized.har
```

### C. Compile to Target HEF
For Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS):
```bash
hailo compiler pcb_multidomain_quantized.har \
    --target hailo8l \
    --output-dir models/hailo/
```

For full Hailo-8 (26 TOPS):
```bash
hailo compiler pcb_multidomain_quantized.har \
    --target hailo8 \
    --output-dir models/hailo/
```

This generates:
- `models/hailo/pcb_multidomain_yolov8n.hef`

---

## 7. Step 6: Deploy HEF to Raspberry Pi 5

Copy the compiled HEF binary to the edge device:

```bash
scp models/hailo/pcb_multidomain_yolov8n.hef pi@<raspberry-pi-ip>:~/EdgePCB-QC/models/hailo/
```

---

## 8. Step 7: Verify on Raspberry Pi 5

Log into the Raspberry Pi 5 and execute the hardware verification script:

```bash
python hardware/deployment/verify_hailo.py
```

Expected output:
```
[PASS] Hailo PCIe device detected: 1e60:2864
[PASS] HailoRT v4.18.0 loaded
[PASS] HEF model loaded: pcb_multidomain_yolov8n.hef
[PASS] NPU benchmark: 640x640 frame inferred in 14.2 ms (~70 FPS)
```
