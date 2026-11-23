# Hailo-8 / Hailo-8L Model Deployment & Runtime Guide
**EdgePCB-QC | Bharat AI-SoC Challenge 2026–27 | PS5**

This document covers the end-to-end deployment of the compiled **Hailo Executable Format (HEF)** vision model on the Raspberry Pi AI HAT+.

---

## 1. Model Lifecycle Overview

```
Frozen PyTorch Baseline (best.pt)
  SHA-256: 66d8d225fe...
         |
         |  python hardware/deployment/export_onnx.py
         v
Standard ONNX Model (pcb_multidomain_yolov8n_640.onnx)
  Static [1, 3, 640, 640], Opset 12
         |
         |  Hailo Dataflow Compiler (DFC v3.28+ in Ubuntu 22.04 / WSL2)
         |  Quantization (INT8) using representative PCB calibration dataset
         v
Compiled Hailo Executable Format (pcb_multidomain_yolov8n.hef)
         |
         |  scp to Raspberry Pi 5
         v
Target Edge Device: Raspberry Pi 5 + Raspberry Pi AI HAT+
  PCIe Gen3 x1 link -> HailoRT v4.18+ / v4.19+
         |
         |  hardware/inference/hailo_backend.py
         v
Hardware-Accelerated Inference (< 15 ms latency per 640x640 frame)
```

---

## 2. Directory Placement on Raspberry Pi 5

Place the compiled HEF binary in the designated model directory:

```bash
mkdir -p ~/EdgePCB-QC/models/hailo/
cp pcb_multidomain_yolov8n.hef ~/EdgePCB-QC/models/hailo/
```

Verify permissions and file presence:
```bash
ls -lh ~/EdgePCB-QC/models/hailo/pcb_multidomain_yolov8n.hef
```

---

## 3. Running Hardware Verification

Execute the on-device verification script:

```bash
python hardware/deployment/verify_hailo.py
```

What the script tests:
1. **Python Bindings:** Confirms `import hailo_platform` functions.
2. **PCIe Device Scan:** Scans PCIe bus for Hailo-8 / Hailo-8L (`hailo_platform.Device.scan()`).
3. **HEF Configuration:** Configures VStreams (`HailoStreamInterface.PCIe`).
4. **Latency Benchmark:** Runs 50 benchmark iterations measuring mean, P95, and P99 latency.

---

## 4. Backend Resolution Strategy (`auto` vs `hailo` vs `cpu`)

EdgePCB-QC features an adaptive inference manager (`hardware/inference/inference_manager.py`):

- **`auto` (Default):**
  - Probes for Hailo device presence and valid HEF file.
  - If both exist, Hailo NPU acceleration is automatically selected.
  - If hardware is absent (e.g. running on laptop development workstation), the system seamlessly runs the approved YOLOv8n CPU baseline without crashing.
- **`hailo`:**
  - Forces Hailo NPU acceleration. If hardware is absent, logs diagnostic warning and transparently triggers CPU fallback with a clear disclaimer in the UI.
- **`cpu`:**
  - Forces PyTorch CPU baseline execution (ideal for verifying identical detection outputs).

> [!IMPORTANT]
> The Streamlit dashboard **never** labels CPU inference as Hailo inference. The active backend is explicitly displayed with latency measurements on every screen.
