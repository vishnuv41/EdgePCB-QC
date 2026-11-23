# Hailo-8 / Hailo-8L HEF Deployment Directory

This directory is the designated location for compiled **Hailo Executable Format (`.hef`)** neural network binaries for the Raspberry Pi AI HAT+.

## Expected HEF File Name
- Target file: `pcb_multidomain_yolov8n_640.hef`
- Target path: `models/hailo/pcb_multidomain_yolov8n_640.hef`

## Automatic Hardware Detection & Fallback
The `InferenceManager` subsystem automatically inspects this directory:
1. **If `.hef` is present** and Hailo-8 hardware/driver is detected:
   - High-performance NPU acceleration is automatically enabled (projected ~8.4 ms latency).
2. **If `.hef` is NOT present** (current default state):
   - The system automatically and safely falls back to the Raspberry Pi 5 ARM Cortex-A76 CPU baseline via ONNX Runtime / PyTorch (`models/onnx/pcb_multidomain_yolov8n_640.onnx`).
   - The UI displays: `CPU Pipeline (YOLOv8n)`.

## How to compile the HEF file
1. Export YOLOv8n to ONNX with 640x640 shape:
   ```bash
   python hardware/deployment/export_onnx.py
   ```
2. On your x86_64 host machine with Hailo Dataflow Compiler (DFC):
   ```bash
   hailo parser onnx models/onnx/pcb_multidomain_yolov8n_640.onnx --start-node-names images --end-node-names /model.22/Concat_3 /model.22/Concat_4 /model.22/Concat_5
   hailo optimize pcb_multidomain_yolov8n_640.har --calib-set-path datasets/calibration
   hailo compiler pcb_multidomain_yolov8n_640_quantized.har --hw-arch hailo8
   ```
3. Copy the resulting `pcb_multidomain_yolov8n_640.hef` into this folder:
   ```bash
   cp pcb_multidomain_yolov8n_640.hef models/hailo/
   ```
4. Restart EdgePCB-QC — it will immediately detect and use the Hailo NPU!
