# EdgePCB-QC Hardware Integration Subsystem
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**

This directory contains the edge integration layer for deploying EdgePCB-QC on a **Raspberry Pi 5** paired with a **Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L)** and a **Raspberry Pi Camera Module 3** or **USB inspection camera**.

---

## Directory Architecture

```
hardware/
├── camera/
│   ├── camera_manager.py     # Unified camera controller, discovery, bounded frame queue, thread-safe streaming
│   ├── csi_camera.py         # Picamera2 interface for Raspberry Pi Camera Module 3 (IMX708 AF)
│   └── usb_camera.py         # OpenCV / V4L2 interface for benchtop USB webcams
├── inference/
│   ├── inference_manager.py  # Coordinates 'auto', 'hailo', and 'cpu' vision backends
│   ├── cpu_backend.py        # PyTorch YOLOv8n frozen baseline CPU runner
│   ├── hailo_backend.py      # HailoRT VStreams inference engine for Hailo-8/8L AI HAT+
│   └── yolo_postprocess.py   # Pure NumPy letterbox padding, unletterbox scaling, NMS & defect mapping
├── llm/
│   ├── executorch_manager.py # ExecuTorch runtime manager on Cortex-A76 CPU with memory preflight
│   └── advisory.py           # Structured IPC guidance builder
├── network/
│   └── network_status.py     # Local IP discovery & connection URL generator for laptop browser
├── system/
│   ├── hardware_detection.py # Truthful hardware auditing (Pi model, Hailo PCIe device, HEF validation)
│   └── system_monitor.py     # Real-time CPU, RAM, thermal, and storage consumption telemetry
├── deployment/
│   ├── setup_pi.sh           # Raspberry Pi OS setup script
│   ├── install_dependencies.sh # Air-gapped dependency installer
│   ├── verify_hardware.sh    # Preflight hardware verification script
│   ├── start_edgepcb.sh      # Production startup script (Streamlit on 0.0.0.0:8501)
│   ├── stop_edgepcb.sh       # Clean shutdown script
│   ├── export_onnx.py        # Export frozen PyTorch best.pt to Hailo-compatible ONNX
│   ├── compile_hailo.md      # DFC compilation guide (Ubuntu 22.04 / WSL2)
│   └── verify_hailo.py       # On-device Hailo benchmark and verification script
├── requirements_pi.txt       # Python package list for Raspberry Pi OS
└── requirements_hailo.txt    # Hailo package guidance
```
