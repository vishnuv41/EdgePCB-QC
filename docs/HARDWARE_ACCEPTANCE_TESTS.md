# EdgePCB-QC Hardware Acceptance Test Plan
**Bharat AI-SoC Challenge 2026–27 | PS5**

This document details the formal criteria and automated test suites used to verify that EdgePCB-QC is hardware-ready for Raspberry Pi 5 + Hailo deployment.

---

## 1. Automated Software Regression Tests (Run Pre-Hardware)

Run all 147 unit and integration tests on the development workstation:

```bash
pytest tests/ -v
```

### Breakdown of Acceptance Suites:
- `tests/test_camera_manager.py`: Tests camera abstraction, discovery, streaming worker thread, bounded latest-frame buffer, and mock simulator. (5 tests)
- `tests/test_inference_manager.py`: Tests backend resolution, letterbox geometry, pure NumPy NMS, and CPU fallback. (5 tests)
- `tests/test_hardware_status.py`: Tests truthful hardware auditing, telemetry extraction, and network URL discovery. (3 tests)
- `tests/test_offline_mode.py`: Tests 100% offline air-gap compliance and zero external CDN dependencies. (4 tests)
- `tests/test_streamlit_hardware_workflow.py`: Tests the complete end-to-end simulated hardware flow from camera capture to database persistence. (1 test)
- Core vision, decision, advisory, and database regression tests. (129 tests)

**Total Test Count:** **147 / 147 passing (100%)**.

---

## 2. Physical Hardware Acceptance Checklist (To Run When Hardware Arrives)

When physical hardware is assembled, verify each line item:

| Step | Acceptance Item | Verification Procedure | Expected Outcome | Pass/Fail |
|---|---|---|---|---|
| **H-01** | **Active Cooler** | Power on Pi 5. Verify PWM fan spins up during boot. | Fan spins up, SoC temp stays $< 55^\circ\text{C}$ idle. | [ ] |
| **H-02** | **CSI Camera Link** | Run `libcamera-hello --list-cameras`. | IMX708 sensor detected on `CAM0` or `CAM1`. | [ ] |
| **H-03** | **Live Camera Preview** | Open `📹 Live Camera Inspection` in Streamlit. Select CSI-0 and click `Start Camera`. | Smooth live video stream visible in browser at $\ge 15\text{ FPS}$. | [ ] |
| **H-04** | **Camera Autofocus** | Place PCB under lens at $15\text{ cm}$. Verify focus adjusts cleanly. | Sharp, focused image of copper traces and via holes. | [ ] |
| **H-05** | **Hailo PCIe Link** | Run `lspci -d 1e60:`. | PCIe device `1e60:2864` detected with Gen3 link active. | [ ] |
| **H-06** | **HailoRT Runtime** | Run `python -c "import hailo_platform; print(hailo_platform.Device.scan())"`. | List of connected Hailo devices returned without exception. | [ ] |
| **H-07** | **HEF Loading** | Run `python hardware/deployment/verify_hailo.py`. | HEF binary loaded into NPU memory; 50 iterations run. | [ ] |
| **H-08** | **Hailo NPU Latency** | Verify measured latency in `verify_hailo.py`. | Mean latency $< 16\text{ ms}$ per 640×640 frame ($> 60\text{ FPS}$). | [ ] |
| **H-09** | **Full Inspection Flow**| Click `📸 Capture PCB & Run Inspection` in Streamlit. | Bounding boxes rendered, verdict calculated, advisory produced. | [ ] |
| **H-10** | **Database Audit** | Inspect `📋 Inspection Log & Audit` tab. | Captured board record logged with SHA-256 and `real` tag. | [ ] |
| **H-11** | **Laptop Remote Access**| Access `http://<pi-ip>:8501` from disconnected laptop over Ethernet. | Complete dashboard loads and operates without internet connection. | [ ] |
