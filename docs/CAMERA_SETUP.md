# EdgePCB-QC Camera Setup & Calibration Guide
**Bharat AI-SoC Challenge 2026–27 | PS5**

This document covers camera selection, physical mounting, optical lighting, and software configuration for capturing bare PCBs on Raspberry Pi 5.

---

## 1. Supported Cameras

EdgePCB-QC natively supports two hardware camera classes:

### Option A: Raspberry Pi Camera Module 3 (Recommended)
- **Sensor:** Sony IMX708 (11.9 Megapixel, Back-Illuminated CMOS with HDR)
- **Focusing:** Voice-coil motor powered autofocus (AF) with macro capability down to $\sim 5\text{ cm}$.
- **Interface:** Dedicated 22-pin CSI-2 2-lane connector.
- **Python Driver:** `picamera2` (official Raspberry Pi libcamera binding).
- **Advantages:** Low latency, hardware ISP Bayer demosaicing, auto-exposure, and zero USB bus saturation.

### Option B: USB UVC / Industrial Webcam
- **Interface:** USB 2.0 / 3.0 via standard V4L2 (`cv2.VideoCapture`).
- **Driver:** Standard OpenCV VideoCapture.
- **Advantages:** Plug-and-play compatibility across development workstations and Raspberry Pi without ribbon cables.

### Diagnostic Mode: Mock PCB Simulator
- For testing when no physical camera is connected. Generates simulated bare PCB patterns with synthetic traces, annular rings, and timestamps to verify the UI and detection pipeline offline.

---

## 2. Physical Mounting & Optical Geometry

Accurate bare-PCB defect detection requires controlled optical illumination and perpendicular alignment:

1. **Working Distance:** Mount the camera $120\text{ mm}$ to $180\text{ mm}$ directly above the PCB inspection bed.
2. **Perpendicularity:** Ensure the optical lens axis is at $90^\circ \pm 1^\circ$ to the PCB surface to eliminate perspective distortion and trapezoidal trace skewing.
3. **Lighting Geometry:**
   - **Avoid Direct Flash / Point Sources:** Bare copper and solder mask cause high specular glare, which obscures micro-cracks and creates false pinhole detections.
   - **Use Diffused Ring Lighting or Coaxial Lighting:** Diffused white LEDs ($5000\text{K}$ neutral white) positioned in a ring around the lens provide uniform shadow-free illumination across traces, vias, and ground planes.

---

## 3. Camera Controls in Streamlit Workstation

In the **`📹 Live Camera Inspection`** tab:
1. **Camera Selector:** Switch between `Raspberry Pi Camera Module 3 (CSI-0)`, `USB Video Device`, or `Mock PCB Simulator`.
2. **Resolution Options:**
   - `640x640 (Model Native)`: Zero-letterbox 1:1 inference capture.
   - `1280x720 (HD)`: Standard benchtop inspection resolution.
   - `1920x1080 (Full HD)`: High-fidelity capture for fine pitch ($< 0.15\text{ mm}$) surface tracing.
3. **Streaming Controls:**
   - `▶️ Start Camera`: Launches background capture thread with a single-frame bounded buffer (eliminates lag).
   - `⏹️ Stop Camera`: Gracefully releases hardware video device.
   - `📸 Capture PCB & Run Inspection`: Snaps high-fidelity still, triggers vision inference, evaluates deterministic rules, and updates SQLite.
   - `Continuous Live Inspection`: Auto-polls inspection every frame.
