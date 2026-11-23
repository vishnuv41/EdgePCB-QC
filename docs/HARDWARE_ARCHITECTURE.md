# EdgePCB-QC — Edge Hardware Architecture Specification
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**

---

## 1. Executive Summary

EdgePCB-QC is an offline-first industrial inspection system engineered for bare-PCB defect detection and quality triage. The physical deployment splits functionality into two physical computing nodes connected over an isolated local network (Ethernet / Private Wi-Fi):

1. **Edge Processing Node (Raspberry Pi 5 + AI HAT+):** Responsible for high-speed sensor capture, hard real-time neural vision acceleration on Hailo NPU, deterministic IPC quality gating, ExecuTorch Llama advisory inference on CPU, and local SQLite data logging.
2. **Operator Display & Control Workstation (Laptop):** Web browser interface connecting to the Raspberry Pi over the local subnet (`http://<pi-ip>:8501`), rendering live video feed, inspection bounding boxes, ROI defect magnification, and operator review controls.

```
   +--------------------------------------------------------------------------+
   |                  EDGE PROCESSING NODE (Raspberry Pi 5)                    |
   |                                                                          |
   |   +-----------------------+           +------------------------------+   |
   |   |   Camera Module 3     |           |  Raspberry Pi AI HAT+        |   |
   |   |   (CSI / IMX708 AF)   |           |  Hailo-8 / Hailo-8L NPU      |   |
   |   +-----------+-----------+           |  (PCIe Gen 3 x1 @ 8.2 GFLOPs)|   |
   |               |                       +--------------+---------------+   |
   |               | BGR Frames                           |                   |
   |               v                                      | HEF VStreams      |
   |   +-----------+-----------+                          v                   |
   |   |  OpenCV Preprocessing | ---- 640x640 ----> [ HailoRT NPU ]           |
   |   +-----------------------+                          |                   |
   |                                                      | Raw Tensors       |
   |                                                      v                   |
   |                                            [ YOLOv8n Decoder & NMS ]     |
   |                                                      |                   |
   |                                                      v                   |
   |                                           [ Deterministic Decision ]     |
   |                                           [ Engine (PASS/INSPECT/  ]     |
   |                                           [ REJECT)                ]     |
   |                                                      |                   |
   |         +--------------------------------------------+                   |
   |         |                                            |                   |
   |         v                                            v                   |
   |   [ SQLite DB ]                       [ ExecuTorch Llama 3.2 1B ]        |
   |   (inspections.db)                    (Cortex-A76 CPU @ 2.4 GHz)         |
   |         |                                            |                   |
   |         +---------------------+----------------------+                   |
   |                               |                                          |
   |                               v                                          |
   |                 [ Streamlit Web Server (8501) ]                          |
   +-------------------------------+------------------------------------------+
                                   |
                   Isolated Local Network (Ethernet / LAN)
                                   |
                                   v
   +--------------------------------------------------------------------------+
   |            OPERATOR CONTROL WORKSTATION (Laptop Browser)                 |
   |                                                                          |
   |   • Live Camera Stream View          • Defect ROI Magnifier (1:1 Pixels) |
   |   • Real-Time Bounding Boxes         • Human Review & Operator Overrides |
   |   • Instant Quality Verdicts         • SPC & Pareto Defect Analytics     |
   |   • Llama 3.2 1B Operator Advice     • CSV / JSON Traceability Export    |
   +--------------------------------------------------------------------------+
```

---

## 2. Hard Real-Time vs Soft Real-Time Pipeline Decoupling

A core architectural principle of EdgePCB-QC is the strict decoupling of the safety-critical vision inspection from generative AI reasoning:

### Pipeline A: Hard Real-Time Vision & Deterministic Verdict
- **Hardware:** Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L NPU) over PCIe Gen3 x1.
- **Latency Target:** $< 15\text{ ms}$ per 640×640 tile; $< 180\text{ ms}$ whole-board.
- **Function:** Normalizes pixels, executes INT8 HEF forward pass, runs multiclass NMS, and evaluates deterministic quality rules (`RULE_001` through `RULE_007`).
- **Safety Invariant:** Generative AI / LLMs **never** decide PASS, INSPECT, or REJECT. Verdicts are 100% deterministic and auditable.

### Pipeline B: Soft Real-Time Advisory Reasoning
- **Hardware:** Raspberry Pi 5 Application Processor (Quad-core ARM Cortex-A76 @ 2.4 GHz).
- **Runtime:** Meta ExecuTorch with `llama3_2_1b_fp16.pte` (or INT4 8da4w).
- **Latency:** $\sim 800\text{ ms}$ to $1500\text{ ms}$ (non-blocking).
- **Function:** Ingests structured JSON findings (defect class, severity, coordinates, triggered rule) and generates actionable rework guidance adhering to IPC-A-600 standards.
- **Fallback:** Instant rule-based fallback ($< 0.1\text{ ms}$) activates automatically if memory headroom is $< 1200\text{ MB}$ or ExecuTorch runtime is uninitialized.

---

## 3. Physical Hardware Bill of Materials (BOM)

| Component | Specification | Function |
|---|---|---|
| **Host SBC** | Raspberry Pi 5 Model B (8GB RAM) | Edge processing host running Linux, Streamlit, SQLite, and ExecuTorch. |
| **AI Accelerator** | Raspberry Pi AI HAT+ (Hailo-8L 13 TOPS or Hailo-8 26 TOPS) | PCIe Gen3 NPU for YOLOv8n vision model inference. |
| **Camera Module** | Raspberry Pi Camera Module 3 (Sony IMX708, 12MP, autofocus) | Optical capture of bare PCBs via dedicated 22-pin CSI ribbon. |
| **CSI Cable** | 22-pin to 22-pin 0.5mm pitch FFC cable | Connects Camera Module 3 to Raspberry Pi 5 CAM0/CAM1 port. |
| **Power Supply** | Official Raspberry Pi 27W USB-C Power Supply (5.1V / 5A PD) | Guarantees sufficient current for Pi 5 + AI HAT+ + Camera during peak inference. |
| **Cooling** | Raspberry Pi 5 Active Cooler (aluminum heatsink + PWM blower) | Maintains SoC temperature $< 65^\circ\text{C}$ under sustained continuous load. |
| **Storage** | 64GB+ SanDisk Extreme Pro MicroSD or NVMe SSD | Stores OS, SQLite audit logs, model weights, and inspection captures. |
| **Networking** | Standard Cat6 RJ45 Ethernet Cable | Low-latency, zero-configuration connection between Pi 5 and Laptop. |
| **Workstation** | Any laptop with web browser (Chrome, Edge, Firefox) | Operator interface accessing `http://<pi-ip>:8501`. |

---

## 4. Software Stack Layers

```
Layer 5: Presentation  -> Streamlit Industrial Workstation (0.0.0.0:8501)
Layer 4: Intelligence   -> Deterministic DecisionEngine + ExecuTorch Llama 3.2 1B
Layer 3: Vision Backend -> InferenceManager (HailoRT VStreams / PyTorch CPU)
Layer 2: Sensor Capture -> CameraManager (Picamera2 IMX708 / V4L2 USB / Mock)
Layer 1: Edge OS        -> Raspberry Pi OS 64-bit (Debian 12 Bookworm, Linux 6.6)
```
