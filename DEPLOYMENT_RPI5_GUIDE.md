# EdgePCB-QC — Raspberry Pi 5 Hardware Deployment Guide
## Offline Edge AI Printed Circuit Board Quality-Control Workstation
**Target SBC:** Raspberry Pi 5 (8GB LPDDR4X)  
**Target OS:** Raspberry Pi OS 64-bit (Debian 12 Bookworm) / Ubuntu 24.04 LTS  
**Target Hardware:** Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L NPU) & Camera Module 3 (Sony IMX708)  
**Deployment Archive:** `edgepcb_rpi5_deployment.zip` (17.88 MB)  
**Archive SHA-256:** `57a31e56b5f6e7273b6ce92ddea095f95c8bcbbe84c5f780511479b59f5eae93`

---

## 1. Architectural Division: Models & Execution Backends

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EDGEPCB-QC SYSTEM ARCHITECTURE                     │
├──────────────────────────────────────┬──────────────────────────────────────┤
│ PIPELINE A: VISION DEFECT LOCALIZATION │ PIPELINE B: ENGINEERING ADVISORY     │
│ (Hard Real-Time Defect Detection)     │ (Soft Real-Time Reasoning & Rework)  │
├──────────────────────────────────────┼──────────────────────────────────────┤
│ • Task: 6-class PCB defect bounding   │ • Task: IPC-A-610 root-cause triage, │
│   boxes & confidence scoring.         │   troubleshooting, rework guidance.  │
│                                      │                                      │
│ • Hardware: Raspberry Pi AI HAT+     │ • Hardware: Raspberry Pi 5 Host CPU  │
│   (Hailo-8 26 TOPS / Hailo-8L 13 TOPS)│   (ARM Cortex-A76, 4 cores @ 2.4GHz) │
│                                      │                                      │
│ • Format: Hailo Executable (.hef)    │ • Format: ExecuTorch (.pte) or       │
│   models/hailo/pcb_multidomain_      │   Deterministic Fallback Engine      │
│   yolov8n_640.hef                    │   models/llama/llama3_2_1b_fp16.pte  │
│                                      │                                      │
│ • CPU Fallback: ONNX Runtime /       │ • Fallback: Instant Deterministic    │
│   PyTorch CPU baseline (included).   │   Expert Rules Engine (<1 ms).       │
│                                      │                                      │
│ • Is YOLOv8n converted to .pte?      │ • Is Llama converted to .pte?        │
│   NO. YOLO runs on Hailo (.hef) or   │   YES. Llama runs via ExecuTorch     │
│   CPU (.onnx / .pt).                 │   (.pte) on the ARM CPU via XNNPACK. │
└──────────────────────────────────────┴──────────────────────────────────────┘
```

> [!IMPORTANT]
> **Files That Must Be Supplied Separately:**
> 1. **`models/hailo/pcb_multidomain_yolov8n_640.hef`:** Since you will supply the compiled `.hef` file later, the system boots automatically in **CPU Baseline Mode** using `models/onnx/pcb_multidomain_yolov8n_640.onnx`. As soon as you copy your compiled `.hef` file into `models/hailo/`, the system automatically detects it and enables hardware NPU acceleration with zero code changes!
> 2. **`models/llama/llama3_2_1b_fp16.pte`:** At 2.3 GB, this binary is not packaged in the 17.87 MB distribution ZIP. If omitted, the system seamlessly uses the instant deterministic IPC-A-610 fallback engine (<1 ms).

---

## 2. Hardware Bill of Materials (BOM)

| Item | Component | Specification | Function |
|---|---|---|---|
| **1** | Single-Board Computer | **Raspberry Pi 5 (8GB)** | Host processor, SQLite database, FastAPI web server. |
| **2** | AI Accelerator | **Raspberry Pi AI HAT+** | Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) PCIe Gen3 NPU. |
| **3** | Optical Sensor | **Raspberry Pi Camera Module 3** | Sony IMX708, 12MP, autofocus, macro diffuser ring. |
| **4** | Power Supply | **Official 27W USB-C PSU** | 5.1V / 5.0A with Power Delivery (required for Pi 5 + HAT). |
| **5** | Storage | **MicroSD / NVMe SSD (32GB+)** | Fast A2/V30 MicroSD card or M.2 NVMe SSD. |
| **6** | Networking | **Ethernet Cable / Wi-Fi** | Local network communication between Pi 5 and operator laptop. |

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Copy and Extract the Deployment Package
Transfer `edgepcb_rpi5_deployment.zip` to the Raspberry Pi 5 (via SCP, USB flash drive, or SFTP):

```bash
# Verify checksum before extraction:
sha256sum -c edgepcb_rpi5_deployment.zip.sha256

# Extract:
unzip edgepcb_rpi5_deployment.zip -d ~/EdgePCB-QC
cd ~/EdgePCB-QC
```

### Step 2: One-Click System Provisioning
Run the automated installation script with root privileges:

```bash
sudo ./setup_pi.sh
```

This script automatically:
1. Updates package repositories.
2. Installs required system libraries (`libgl1`, `libglib2.0-0`, `v4l-utils`, `pciutils`).
3. Installs Hailo drivers and runtime packages (`hailo-all`, `python3-hailort`).
4. Configures PCIe Gen 3 in `/boot/firmware/config.txt` (`dtparam=pciex1_gen=3`).
5. Creates the Python virtual environment with `--system-site-packages` so `picamera2` is accessible.
6. Installs all required Python dependencies from `hardware/requirements_pi.txt`.

### Step 3: Reboot the Raspberry Pi
Reboot once to apply the PCIe Gen 3 device tree configuration:

```bash
sudo reboot
```

### Step 4: Verify Hardware Connectivity
After reboot, run the preflight verification script:

```bash
cd ~/EdgePCB-QC
bash hardware/deployment/verify_hardware.sh
```

Expected diagnostic checks:
- Camera: Sony IMX708 detected on CSI port.
- Hailo NPU: PCIe link up at Gen 3 speed.
- Database: SQLite `inspections.db` initialized and writable.
- Vision Model: `pcb_multidomain_yolov8n_640.onnx` verified.

### Step 5: Launch the EdgePCB-QC Workstation
Start the production server:

```bash
./start_edgepcb.sh
```

The terminal will display the active IP address:
```
==========================================================================
EDGEPCB-QC WORKSTATION READY (FastAPI + High-Performance Web Dashboard)
Port: 8000 (0.0.0.0 - Bound to all local network interfaces)

Open this URL in your Laptop browser (connected to same Wi-Fi / Ethernet):
>>> http://192.168.1.45:8000 <<<
==========================================================================
```

### Step 6: Connect from Your Laptop Browser
Open your laptop browser and navigate to:
```
http://<RASPBERRY_PI_IP>:8000
```
*(Example: `http://192.168.1.45:8000`)*

The workstation loads instantly with **100% offline capability** (no internet or cloud dependencies).

---

## 4. Workstation Features in the Deployed Web App

1. **📹 Live Camera Station:**
   - Real-time video streaming from Camera Module 3.
   - Resolution toggle (1280x720, 1920x1080, 640x480).
   - "Capture & Inspect" button: Only activates camera when triggered.
   - **Curated Test Boards Gallery:** 5 pre-loaded bare PCB boards (Clean PASS, Open Circuit, Short Circuit, Mouse Bite, Multi-Defect) for instant demonstration even without a camera connected!

2. **🔍 Inspection Workspace (ImageFit & Scroll-To-Zoom):**
   - **Hero PCB Viewport:** Displays the PCB with `object-fit: contain` (ImageFit).
   - **Scroll-to-Zoom:** Scroll mouse wheel to zoom in/out smoothly up to 600%.
   - **Drag-to-Pan:** Click and drag to pan across the magnified PCB.
   - **Fit to Screen Button:** Resets zoom and pan instantly.
   - **Original Raw vs Annotated Overlay:** Toggle between defect bounding boxes and pristine raw PCB image.
   - **Defect Magnifier:** High-detail crop viewer of the selected defect.
   - **Pipeline Execution Stepper:** Real measured timings for Capture, Quality Gate, YOLO Inference, Decision Engine, and Advisory.
   - **Verdict Card:** Large PASS (green), INSPECT (amber), or REJECT (red) badge with triggered rule ID.
   - **Electrical Verification Alert:** Automatic warning if short/open circuits are detected.
   - **Defect Manifest Table:** Structured defect ID, class, severity pill, confidence, and bounding box coordinates.
   - **Engineering Advisory Card:** Structured IPC-A-610 root-cause breakdown and rework instructions.
   - **Supervisor Override:** Authorize and log manual disposition overrides with cryptographic audit hashing.

3. **🔬 Human Review & Verification:**
   - Inspector verification of detections: confirm true defects or log false calls.
   - Live false-call rate tracking and defect class breakdown.

4. **📋 Inspection Log & Audit:**
   - Full historical inspection records filterable by verdict and source.
   - One-click **Export Audit CSV** button.

5. **📊 Pareto & Defect Analytics:**
   - Real-time first-pass yield and verdict distribution donut.
   - Defect Pareto breakdown sorted by frequency with cumulative 80/20 line.

6. **⚙️ Edge Hardware & Telemetry:**
   - Real-time CPU %, RAM, and core thermal telemetry.
   - Air-gapped offline security checklist.

---

## 5. How to Add Your Compiled Hailo HEF File Later

When your `.hef` file is compiled:
1. Copy the file into the `models/hailo/` directory:
   ```bash
   cp pcb_multidomain_yolov8n_640.hef ~/EdgePCB-QC/models/hailo/
   ```
2. Restart EdgePCB-QC:
   ```bash
   sudo systemctl restart edgepcb
   # Or Ctrl+C and re-run ./start_edgepcb.sh
   ```
3. The dashboard header pill will automatically change from:
   `CPU Pipeline` ➔ `Hailo NPU Pipeline (Accelerated)`

---

## 6. Autostart on Boot (Systemd Service)

To run EdgePCB-QC automatically whenever the Raspberry Pi powers on:

```bash
sudo cp hardware/deployment/edgepcb.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable edgepcb.service
sudo systemctl start edgepcb.service
```

---

## 7. Streamlit Alternate Application

The original Streamlit application remains **100% preserved** in `dashboard/app.py` as an alternate interface. To launch Streamlit on port 8501:

```bash
streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```
