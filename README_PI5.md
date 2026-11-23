# EdgePCB-QC: Raspberry Pi 5 + Hailo-8 Deployment Release

**Bharat AI-SoC Challenge 2026–27 | Problem Statement 5**
*Real-Time Industrial Defect Detection and Quality-Control Assistant*

Target Hardware:
- **Raspberry Pi 5** (8 GB RAM, 64-bit OS Bookworm)
- **Hailo-8 AI HAT+** (26 TOPS NPU)
- **Raspberry Pi Camera Module v3** (IMX708) or standard USB UVC Camera
- **100% Offline / Air-Gapped Operation**

---

## 1. Quick Start Installation

```bash
# 1. Unzip release package
unzip EdgePCB-QC_RPI5_HAILO8_FINAL_DEPLOYMENT.zip -d EdgePCB-QC
cd EdgePCB-QC

# 2. Run automated installation script
chmod +x *.sh
bash INSTALL_PI5.sh

# 3. If PCIe Gen 3 was just enabled, reboot the Pi
sudo reboot

# 4. Start EdgePCB-QC Station
./START_EDGEPCB_QC.sh
```

Access the operator workstation UI in any browser:
- Local on Pi: `http://localhost:8000`
- Over LAN: `http://<PI_IP_ADDRESS>:8000`

---

## 2. Dual-Pipeline Architecture

1. **Pipeline A (Real-Time Quality Gating)**
   - **Accelerator:** Hailo-8 M.2 AI HAT+ (26 TOPS INT8)
   - **Model:** `pcb_yolov8n_raw_heads.hef` (DFL decoded on host CPU)
   - **CPU Fallback:** `pcb_multidomain_yolov8n_640.onnx` via ARM64 ONNX Runtime
   - **Decision Engine:** Deterministic IPC-A-610 decision rules (<0.1 ms latency)

2. **Pipeline B (Edge SLM Advisory Engine)**
   - **Model:** Meta Llama 3.2 1B Instruct
   - **Runtime:** ExecuTorch with XNNPACK on ARM64 CPU
   - **Quantization:** `8da4w` (8-bit Dynamic Activation / 4-bit Weight)
   - **Artifact:** `llama3_2_1b_xnnpack_8da4w.pte` + `tokenizer.model`
   - **Safety Invariance:** Llama never overrides the deterministic PASS/FAIL verdict

---

## 3. Operational Scripts

| Script | Purpose |
|---|---|
| `INSTALL_PI5.sh` | Automated system packages, Python venv, Hailo stack, model checksum validation |
| `START_EDGEPCB_QC.sh` | Launches FastAPI backend and workstation UI (foreground or `--daemon`) |
| `STOP_EDGEPCB_QC.sh` | Gracefully halts all running processes |
| `CHECK_HARDWARE.sh` | Audits PCIe node, HailoRT driver, thermal telemetry, and camera interface |
| `VERIFY_DEPLOYMENT.sh` | Executes automated smoke tests for CPU Fallback (Mode A) and Hailo (Mode B) |
| `UNINSTALL_PI5.sh` | Cleanly uninstalls services and virtual environments |

---

## 4. Hardware Verification & PCIe Setup

For the Hailo-8 AI HAT+ to be detected on Raspberry Pi 5, ensure `/boot/firmware/config.txt` contains:
```ini
dtparam=pciex1
dtparam=pciex1_gen=3
```
Run `bash deployment/hailo/setup_hailo_hat.sh` to automatically configure this overlay.
