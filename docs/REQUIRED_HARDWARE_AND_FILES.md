# Required Hardware & File Transfer Manifest
**EdgePCB-QC | Bharat AI-SoC Challenge 2026–27 | PS5**

This document details all physical components needed and the exact manifest of files that must be transferred from the development computer to the Raspberry Pi 5.

---

## 1. Physical Hardware Requirements Checklist

### Mandatory Core Hardware:
1. **Raspberry Pi 5 Model B (8GB RAM)**: Application host computer.
2. **Raspberry Pi AI HAT+**: Contains Hailo-8 (26 TOPS) or Hailo-8L (13 TOPS) NPU accelerator with 16-pin PCIe ribbon cable and standoffs.
3. **Raspberry Pi Camera Module 3**: Sony IMX708 sensor with powered autofocus and 22-pin to 22-pin FFC ribbon cable.
4. **Official Raspberry Pi 27W USB-C Power Supply**: 5.1V / 5.0A PD power adapter (crucial for peak PCIe + camera current stability).
5. **Raspberry Pi 5 Active Cooler**: Aluminum heatsink with PWM temperature-controlled fan.
6. **MicroSD Card (64GB+)**: High-speed Class 10 / A2 rated (e.g., SanDisk Extreme Pro).
7. **Cat6 Ethernet Cable**: Connects Raspberry Pi 5 directly to laptop.

### Inspection Fixture & Bench Accessories:
8. **Diffused White LED Ring Light (5000K)**: Provides uniform, shadow-free illumination without specular glare.
9. **Rigid Camera Stand / Vertical Mount**: Holds camera securely at $120\text{ mm}$ to $180\text{ mm}$ above the PCB fixture.
10. **Sample Bare PCBs**: Clean boards and known defective boards for acceptance verification.

---

## 2. File Transfer Manifest: What to Copy to Raspberry Pi 5

> [!TIP]
> You do **NOT** need to transfer the multi-gigabyte raw training datasets (`datasets/`) or archived experiment checkpoints (`pcb_smoke_experiment/`, `exp_c2/`, `exp_g/`, `exp_h1/`) to the Raspberry Pi. Only copy the operational application files below.

### File Manifest:

| Source Path (Dev Workstation) | Destination Path (Raspberry Pi 5) | Size | Description |
|---|---|---|---|
| `models/checkpoints/pcb_multidomain_experiment/weights/best.pt` | `~/EdgePCB-QC/models/checkpoints/pcb_multidomain_experiment/weights/best.pt` | 6.2 MB | Frozen YOLOv8n baseline PyTorch checkpoint. |
| `models/hailo/pcb_multidomain_yolov8n.hef` | `~/EdgePCB-QC/models/hailo/pcb_multidomain_yolov8n.hef` | ~4 MB | Compiled Hailo Executable Format binary (compiled on Ubuntu host). |
| `models/llama/llama3_2_1b_fp16.pte` | `~/EdgePCB-QC/models/llama/llama3_2_1b_fp16.pte` | 2.36 GB | ExecuTorch Llama 3.2 1B Instruct runtime model. |
| `models/llama/tokenizer.model` (or `tokenizer.json`) | `~/EdgePCB-QC/models/llama/` | ~9 MB | Llama 3.2 tokenizer assets. |
| `dashboard/` | `~/EdgePCB-QC/dashboard/` | ~100 KB | Streamlit UI source code (`app.py`, `utils.py`). |
| `decision_engine/` | `~/EdgePCB-QC/decision_engine/` | ~40 KB | Deterministic decision rules & `config.json`. |
| `advisory/` | `~/EdgePCB-QC/advisory/` | ~60 KB | ExecuTorch runtime & fallback logic. |
| `inspection/` | `~/EdgePCB-QC/inspection/` | ~80 KB | Core inspection modules. |
| `database/` | `~/EdgePCB-QC/database/` | ~1.3 MB | SQLite schema models & baseline inspection database. |
| `hardware/` | `~/EdgePCB-QC/hardware/` | ~80 KB | Hardware abstraction layer & deployment bash scripts. |
| `docs/` | `~/EdgePCB-QC/docs/` | ~120 KB | Architecture and deployment documentation. |

---

## 3. Automated Archive & Transfer Command

On your development PC, you can generate a clean deployment archive with one command:

```bash
tar --exclude="datasets" \
    --exclude=".venv" \
    --exclude=".pytest_cache" \
    --exclude="__pycache__" \
    --exclude="models/checkpoints/pcb_smoke_experiment" \
    --exclude="models/checkpoints/exp_*" \
    -czvf edgepcb_pi_deploy.tar.gz \
    dashboard decision_engine advisory inspection database hardware docs models README.md requirements.txt pytest.ini
```

Transfer to the Raspberry Pi:
```bash
scp edgepcb_pi_deploy.tar.gz pi@<raspberry-pi-ip>:~/
```

On the Raspberry Pi:
```bash
mkdir -p ~/EdgePCB-QC
tar -xzvf ~/edgepcb_pi_deploy.tar.gz -C ~/EdgePCB-QC
cd ~/EdgePCB-QC
sudo bash hardware/deployment/setup_pi.sh
```
