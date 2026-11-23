# EdgePCB-QC Hardware Troubleshooting & Diagnostic Guide
**Bharat AI-SoC Challenge 2026–27 | PS5**

This document lists symptom-based troubleshooting steps for common hardware and software integration issues on Raspberry Pi 5.

---

## 1. Camera Issues

### Symptom: `Camera Status: DISCONNECTED` or "No physical camera found"
- **Cause 1 (Ribbon Cable Direction):** The 22-pin CSI ribbon on Pi 5 must have its metal contacts facing **towards the micro-HDMI ports**. If inverted, the sensor will not receive power.
- **Cause 2 (Firmware Auto-Detect Disabled):** Verify that `/boot/firmware/config.txt` contains:
  ```ini
  camera_auto_detect=1
  ```
- **Cause 3 (Ribbon Cable Seating):** Pull the connector collar up, firmly insert the ribbon until straight, and push the collar down evenly on both sides.
- **Test Command:**
  ```bash
  libcamera-hello --list-cameras
  ```
  If detected, you will see `imx708 [4608x2592 10-bit RGGB] (/base/axi/pcie@120000/...)`.

---

## 2. Hailo AI HAT+ Issues

### Symptom: `Hailo Hardware Device: NOT CONNECTED`
- **Cause 1 (PCIe Cable Loose):** Check the 16-pin PCIe ribbon connecting the Pi 5 to the AI HAT+. Ensure the contacts face towards the USB-C power jack on the Pi side.
- **Cause 2 (PCIe Not Enabled in Firmware):** Verify `/boot/firmware/config.txt` has:
  ```ini
  dtparam=pciex1
  dtparam=pciex1_gen=3
  ```
  Reboot after adding these lines (`sudo reboot`).
- **Cause 3 (Driver Not Installed):** Check if kernel module is loaded:
  ```bash
  lspci -d 1e60:
  ls -l /dev/hailo0
  ```
  If missing, reinstall driver: `sudo apt install --reinstall hailo-all`.

### Symptom: `Compiled HEF Model: NOT COMPILED`
- Ensure you have compiled the model on an Ubuntu 22.04 LTS host using `hardware/deployment/compile_hailo.md` and copied it to `~/EdgePCB-QC/models/hailo/pcb_multidomain_yolov8n.hef`.
- Verify file size: `ls -lh models/hailo/pcb_multidomain_yolov8n.hef` (should be $\sim 3\text{ MB}$ to $6\text{ MB}$).

---

## 3. Network & Browser Connection Issues

### Symptom: Laptop browser shows "ERR_CONNECTION_REFUSED" or timeout
- **Cause 1 (Streamlit Bound to Localhost):** Ensure Streamlit was launched with `--server.address 0.0.0.0`. If bound to `127.0.0.1`, external network devices cannot connect. The startup script `./hardware/deployment/start_edgepcb.sh` binds to `0.0.0.0` automatically.
- **Cause 2 (Different Subnets):** Verify that the Pi and Laptop are on the same subnet (e.g. both `192.168.1.x`).
- **Cause 3 (Linux Firewall):** If UFW is active on the Pi, open port 8501:
  ```bash
  sudo ufw allow 8501/tcp
  ```

---

## 4. Memory & Thermal Throttling Issues

### Symptom: System slows down or ExecuTorch crashes with Out Of Memory (OOM)
- Raspberry Pi 5 has 8GB RAM shared between CPU and GPU.
- EdgePCB-QC includes an automatic memory headroom guard: if free RAM drops below $1200\text{ MB}$, the system automatically falls back to the deterministic IPC rule engine ($< 0.1\text{ ms}$, $< 1\text{ MB}$ RAM) instead of executing the 2.36GB Llama `.pte` model.
- Ensure the **Raspberry Pi 5 Active Cooler** is installed. If CPU temperature exceeds $80^\circ\text{C}$, the SoC throttles CPU frequency from $2.4\text{ GHz}$ down to $1.5\text{ GHz}$.
