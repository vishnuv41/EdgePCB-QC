# Raspberry Pi 5 & AI HAT+ Provisioning Guide
**EdgePCB-QC | Bharat AI-SoC Challenge 2026–27 | PS5**

This document provides step-by-step instructions to set up a brand-new **Raspberry Pi 5** with the **Raspberry Pi AI HAT+** and **Camera Module 3** for offline EdgePCB-QC operation.

---

## 1. Operating System Installation

1. Download and launch **Raspberry Pi Imager** on your workstation.
2. Under **Operating System**, choose:
   - `Raspberry Pi OS (64-bit)` — Debian Bookworm with desktop or Lite.
3. Under **Storage**, select your High-End MicroSD card (64GB+) or NVMe drive.
4. Click **Edit Settings** (OS Customization):
   - Set hostname: `edgepcb`
   - Set username: `pi` (or your preferred user)
   - Set a strong password
   - (Optional) Configure private Wi-Fi SSID and password
   - Under **Services**, enable **SSH** with password authentication.
5. Click **Write** and wait for verification to complete.

---

## 2. Hardware Assembly Sequence

> [!CAUTION]
> Always disconnect the USB-C power supply before connecting or disconnecting ribbon cables or HATs. Static electricity can damage the IMX708 sensor or Hailo NPU.

1. **Install Active Cooler:** Mount the heatsink onto the Pi 5 SoC and connect the 4-pin PWM fan connector.
2. **Connect Camera Ribbon Cable:**
   - Locate the `CAM/DISP0` or `CAM/DISP1` 22-pin connector on the Raspberry Pi 5.
   - Gently pull the collar up.
   - Insert the ribbon cable with contacts facing **towards the HDMI connectors**.
   - Push the collar down firmly to lock.
   - On Camera Module 3, ensure contacts face the board surface.
3. **Install Raspberry Pi AI HAT+:**
   - Insert the 16-pin PCIe flexible flat cable (FFC) into the Pi 5 PCIe port (contacts facing towards the USB-C power jack).
   - Secure the 4 brass standoffs on the Pi 5 mounting holes.
   - Press the AI HAT+ onto the 40-pin GPIO header and align the PCIe ribbon into the HAT connector.
   - Fasten with the 4 nylon screws.

---

## 3. Firmware Configuration for PCIe Gen 3 & Hailo

Boot up the Raspberry Pi 5 and edit `/boot/firmware/config.txt`:

```bash
sudo nano /boot/firmware/config.txt
```

Append the following directives at the bottom of the file:

```ini
# Enable Raspberry Pi 5 PCIe connector
dtparam=pciex1

# Force PCIe Gen 3 speed (8 Gbps lane rate for maximum Hailo NPU throughput)
dtparam=pciex1_gen=3

# Enable Camera Auto-Detection
camera_auto_detect=1
```

Save (`Ctrl+O`, `Enter`) and exit (`Ctrl+X`).

---

## 4. Install EdgePCB-QC & Dependencies

Run the automated provisioning script provided in the repository:

```bash
# 1. Clone or copy EdgePCB-QC repository to the Pi home directory
cd ~
git clone <your-edgepcb-repo-url> EdgePCB-QC
cd EdgePCB-QC

# 2. Run system setup script (installs Debian packages and Hailo drivers)
sudo bash hardware/deployment/setup_pi.sh

# 3. Reboot the Pi to initialize PCIe Gen3 driver
sudo reboot
```

After reboot:

```bash
cd ~/EdgePCB-QC

# 4. Initialize Python virtual environment with access to system Picamera2 and HailoRT
bash hardware/deployment/install_dependencies.sh
```

---

## 5. Verify Hardware Link

Execute the preflight verification script:

```bash
bash hardware/deployment/verify_hardware.sh
```

Expected confirmation:
```
1. Host Device Model    : Raspberry Pi 5 Model B Rev 1.0
   Is Raspberry Pi 5    : True
   Operating System     : Linux (aarch64)
   Python Version       : 3.11.2

2. Vision Accelerator   : Hailo-8 / Hailo-8L NPU
   Hailo Hardware Link  : OPERATIONAL
   HailoRT Package      : 4.18.0
   Compiled HEF Model   : VALIDATED (pcb_multidomain_yolov8n.hef)

3. Discovered Cameras   : 1 device(s) found
   [✓] csi:0: Raspberry Pi Camera Module 3 (CSI-0)

4. Local Network & UI   :
   Recommended URL      : http://192.168.1.100:8501
   - Ethernet (eth0)   : http://192.168.1.100:8501
```

---

## 6. Launch EdgePCB-QC

```bash
./hardware/deployment/start_edgepcb.sh
```

Open the printed URL on your laptop browser to inspect boards!
