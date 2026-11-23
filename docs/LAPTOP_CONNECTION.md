# Laptop Connection & Local Network Guide
**EdgePCB-QC | Bharat AI-SoC Challenge 2026–27 | PS5**

This document describes how to connect your laptop to the Raspberry Pi 5 to operate EdgePCB-QC over an isolated, air-gapped local network without internet access.

---

## 1. Connection Topologies

You can connect your laptop to the Raspberry Pi using either of the following two offline methods:

### Method A: Direct Ethernet Cable (Recommended)
- Connect a standard Cat5e/Cat6 RJ45 Ethernet cable directly from the laptop's Ethernet port (or USB-C Ethernet adapter) to the Raspberry Pi 5 Ethernet port.
- Both devices will self-assign link-local IPv4 addresses in the `169.254.x.x` subnet, or you can configure a static IP:
  - **Raspberry Pi:** `192.168.10.2`, Netmask: `255.255.255.0`
  - **Laptop:** `192.168.10.1`, Netmask: `255.255.255.0`
- **Advantage:** Maximum reliability, immune to RF interference, zero packet jitter.

### Method B: Private Wi-Fi Router / Hotspot
- Connect both the Raspberry Pi 5 and your laptop to the same private Wi-Fi network (or configure the Pi as a standalone Wi-Fi Access Point).
- Neither the router nor the devices need an active internet connection.

---

## 2. Discovering the Raspberry Pi IP Address

### From the Raspberry Pi Terminal:
When you launch the system using `./hardware/deployment/start_edgepcb.sh`, the script automatically detects and prints the primary local IP address:

```bash
==========================================================================
STARTING STREAMLIT SERVER (Port 8501, bound to all network interfaces)...
Open the following URL on your laptop browser (same local network):
>>> http://192.168.1.100:8501 <<<
==========================================================================
```

You can also run:
```bash
hostname -I
```

### From your Laptop:
If mDNS is active on your laptop, you can access the Pi directly by its hostname:
```
http://edgepcb.local:8501
```

---

## 3. Laptop Firewall Guidance

- Ensure your laptop firewall does not block inbound HTTP traffic on private network connections.
- On Windows: Set the Ethernet or Wi-Fi connection profile to **Private network**.
- Open Google Chrome, Mozilla Firefox, or Microsoft Edge and navigate to:
  ```
  http://<raspberry-pi-ip>:8501
  ```

---

## 4. Multi-Screen Operator Workflow

Once loaded in the laptop browser, you have complete control over all 7 industrial inspection screens:
1. `📹 Live Camera Inspection`: View live camera feed, trigger capture, run real-time inference.
2. `🔍 Captured PCB Inspection`: Pan, zoom (25%–500%), and inspect defect ROIs with 1:1 original pixel magnification.
3. `🔬 Human Review & Verification`: Triage flagged boards using keyboard hotkeys (`[C]`, `[F]`, `[U]`).
4. `📋 Inspection Log & Audit`: Search history, inspect audit payloads, and export CSV/JSON logs.
5. `📊 Pareto & Defect Analytics`: Real-time defect mode Pareto charts and yield rates.
6. `⚙️ Hardware Status & Telemetry`: Inspect Raspberry Pi CPU/RAM load, Hailo NPU state, and thermal readings.
7. `🛠️ Station Settings & Rules`: Adjust detection thresholds and inspect read-only safety presets.
