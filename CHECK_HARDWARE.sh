#!/usr/bin/env bash
# ==============================================================================
# EdgePCB-QC: Hardware Audit & Diagnostic Inspection
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================================="
echo " EdgePCB-QC: Raspberry Pi 5, Hailo-8 & Camera Audit"
echo "=================================================================="

# 1. Host SBC & Architecture
ARCH=$(uname -m)
OS=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"' || uname -s)
echo "[1] Host System:"
echo "    CPU Arch:    $ARCH"
echo "    OS:          $OS"
if [ -f /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model)
    echo "    SBC Model:   $MODEL"
fi

if command -v vcgencmd &>/dev/null; then
    TEMP=$(vcgencmd measure_temp)
    echo "    Thermal:     $TEMP"
elif [ -f /sys/class/thermal/thermal_zone0/temp ]; then
    TEMP_RAW=$(cat /sys/class/thermal/thermal_zone0/temp)
    TEMP_C=$(awk "BEGIN {print $TEMP_RAW / 1000}")
    echo "    Thermal:     ${TEMP_C}°C"
fi
free -h | awk '/Mem:/ {print "    Memory:      Total: " $2 ", Used: " $3 ", Free: " $4}'

# 2. Hailo PCIe Accelerator
echo ""
echo "[2] Hailo-8 AI HAT+ Status:"
if lspci -d 1e60: &>/dev/null; then
    echo "    PCIe Device: DETECTED (Vendor 1e60)"
    lspci -d 1e60: | sed 's/^/      /'
else
    echo "    PCIe Device: NOT DETECTED via lspci"
fi

if [ -e "/dev/hailo0" ]; then
    echo "    Device Node: /dev/hailo0 (PRESENT)"
    ls -l /dev/hailo0 | sed 's/^/      /'
else
    echo "    Device Node: /dev/hailo0 (MISSING - check PCIe overlay in config.txt)"
fi

if command -v hailortcli &>/dev/null; then
    HVER=$(hailortcli --version 2>&1 | head -n 1)
    echo "    HailoRT CLI: $HVER"
else
    echo "    HailoRT CLI: NOT INSTALLED"
fi

# 3. Camera Detection (Logitech / USB / CSI)
echo ""
echo "[3] Camera Interfaces (Logitech / USB / CSI):"
if command -v lsusb &>/dev/null; then
    echo "    USB Bus Devices:"
    lsusb | grep -iE 'logitech|camera|webcam|video' | sed 's/^/      /' || echo "      (No USB video devices listed by lsusb)"
fi

echo "    V4L2 Video Nodes in sysfs:"
for name_file in /sys/class/video4linux/video*/name; do
    if [ -f "$name_file" ]; then
        node_name=$(cat "$name_file")
        node_id=$(basename $(dirname "$name_file"))
        echo "      /dev/$node_id -> $node_name"
    fi
done

if command -v v4l2-ctl &>/dev/null; then
    echo "    V4L2 Devices via v4l2-ctl:"
    v4l2-ctl --list-devices 2>/dev/null | sed 's/^/      /' || true
fi

if command -v rpicam-hello &>/dev/null; then
    echo "    CSI Cameras:"
    rpicam-hello --list-cameras 2>&1 | sed 's/^/      /' || true
fi

# 4. Production Models Check
echo ""
echo "[4] Frozen Models on Disk:"
ls -lh models/pcb/ models/llama/ 2>/dev/null | grep -E '\.(onnx|hef|pte|model)$' | awk '{print "    " $9 " (" $5 ")"}'

echo "=================================================================="
