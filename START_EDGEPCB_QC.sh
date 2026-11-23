#!/usr/bin/env bash
# ==============================================================================
# EdgePCB-QC: Startup Script for Raspberry Pi 5
# ==============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment (.venv) not found. Please run ./INSTALL_PI5.sh first."
    exit 1
fi

# Set up hardware runtime environment variables
export LD_LIBRARY_PATH="/home/user/.local/lib/aarch64-linux-gnu:/usr/local/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/home/user/Downloads/hailort-4.24.0/hailort/libhailort/bindings/python/platform:${SCRIPT_DIR}:${PYTHONPATH}"
export LIBCAMERA_IPA_MODULE_PATH="/home/user/.local/lib/aarch64-linux-gnu/libcamera/ipa"
export LIBCAMERA_IPA_CONFIG_PATH="/home/user/.local/share/libcamera/ipa"

# Ensure libpisp staging directory and config exist
mkdir -p /var/tmp/pisp
if [ ! -f "/var/tmp/pisp/backend_default_config.json" ] && [ -f "/tmp/cam_extract/usr/share/libpisp/backend_default_config.json" ]; then
    cp /tmp/cam_extract/usr/share/libpisp/backend_default_config.json /var/tmp/pisp/
fi

# Detect IP address
PI_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")

echo "=================================================================="
echo " Starting EdgePCB-QC Industrial Inspection Station"
echo "=================================================================="
echo " Local Access:    http://localhost:8000"
echo " Network Access:  http://${PI_IP}:8000"
echo " Hardware Mode:   AUTO (Hailo-8 with ONNX CPU Fallback)"
echo " Operating Mode:  100% Offline / Air-Gapped"
echo "=================================================================="

export PYTHONUNBUFFERED=1

if [[ "$1" == "--daemon" || "$1" == "-d" ]]; then
    nohup "$SCRIPT_DIR/.venv/bin/python3" -u run_server.py > edgepcb_server.log 2>&1 &
    PID=$!
    echo "$PID" > .edgepcb.pid
    echo "[✓] EdgePCB-QC started in background (PID: $PID)"
    echo "    Log file: edgepcb_server.log"
else
    exec "$SCRIPT_DIR/.venv/bin/python3" run_server.py
fi
