#!/bin/bash
# EdgePCB-QC Python Environment & Dependency Installation
# Bharat AI-SoC Challenge 2026-27 | PS5

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=========================================================================="
echo "EDGEPCB-QC: PYTHON ENVIRONMENT INITIALIZATION"
echo "=========================================================================="
echo "Project Root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

# Create virtual environment with --system-site-packages so Picamera2 and HailoRT are accessible
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python 3 virtual environment (.venv) with system-site-packages..."
    python3 -m venv --system-site-packages .venv
fi

echo "[*] Activating virtual environment..."
source .venv/bin/activate

echo "[*] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "[*] Installing Raspberry Pi dependencies from hardware/requirements_pi.txt..."
pip install -r hardware/requirements_pi.txt

echo "[*] Verifying import of core dependencies..."
python3 -c "
import cv2, numpy, streamlit, plotly, sqlalchemy, psutil
print('  [PASS] Core libraries loaded successfully.')
"

echo "[*] Checking Picamera2 and HailoRT bindings..."
python3 -c "
try:
    import picamera2
    print('  [PASS] Picamera2 binding detected.')
except ImportError:
    print('  [INFO] Picamera2 not found in system site-packages (required for CSI camera).')

try:
    import hailo_platform
    print('  [PASS] HailoRT hailo_platform binding detected.')
except ImportError:
    print('  [INFO] hailo_platform not found (CPU baseline fallback will be used until AI HAT+ is installed).')
"

echo "=========================================================================="
echo "ENVIRONMENT SETUP COMPLETE. Launch EdgePCB-QC using:"
echo "./hardware/deployment/start_edgepcb.sh"
echo "=========================================================================="
