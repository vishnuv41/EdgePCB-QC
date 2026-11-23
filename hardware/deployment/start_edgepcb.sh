#!/bin/bash
# EdgePCB-QC Production Startup Script
# Bharat AI-SoC Challenge 2026-27 | PS5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================================================="
echo "LAUNCHING EDGEPCB-QC OFFLINE EDGE INDUSTRIAL WORKSTATION"
echo "=========================================================================="

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "[!] Virtual environment not found. Running with system python3..."
fi

# Run preflight verification
if [ -f "hardware/deployment/verify_hardware.sh" ]; then
    bash hardware/deployment/verify_hardware.sh || true
fi

# Resolve primary IP address for laptop connection
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "$HOST_IP" ]; then
    HOST_IP="127.0.0.1"
fi

echo ""
echo "=========================================================================="
echo "EDGEPCB-QC WORKSTATION READY (FastAPI + High-Performance Web Dashboard)"
echo "Port: 8000 (0.0.0.0 - Bound to all local network interfaces)"
echo ""
echo "Open this URL in your Laptop browser (connected to same Wi-Fi / Ethernet):"
echo ">>> http://${HOST_IP}:8000 <<<"
echo "=========================================================================="
echo ""

# Start FastAPI Uvicorn Server
exec python3 run_server.py
