#!/usr/bin/env bash
# ==============================================================================
# EdgePCB-QC: Production Installer for Raspberry Pi 5 + Hailo-8
# Bharat AI-SoC Challenge 2026-27 | PS5
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE} EdgePCB-QC Industrial AOI — Raspberry Pi 5 Automated Installer   ${NC}"
echo -e "${BLUE}==================================================================${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Architecture Check
ARCH=$(uname -m)
echo -e "\n[*] Checking CPU architecture: ${YELLOW}${ARCH}${NC}"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}[!] WARNING: Architecture is ${ARCH}. Raspberry Pi 5 expects aarch64.${NC}"
    read -p "Continue anyway for simulation? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}[X] Installation aborted.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}[✓] Verified 64-bit ARM architecture (aarch64).${NC}"
fi

# 2. Check available RAM and Disk space
echo -e "\n[*] Checking system hardware resources..."
TOTAL_RAM_MB=$(free -m | awk '/Mem:/ {print $2}')
echo -e "    Total RAM: ${YELLOW}${TOTAL_RAM_MB} MB${NC}"
if [[ "$TOTAL_RAM_MB" -lt 7000 ]]; then
    echo -e "${YELLOW}[!] Note: 8 GB RAM model recommended for running concurrent Llama SLM.${NC}"
fi

FREE_DISK_MB=$(df -m . | awk 'NR==2 {print $4}')
echo -e "    Available disk space: ${YELLOW}${FREE_DISK_MB} MB${NC}"
if [[ "$FREE_DISK_MB" -lt 3000 ]]; then
    echo -e "${RED}[X] Error: At least 3 GB of free disk space is required.${NC}"
    exit 1
fi

# 3. Install required system packages
echo -e "\n[*] Installing system apt dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq || true
    sudo apt-get install -y --no-install-recommends \
        python3-pip python3-venv python3-dev build-essential \
        libgl1-mesa-glx libglib2.0-0 v4l-utils pciutils curl git || true
fi

# 3b. Add user to video and render groups for USB camera access
echo -e "\n[*] Configuring camera user permissions (video, render groups)..."
sudo usermod -aG video,render "$USER" 2>/dev/null || true

# 4. Check for Hailo-8 hardware & install stack
echo -e "\n[*] Checking for Hailo-8 AI HAT+ hardware..."
if [[ -e "/dev/hailo0" ]]; then
    echo -e "${GREEN}[✓] Hailo-8 device node active: /dev/hailo0${NC}"
else
    echo -e "${YELLOW}[!] /dev/hailo0 not yet detected.${NC}"
    echo -e "    Running Hailo setup script to configure PCIe overlay & install hailo-all..."
    bash deployment/hailo/setup_hailo_hat.sh || true
fi

# 5. Virtual Environment Setup with system packages
echo -e "\n[*] Setting up Python virtual environment (.venv)..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv --system-site-packages .venv
    echo -e "${GREEN}[✓] Created virtual environment (.venv).${NC}"
else
    echo -e "    Existing virtual environment found."
fi

source .venv/bin/activate
pip install --upgrade pip setuptools wheel -q

# 6. Install Python Dependencies
echo -e "\n[*] Installing application dependencies from requirements-pi.txt..."
pip install -r requirements-pi.txt

# 7. Model Checksums Verification
echo -e "\n[*] Verifying frozen production model files..."
python3 -c "
import hashlib
from pathlib import Path

def check_file(rel_path, exp):
    p = Path(rel_path)
    if not p.exists():
        print(f'[-] {p.name}: MISSING')
        return False
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    act = h.hexdigest()
    if act == exp:
        print(f'[✓] {p.name}: VERIFIED ({act[:16]}...)')
        return True
    else:
        print(f'[!] {p.name}: CHECKSUM MISMATCH')
        return False

check_file('models/pcb/pcb_multidomain_yolov8n_640.onnx', '0f219cae19807720f6528d635fc7168e1c067a91370e59d165a79c7536089662')
check_file('models/pcb/pcb_yolov8n_raw_heads.hef', '053a1967ad9489847ee80e058a4608518713706baf2fa79915159d952df3dada')
check_file('models/llama/llama3_2_1b_xnnpack_8da4w.pte', '2de22a466571e7e0cdd0adf01d72bed5f89d53d14784f759c06fd597616a9791')
check_file('models/llama/tokenizer.model', '82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55')
"

# 8. Set File Permissions
chmod +x *.sh deployment/**/*.sh 2>/dev/null || true

echo -e "\n${GREEN}==================================================================${NC}"
echo -e "${GREEN} EdgePCB-QC Installation Successfully Completed!                 ${NC}"
echo -e "${GREEN}==================================================================${NC}"
echo -e "Commands to operate:"
echo -e "  - Start Station:      ${BLUE}./START_EDGEPCB_QC.sh${NC}"
echo -e "  - Start as Daemon:    ${BLUE}./START_EDGEPCB_QC.sh --daemon${NC}"
echo -e "  - Check Hardware:     ${BLUE}./CHECK_HARDWARE.sh${NC}"
echo -e "  - Run Smoke Tests:    ${BLUE}./VERIFY_DEPLOYMENT.sh${NC}"
echo -e "  - Stop Station:       ${BLUE}./STOP_EDGEPCB_QC.sh${NC}"
