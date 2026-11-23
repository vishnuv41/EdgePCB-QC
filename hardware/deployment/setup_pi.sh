#!/bin/bash
# EdgePCB-QC Raspberry Pi 5 System Setup Script
# Bharat AI-SoC Challenge 2026-27 | PS5
# Target OS: Raspberry Pi OS (64-bit Bookworm)

set -e

echo "=========================================================================="
echo "EDGEPCB-QC: RASPBERRY PI 5 + AI HAT+ SYSTEM PROVISIONING"
echo "=========================================================================="

if [ "$EUID" -ne 0 ]; then
  echo "[ERROR] Please run with sudo: sudo ./hardware/deployment/setup_pi.sh"
  exit 1
fi

echo "[1/5] Updating Debian package repositories..."
apt-get update

echo "[2/5] Installing core system libraries..."
apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libcap-dev \
    v4l-utils \
    pciutils \
    dkms

echo "[3/5] Installing Raspberry Pi AI HAT+ (HailoRT) system packages..."
apt-get install -y hailo-all python3-hailort || {
    echo "[WARNING] hailo-all package could not be installed directly from current apt source."
    echo "          Ensure the Raspberry Pi package repository is active in /etc/apt/sources.list.d/"
}

echo "[4/5] Configuring PCIe Gen 3 in /boot/firmware/config.txt..."
CONFIG_FILE="/boot/firmware/config.txt"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="/boot/config.txt"
fi

if ! grep -q "dtparam=pciex1" "$CONFIG_FILE"; then
    echo "dtparam=pciex1" >> "$CONFIG_FILE"
    echo "[+] Added 'dtparam=pciex1' to $CONFIG_FILE"
fi

if ! grep -q "dtparam=pciex1_gen=3" "$CONFIG_FILE"; then
    echo "dtparam=pciex1_gen=3" >> "$CONFIG_FILE"
    echo "[+] Added 'dtparam=pciex1_gen=3' to $CONFIG_FILE"
fi

echo "[5/5] Ensuring camera permissions for non-root users..."
usermod -a -G video,render "$SUDO_USER" || true

echo "=========================================================================="
echo "PROVISIONING COMPLETE. Please reboot your Raspberry Pi 5 to apply PCIe changes:"
echo "sudo reboot"
echo "=========================================================================="
