#!/bin/bash
# EdgePCB-QC Safe Shutdown Script
# Bharat AI-SoC Challenge 2026-27 | PS5

echo "[*] Stopping EdgePCB-QC Streamlit server..."
pkill -f "streamlit run dashboard/app.py" || true

echo "[*] Releasing any camera resources..."
# In case camera lock remained open
fuser -k /dev/video* 2>/dev/null || true

echo "[+] EdgePCB-QC stopped cleanly."
