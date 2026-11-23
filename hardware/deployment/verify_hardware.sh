#!/bin/bash
# EdgePCB-QC Preflight Hardware Verification Script
# Bharat AI-SoC Challenge 2026-27 | PS5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=========================================================================="
echo "EDGEPCB-QC: PREFLIGHT HARDWARE VERIFICATION AUDIT"
echo "=========================================================================="

python3 -c "
import sys
from hardware.system.hardware_detection import get_complete_hardware_audit
from hardware.network.network_status import get_network_diagnostics
from hardware.camera.camera_manager import CameraManager

audit = get_complete_hardware_audit()
net = get_network_diagnostics()
cm = CameraManager()
cameras = cm.discover_cameras()

print(f'1. Host Device Model    : {audit[\"device_model\"]}')
print(f'   Is Raspberry Pi 5    : {audit[\"is_raspberry_pi\"]}')
print(f'   Operating System     : {audit[\"os_name\"]} ({audit[\"architecture\"]})')
print(f'   Python Version       : {audit[\"python_version\"]}')

print(f'\n2. Vision Accelerator   : {audit[\"active_vision_backend\"]}')
print(f'   Hailo Hardware Link  : {audit[\"hailo_device\"][\"status\"]}')
print(f'   HailoRT Package      : {audit[\"hailort_version\"]}')
print(f'   Compiled HEF Model   : {audit[\"hef_model\"][\"status\"]} ({audit[\"hef_model\"][\"file_name\"]})')

print(f'\n3. Discovered Cameras   : {len(cameras)} device(s) found')
for c in cameras:
    status_icon = '✓' if c.get('available') else '✗'
    print(f'   [{status_icon}] {c[\"id\"]}: {c[\"name\"]}')

print(f'\n4. Local Network & UI   :')
print(f'   Recommended URL      : {net[\"primary_url\"]}')
for iface in net[\"interfaces\"]:
    print(f'   - {iface[\"type\"]} ({iface[\"interface\"]}): http://{iface[\"ip_address\"]}:8501')

print('\n==========================================================================')
if audit[\"active_vision_backend\"].startswith('Hailo'):
    print('VERIFICATION: FULL HAILO HARDWARE ACCELERATION READY.')
else:
    print('VERIFICATION: CPU BASELINE ACTIVE (Ready for development & offline demo).')
print('==========================================================================')
"
