"""
Edge System Health & Resource Monitor for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Monitors live hardware resource metrics:
- CPU utilization percentage
- RAM consumption (used, available, percentage)
- CPU temperature (from Raspberry Pi thermal zone or host sensor)
- Disk storage space (available GB, total GB, percentage)
"""

import os
import shutil
import psutil
from pathlib import Path
from typing import Dict, Any


def get_cpu_temperature() -> Dict[str, Any]:
    """Reads CPU temperature from Linux thermal zone (Raspberry Pi) or psutil sensors."""
    # Method 1: Raspberry Pi / Linux thermal zone
    thermal_zone = Path("/sys/class/thermal/thermal_zone0/temp")
    if thermal_zone.exists():
        try:
            with open(thermal_zone, "r") as f:
                temp_mc = int(f.read().strip())
                temp_c = round(temp_mc / 1000.0, 1)
                return {
                    "temperature_c": temp_c,
                    "status": "NORMAL" if temp_c < 75.0 else "THROTTLING_WARNING",
                    "sensor": "/sys/class/thermal/thermal_zone0"
                }
        except Exception:
            pass

    # Method 2: psutil sensors_temperatures (Linux)
    if hasattr(psutil, "sensors_temperatures"):
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        temp_c = round(entries[0].current, 1)
                        return {
                            "temperature_c": temp_c,
                            "status": "NORMAL" if temp_c < 75.0 else "THROTTLING_WARNING",
                            "sensor": f"psutil:{name}"
                        }
        except Exception:
            pass

    # If sensor unavailable on Windows workstation or VM
    return {
        "temperature_c": None,
        "status": "N/A (No hardware thermal sensor available on host)",
        "sensor": "None"
    }


def get_system_telemetry() -> Dict[str, Any]:
    """Reads live hardware resource consumption."""
    # CPU
    cpu_percent = psutil.cpu_percent(interval=None)

    # Memory
    mem = psutil.virtual_memory()
    total_ram_mb = round(mem.total / (1024 * 1024), 1)
    used_ram_mb = round(mem.used / (1024 * 1024), 1)
    avail_ram_mb = round(mem.available / (1024 * 1024), 1)

    # Storage
    cwd = Path.cwd()
    disk = shutil.disk_usage(cwd)
    total_disk_gb = round(disk.total / (1024 ** 3), 1)
    used_disk_gb = round(disk.used / (1024 ** 3), 1)
    free_disk_gb = round(disk.free / (1024 ** 3), 1)
    disk_percent = round((disk.used / disk.total) * 100.0, 1)

    # Thermal
    thermal = get_cpu_temperature()

    return {
        "cpu_percent": cpu_percent,
        "ram_total_mb": total_ram_mb,
        "ram_used_mb": used_ram_mb,
        "ram_avail_mb": avail_ram_mb,
        "ram_percent": mem.percent,
        "disk_total_gb": total_disk_gb,
        "disk_used_gb": used_disk_gb,
        "disk_free_gb": free_disk_gb,
        "disk_percent": disk_percent,
        "cpu_thermal": thermal
    }
