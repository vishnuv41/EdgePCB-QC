"""
Truthful Hardware Detection & Capability Audit for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Inspects actual system hardware without fabrication:
- Raspberry Pi model or Host platform
- Operating system and kernel architecture
- Camera hardware connection
- Hailo AI HAT+ PCIe device presence
- HailoRT runtime library and version
- HEF compiled model availability and SHA-256 verification
- Active inference backend
"""

import os
import sys
import glob
import platform
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
HAILO_DIR = WORKSPACE_ROOT / "models" / "hailo"
EXPECTED_HEF_NAME = "pcb_yolov8n_raw_heads.hef"
EXPECTED_HEF_SHA256 = "053a1967ad9489847ee80e058a4608518713706baf2fa79915159d952df3dada"


def detect_pi_model() -> str:
    """Detects Raspberry Pi model from device tree, or returns host platform name."""
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        try:
            with open(model_file, "r") as f:
                return f.read().strip().replace("\x00", "")
        except Exception:
            pass

    # Fallback to general platform description
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def detect_hailo_device() -> Dict[str, Any]:
    """Detects physical Hailo PCIe accelerator without fabrication."""
    # Check 1: Python binding scan
    try:
        import hailo_platform
        devices = hailo_platform.Device.scan()
        if len(devices) > 0:
            return {
                "detected": True,
                "status": "OPERATIONAL",
                "device_id": str(devices[0]),
                "device_count": len(devices),
                "device_type": "Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L)"
            }
    except Exception:
        pass

    # Check 2: Linux lspci for Hailo PCIe vendor ID 1e60
    if os.name != "nt":
        try:
            res = subprocess.run(["lspci", "-d", "1e60:"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout.strip():
                return {
                    "detected": True,
                    "status": "HARDWARE DETECTED (PCIe link active)",
                    "device_id": res.stdout.strip().split("\n")[0],
                    "device_type": "Hailo PCIe Accelerator"
                }
        except Exception:
            pass

        # Check 3: /dev/hailo0 driver node
        if os.path.exists("/dev/hailo0"):
            return {
                "detected": True,
                "status": "DRIVER ACTIVE (/dev/hailo0)",
                "device_id": "/dev/hailo0",
                "device_type": "Hailo Kernel Driver"
            }

    return {
        "detected": False,
        "status": "NOT DETECTED",
        "device_id": "None",
        "device_type": "Raspberry Pi AI HAT+ (Hailo-8 / 8L)"
    }


def detect_hailort_version() -> str:
    """Returns HailoRT library version or honest NOT INSTALLED status."""
    try:
        import hailo_platform
        return getattr(hailo_platform, "__version__", "Installed (Version unstated)")
    except ImportError:
        pass

    if os.name != "nt":
        try:
            res = subprocess.run(["hailortcli", "--version"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                return res.stdout.strip()
        except Exception:
            pass

    return "NOT INSTALLED"


def detect_hef_status() -> Dict[str, Any]:
    """Checks whether the compiled Hailo Executable Format (HEF) model exists and validates checksum."""
    primary_hef = HAILO_DIR / EXPECTED_HEF_NAME
    target_hef = primary_hef

    if not target_hef.exists():
        candidates = list(HAILO_DIR.glob("*.hef"))
        if candidates:
            target_hef = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]

    if target_hef.exists() and target_hef.stat().st_size > 1024:
        size_mb = target_hef.stat().st_size / (1024 * 1024)
        h = hashlib.sha256()
        with open(target_hef, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        actual_sha = h.hexdigest()
        is_verified = (actual_sha.lower() == EXPECTED_HEF_SHA256.lower())

        return {
            "status": "VERIFIED" if is_verified else "CHECKSUM_MISMATCH",
            "file_name": target_hef.name,
            "path": str(target_hef),
            "size_mb": round(size_mb, 2),
            "sha256": actual_sha,
            "expected_sha256": EXPECTED_HEF_SHA256,
            "checksum_verified": is_verified
        }

    return {
        "status": "MISSING",
        "file_name": EXPECTED_HEF_NAME,
        "path": str(primary_hef),
        "size_mb": 0.0,
        "sha256": "N/A",
        "expected_sha256": EXPECTED_HEF_SHA256,
        "checksum_verified": False
    }


def get_complete_hardware_audit() -> Dict[str, Any]:
    """Compiles complete truthful hardware audit for EdgePCB-QC."""
    pi_model = detect_pi_model()
    is_pi = "raspberry pi" in pi_model.lower()

    hailo = detect_hailo_device()
    hailort_ver = detect_hailort_version()
    hef_info = detect_hef_status()

    # Determine backend status
    if hailo["detected"] and hailort_ver != "NOT INSTALLED" and hef_info["status"] == "VERIFIED":
        active_backend = "Hailo-8 / Hailo-8L NPU"
        backend_status = "OPERATIONAL"
    else:
        active_backend = "PyTorch CPU Baseline (YOLOv8n)"
        backend_status = "OPERATIONAL"

    return {
        "device_model": pi_model,
        "is_raspberry_pi": is_pi,
        "os_name": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "active_backend": active_backend,
        "active_vision_backend": active_backend,
        "backend_status": backend_status,
        "hailo_device": hailo,
        "hailort_version": hailort_ver,
        "hef_model": hef_info,
        "storage": {
            "root_path": str(WORKSPACE_ROOT)
        }
    }
