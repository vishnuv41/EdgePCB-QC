"""
Tests for Hardware Detection, Telemetry & Network Status
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import pytest
from hardware.system.hardware_detection import (
    detect_pi_model, detect_hailo_device, detect_hailort_version,
    detect_hef_status, get_complete_hardware_audit
)
from hardware.system.system_monitor import get_system_telemetry, get_cpu_temperature
from hardware.network.network_status import get_local_interfaces, get_primary_connection_url, get_network_diagnostics


def test_hardware_detection_truthfulness():
    audit = get_complete_hardware_audit()
    assert isinstance(audit["device_model"], str)
    assert isinstance(audit["is_raspberry_pi"], bool)
    assert isinstance(audit["os_name"], str)
    assert isinstance(audit["architecture"], str)
    assert isinstance(audit["python_version"], str)

    # Hailo detection must be honest: if no hardware is attached, detected is False
    hailo = audit["hailo_device"]
    assert "detected" in hailo
    assert "status" in hailo

    # Backend status must match detection reality
    if not hailo["detected"]:
        assert "CPU" in audit["active_vision_backend"]


def test_system_telemetry():
    telem = get_system_telemetry()
    assert 0.0 <= telem["cpu_percent"] <= 100.0
    assert telem["ram_total_mb"] > 0
    assert telem["ram_used_mb"] > 0
    assert 0.0 <= telem["ram_percent"] <= 100.0
    assert telem["disk_total_gb"] > 0
    assert telem["disk_free_gb"] > 0

    thermal = telem["cpu_thermal"]
    assert "status" in thermal


def test_network_diagnostics():
    diag = get_network_diagnostics()
    assert diag["offline_mode"] is True
    assert isinstance(diag["primary_url"], str)
    assert diag["primary_url"].startswith("http://")
    assert ":8501" in diag["primary_url"]
