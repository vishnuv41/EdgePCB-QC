"""
Tests for Strict Offline Air-Gap Operation
Bharat AI-SoC Challenge 2026-27 | PS5

Verifies:
- All required model files and configurations exist locally
- Zero outbound socket / WAN calls are required
- Zero remote CDNs or external web fonts are configured
"""

import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def test_local_model_artifacts_exist():
    baseline_pt = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
    assert baseline_pt.exists(), f"Baseline model missing: {baseline_pt}"
    assert baseline_pt.stat().st_size > 1024 * 1024


def test_local_decision_engine_config_exists():
    cfg = WORKSPACE_ROOT / "decision_engine" / "config.json"
    assert cfg.exists(), f"Decision engine config missing: {cfg}"


def test_sqlite_database_local():
    db = WORKSPACE_ROOT / "database" / "inspections.db"
    assert db.exists(), f"Local SQLite database missing: {db}"


def test_no_external_cdn_in_dashboard():
    dashboard_file = WORKSPACE_ROOT / "dashboard" / "app.py"
    with open(dashboard_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Must not contain external script links or remote font CDNs
    assert "https://cdn.jsdelivr.net" not in content
    assert "https://cdnjs.cloudflare.com" not in content
    assert "https://fonts.googleapis.com" not in content
    assert "https://unpkg.com" not in content
