#!/usr/bin/env bash
# ==============================================================================
# EdgePCB-QC: Automated Deployment Smoke-Test Runner
# ==============================================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

MODE="${1:-all}"
python3 tests/deployment_smoke_test.py --mode="$MODE"
