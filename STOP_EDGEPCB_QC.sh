#!/usr/bin/env bash
# ==============================================================================
# EdgePCB-QC: Stop Script
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Stopping EdgePCB-QC Server ==="

if [ -f ".edgepcb.pid" ]; then
    PID=$(cat .edgepcb.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "[*] Terminating process $PID..."
        kill "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID"
        fi
        echo "[✓] Process $PID stopped."
    fi
    rm -f .edgepcb.pid
fi

# Find any remaining uvicorn/backend instances
PIDS=$(pgrep -f "run_server.py|backend.app:app" || true)
if [ -n "$PIDS" ]; then
    echo "[*] Stopping lingering instances: $PIDS"
    kill $PIDS 2>/dev/null || true
    sleep 1
    kill -9 $PIDS 2>/dev/null || true
fi

echo "[✓] EdgePCB-QC is stopped."
