"""
EdgePCB-QC Server Runner
Runs FastAPI high-performance Uvicorn server on port 8000.
"""

import sys
import uvicorn
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

if __name__ == "__main__":
    print("==================================================")
    print("  EdgePCB-QC Industrial AOI Workstation Server   ")
    print("  Port: 8000 | Host: 0.0.0.0 (Local Network Accessible)")
    print("  Web Dashboard: http://localhost:8000           ")
    print("  Offline Air-Gapped Operation                   ")
    print("==================================================")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=False, workers=1)
