"""
High-Speed Llama 3.2 1B Downloader via hf_transfer
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(WORKSPACE_ROOT / ".env")

token = os.getenv("HF_TOKEN")
if not token:
    print("[!] Error: HF_TOKEN missing in .env")
    sys.exit(1)

# Enable Rust-based multi-threaded transfer accelerator
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

TARGET_DIR = WORKSPACE_ROOT / "models" / "llama" / "checkpoints" / "Llama-3.2-1B-Instruct"
ORIGINAL_DIR = TARGET_DIR / "original"
ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)

REPO_ID = "meta-llama/Llama-3.2-1B-Instruct"

from huggingface_hub import hf_hub_download

print("=" * 65)
print("DOWNLOADING CONSOLIDATED.00.PTH VIA HF_TRANSFER")
print("=" * 65)
print(f"[*] Target directory: {ORIGINAL_DIR}")

t0 = time.perf_counter()
downloaded_file = hf_hub_download(
    repo_id=REPO_ID,
    filename="original/consolidated.00.pth",
    token=token,
    local_dir=str(TARGET_DIR)
)
elapsed = time.perf_counter() - t0
size_mb = os.path.getsize(downloaded_file) / (1024 ** 2)
print(f"[+] Download complete: {downloaded_file} ({size_mb:.2f} MB in {elapsed:.1f}s, {size_mb/elapsed:.2f} MB/s)")
