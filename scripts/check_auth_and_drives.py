"""
Secure HF Authentication and Drive Space Verification
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import os
import sys
import shutil
from pathlib import Path
from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# 1. Check drive space
print("=" * 60)
print("1. DISK STORAGE AUDIT")
print("=" * 60)
drives = ["C", "D", "E", "F"]
drive_info = {}
for d in drives:
    root = f"{d}:\\"
    if os.path.exists(root):
        usage = shutil.disk_usage(root)
        free_gb = round(usage.free / (1024 ** 3), 2)
        total_gb = round(usage.total / (1024 ** 3), 2)
        drive_info[d] = {"free_gb": free_gb, "total_gb": total_gb}
        print(f"[*] Drive {d}: {free_gb} GB free / {total_gb} GB total")

# 2. Check .env and HF_TOKEN
print("\n" + "=" * 60)
print("2. SECURE AUTHENTICATION VERIFICATION")
print("=" * 60)
env_path = WORKSPACE_ROOT / ".env"
if not env_path.exists():
    print("[!] .env file not found at project root!")
    sys.exit(1)

load_dotenv(dotenv_path=env_path)
token = os.getenv("HF_TOKEN")

if not token:
    print("[!] HF_TOKEN is missing or empty in .env!")
    sys.exit(1)

# Verify token format without printing it
is_valid_format = token.startswith("hf_") and len(token) > 10
print(f"[*] HF_TOKEN detected in .env: True (Valid prefix 'hf_': {is_valid_format}, Length: {len(token)} chars)")

# 3. Test access to meta-llama/Llama-3.2-1B-Instruct
from huggingface_hub import HfApi
try:
    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name", "unknown")
    print(f"[+] Hugging Face authenticated user: '{username}'")

    repo_id = "meta-llama/Llama-3.2-1B-Instruct"
    print(f"[*] Querying repository metadata for '{repo_id}'...")
    info = api.model_info(repo_id=repo_id)
    print(f"[+] Successfully authenticated and verified access to: {info.id}")
    print(f"[*] Model Sha: {info.sha}")
    print(f"[*] Pipeline tag: {info.pipeline_tag}")
    files = [s.rfilename for s in info.siblings]
    print(f"[*] Repository files ({len(files)} files):")
    for f in files[:8]:
        print(f"    - {f}")
    if len(files) > 8:
        print(f"    ... and {len(files) - 8} more files")

except Exception as e:
    # Print error class and message sanitizing any token occurrences
    err_msg = str(e).replace(token, "[REDACTED_TOKEN]")
    print(f"[!] Authentication or repository access failed: {err_msg}")
    sys.exit(1)

print("\n[+] PHASE 1 VERIFICATION PASSED SUCCESSFULLY!")
