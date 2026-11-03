"""
Dataset Downloader & Verifier for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Handles:
1. DsPCBSD+ (Figshare DOI: 10.6084/m9.figshare.24970329.v1)
2. DeepPCB (GitHub: https://github.com/tangsanli5201/DeepPCB)
3. PCB-Defect V2 (Mendeley: https://data.mendeley.com/datasets/vdj74sngvn/2)
"""

import os
import sys
import hashlib
import zipfile
import urllib.request
import subprocess
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = WORKSPACE_ROOT / "datasets" / "raw"

DSPCBSD_URL = "https://ndownloader.figshare.com/files/44069552"
DSPCBSD_MD5 = "508334b65bdaea7336f4c1b5d5a80a81"
DSPCBSD_DIR = RAW_DIR / "dspcbsd"

DEEPPCB_REPO = "https://github.com/tangsanli5201/DeepPCB.git"
DEEPPCB_DIR = RAW_DIR / "deeppcb"

PCB_DEFECT_V2_DIR = RAW_DIR / "pcb_defect_v2"


def calculate_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def download_file_with_progress(url: str, dest_path: Path, expected_md5: str = None):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    print(f"[*] Downloading from {url} to {dest_path.name}...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r    Progress: {percent:5.1f}% ({mb_down:6.1f} MB / {mb_total:6.1f} MB)", end="", flush=True)

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        block_size = 1024 * 64
        while chunk := response.read(block_size):
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r    Progress: {percent:5.1f}% ({mb_down:6.1f} MB / {mb_total:6.1f} MB)", end="", flush=True)

    print()
    if temp_path.exists():
        temp_path.replace(dest_path)

    if expected_md5:
        print("[*] Verifying MD5 checksum...")
        actual_md5 = calculate_md5(dest_path)
        print(f"    Expected: {expected_md5}")
        print(f"    Actual:   {actual_md5}")
        if actual_md5.lower() != expected_md5.lower():
            raise ValueError(f"MD5 mismatch for {dest_path.name}!")
        print("    [+] MD5 Checksum verified successfully.")


def setup_dspcbsd():
    print("\n" + "=" * 60)
    print("Dataset 1: DsPCBSD+ (Figshare 24970329)")
    print("=" * 60)
    DSPCBSD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DSPCBSD_DIR / "DsPCBSD+.zip"
    extracted_flag = DSPCBSD_DIR / ".extracted"

    if extracted_flag.exists():
        print("[+] DsPCBSD+ is already downloaded and extracted.")
        return

    if not zip_path.exists():
        download_file_with_progress(DSPCBSD_URL, zip_path, DSPCBSD_MD5)
    else:
        print(f"[*] Found existing archive: {zip_path.name}. Verifying MD5...")
        actual_md5 = calculate_md5(zip_path)
        if actual_md5.lower() != DSPCBSD_MD5.lower():
            print("[!] MD5 mismatch. Re-downloading...")
            download_file_with_progress(DSPCBSD_URL, zip_path, DSPCBSD_MD5)
        else:
            print("    [+] MD5 verified.")

    print(f"[*] Extracting {zip_path.name} to {DSPCBSD_DIR}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DSPCBSD_DIR)
    
    with open(extracted_flag, "w") as f:
        f.write("extracted\n")
    print("[+] DsPCBSD+ extracted successfully.")


def setup_deeppcb():
    print("\n" + "=" * 60)
    print("Dataset 3: DeepPCB (GitHub tangsanli5201/DeepPCB)")
    print("=" * 60)
    if DEEPPCB_DIR.exists() and (DEEPPCB_DIR / ".git").exists():
        print("[+] DeepPCB is already cloned.")
        return

    print(f"[*] Cloning {DEEPPCB_REPO} into {DEEPPCB_DIR}...")
    DEEPPCB_DIR.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1", DEEPPCB_REPO, str(DEEPPCB_DIR)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Git clone failed: {result.stderr}")
    else:
        print("[+] DeepPCB cloned successfully.")


def check_pcb_defect_v2():
    print("\n" + "=" * 60)
    print("Dataset 2: PCB-Defect V2 (Mendeley Data)")
    print("=" * 60)
    PCB_DEFECT_V2_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = list(PCB_DEFECT_V2_DIR.glob("*.zip"))
    img_files = list(PCB_DEFECT_V2_DIR.glob("**/*.jpg")) + list(PCB_DEFECT_V2_DIR.glob("**/*.png"))
    
    if zip_files or img_files:
        print(f"[+] Found PCB-Defect V2 files: {len(zip_files)} zip(s), {len(img_files)} image(s).")
    else:
        print("[i] Mendeley Data typically requires browser session tokens for direct downloads.")
        print(f"    Target directory: {PCB_DEFECT_V2_DIR}")
        print("    If manual download is required, place the downloaded ZIP or extracted files in:")
        print(f"    {PCB_DEFECT_V2_DIR}")


if __name__ == "__main__":
    setup_dspcbsd()
    setup_deeppcb()
    check_pcb_defect_v2()
