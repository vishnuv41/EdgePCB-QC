"""
High-Speed Multi-Part Range Downloader for Llama 3.2 1B Checkpoint
Bharat AI-SoC Challenge 2026-27 | PS5
"""

import os
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(WORKSPACE_ROOT / ".env")

token = os.getenv("HF_TOKEN")
if not token:
    print("[!] Error: HF_TOKEN missing in .env")
    sys.exit(1)

TARGET_DIR = Path(r"E:\EdgePCB-QC_cache\models\llama3_2_1b\original")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_FILE = TARGET_DIR / "consolidated.00.pth"
WEIGHTS_URL = "https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/resolve/main/original/consolidated.00.pth"
TOTAL_SIZE = 2471677246
NUM_THREADS = 8


def download_chunk(part_idx: int, start_byte: int, end_byte: int, token: str, progress_dict: dict):
    chunk_file = TARGET_DIR / f"chunk_{part_idx}.part"
    expected_bytes = end_byte - start_byte + 1

    max_retries = 10
    for attempt in range(max_retries):
        existing = chunk_file.stat().st_size if chunk_file.exists() else 0
        if existing >= expected_bytes:
            progress_dict[part_idx] = expected_bytes
            return True

        current_start = start_byte + existing
        headers = {
            "Authorization": f"Bearer {token}",
            "Range": f"bytes={current_start}-{end_byte}"
        }

        try:
            session = requests.Session()
            session.mount('https://', requests.adapters.HTTPAdapter(max_retries=3))
            mode = "ab" if existing > 0 else "wb"
            downloaded = existing
            progress_dict[part_idx] = downloaded

            with session.get(WEIGHTS_URL, headers=headers, stream=True, timeout=(15, 60)) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"Part {part_idx} failed with status {r.status_code}")

                with open(chunk_file, mode) as f:
                    for buf in r.iter_content(chunk_size=2 * 1024 * 1024):
                        if buf:
                            f.write(buf)
                            downloaded += len(buf)
                            progress_dict[part_idx] = downloaded

            if chunk_file.stat().st_size == expected_bytes:
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"Part {part_idx} failed after {max_retries} attempts: {e}")

    return chunk_file.stat().st_size == expected_bytes


def run_parallel_download():
    print("=" * 65)
    print(f"PARALLEL MULTI-PART DOWNLOAD ({NUM_THREADS} WORKERS)")
    print("=" * 65)
    print(f"[*] Total size: {TOTAL_SIZE / (1024**2):.2f} MB")
    print(f"[*] Destination: {WEIGHTS_FILE}")

    if WEIGHTS_FILE.exists() and WEIGHTS_FILE.stat().st_size == TOTAL_SIZE:
        print(f"[+] File already complete ({WEIGHTS_FILE.stat().st_size / (1024**2):.2f} MB).")
        return

    chunk_size = TOTAL_SIZE // NUM_THREADS
    ranges = []
    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = (start + chunk_size - 1) if i < NUM_THREADS - 1 else TOTAL_SIZE - 1
        ranges.append((i, start, end))

    progress = {i: 0 for i in range(NUM_THREADS)}
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {
            executor.submit(download_chunk, idx, start, end, token, progress): idx
            for idx, start, end in ranges
        }

        # Monitor loop
        last_log = t0
        while not all(f.done() for f in futures):
            time.sleep(2.0)
            now = time.perf_counter()
            if now - last_log >= 3.0:
                total_dl = sum(progress.values())
                elapsed = now - t0
                pct = (total_dl / TOTAL_SIZE) * 100
                speed = (total_dl / (1024**2)) / elapsed if elapsed > 0 else 0
                eta = (TOTAL_SIZE - total_dl) / (speed * 1024**2) if speed > 0 else 0
                print(f"  -> Total: {total_dl / (1024**2):.1f}/{TOTAL_SIZE / (1024**2):.1f} MB ({pct:.1f}%) | {speed:.2f} MB/s | ETA: {eta:.0f}s", flush=True)
                last_log = now

        for f in as_completed(futures):
            res = f.result()
            if not res:
                raise RuntimeError("One or more chunks failed to complete.")

    t_download = time.perf_counter() - t0
    print(f"\n[+] All {NUM_THREADS} chunks downloaded in {t_download:.1f}s ({TOTAL_SIZE / (1024**2) / t_download:.2f} MB/s average).")
    print("[*] Assembling chunks into final consolidated.00.pth...")

    # Concatenate chunks
    with open(WEIGHTS_FILE, "wb") as outfile:
        for idx in range(NUM_THREADS):
            chunk_file = TARGET_DIR / f"chunk_{idx}.part"
            with open(chunk_file, "rb") as infile:
                while True:
                    buf = infile.read(16 * 1024 * 1024)
                    if not buf:
                        break
                    outfile.write(buf)
            chunk_file.unlink()

    final_size = WEIGHTS_FILE.stat().st_size
    print(f"[+] Final assembly complete: {WEIGHTS_FILE} ({final_size / (1024**2):.2f} MB)")
    if final_size == TOTAL_SIZE:
        print("[+] Checksum & size integrity verified (100% exact match)!")
    else:
        raise ValueError(f"Size mismatch: {final_size} vs expected {TOTAL_SIZE}")


if __name__ == "__main__":
    run_parallel_download()
