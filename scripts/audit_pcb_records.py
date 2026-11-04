import sqlite3
import json
import hashlib
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
db_path = WORKSPACE / "database" / "inspections.db"

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()
cursor.execute("SELECT board_id, inspection_id, image_path, decision, rule_id, coverage, domain_status, image_quality_status, review_required, requires_electrical_test, vision_model_version FROM inspection_records")
rows = cursor.fetchall()
print(f"Total inspection records in DB: {len(rows)}")

targets = ["5655", "7331", "1334"]
found_targets = []
for r in rows:
    bid = r[0]
    for t in targets:
        if t in bid:
            found_targets.append(r)
            print("=" * 60)
            print(f"Match for target {t}:")
            print(f"  Board ID:                 {r[0]}")
            print(f"  Inspection ID:            {r[1]}")
            print(f"  Image Path:               {r[2]}")
            print(f"  Decision:                 {r[3]}")
            print(f"  Rule Triggered:           {r[4]}")
            print(f"  Coverage:                 {r[5]}")
            print(f"  Domain Status:            {r[6]}")
            print(f"  Image Quality Status:     {r[7]}")
            print(f"  Review Required:          {r[8]}")
            print(f"  Requires Electrical Test: {r[9]}")
            print(f"  Vision Model Version:     {r[10]}")
            
            p = Path(r[2])
            if not p.is_absolute():
                p = WORKSPACE / p
            if p.exists():
                sha256 = hashlib.sha256(open(p, 'rb').read()).hexdigest()
                size = p.stat().st_size
                print(f"  Physical File:            {p.as_posix()} (exists, {size} bytes, sha256={sha256[:16]}...)")
            else:
                print(f"  Physical File:            {p.as_posix()} (NOT FOUND)")

if not any("5655" in r[0] for r in rows):
    print("\n[!] PCB-5655 not found in DB. Checking files on disk for PCB-5655...")
    matches = list(WORKSPACE.rglob("*5655*"))
    for m in matches:
        print("  Disk match:", m.as_posix())

if not any("7331" in r[0] for r in rows):
    print("\n[!] PCB-7331 not found in DB. Checking files on disk for PCB-7331...")
    matches = list(WORKSPACE.rglob("*7331*"))
    for m in matches:
        print("  Disk match:", m.as_posix())
