from pathlib import Path
from collections import Counter

DSP_DIR = Path("datasets/raw/dspcbsd/Data_YOLO/labels")

train_counts = Counter()
val_counts = Counter()

for f in (DSP_DIR / "train").glob("*.txt"):
    with open(f, "r", encoding="utf-8") as fp:
        for line in fp:
            p = line.strip().split()
            if p:
                train_counts[int(p[0])] += 1

for f in (DSP_DIR / "val").glob("*.txt"):
    with open(f, "r", encoding="utf-8") as fp:
        for line in fp:
            p = line.strip().split()
            if p:
                val_counts[int(p[0])] += 1

class_names = {
    0: ("SH", "Short circuit"),
    1: ("SP", "Spur"),
    2: ("SC", "Spurious copper"),
    3: ("OP", "Open circuit"),
    4: ("MB", "Mouse bite"),
    5: ("HB", "Hole breakout / Missing pad"),
    6: ("CS", "Conductor scratch"),
    7: ("CFO", "Conductor foreign object"),
    8: ("BMFO", "Base material foreign object"),
}

canonical_mapping = {
    0: (1, "short_circuit"),
    1: (3, "spur"),
    2: (4, "spurious_copper"),
    3: (0, "open_circuit"),
    4: (2, "mouse_bite"),
    5: (5, "pin_hole_missing_pad"),
    6: (None, "EXCLUDED"),
    7: (None, "EXCLUDED"),
    8: (None, "EXCLUDED"),
}

print(f"{'Raw ID':<8}{'Code':<6}{'Description':<32}{'Train':<10}{'Val':<10}{'Total':<10}{'Canonical ID':<14}{'Canonical Class':<22}{'Status':<10}")
print("-" * 122)

total_all = 0
retained_total = 0
excluded_total = 0

for cid in range(9):
    code, desc = class_names[cid]
    tr = train_counts[cid]
    va = val_counts[cid]
    tot = tr + va
    total_all += tot
    c_id, c_name = canonical_mapping[cid]
    status = "RETAINED" if c_id is not None else "EXCLUDED"
    if c_id is not None:
        retained_total += tot
    else:
        excluded_total += tot
    c_id_str = str(c_id) if c_id is not None else "N/A"
    print(f"{cid:<8}{code:<6}{desc:<32}{tr:<10}{va:<10}{tot:<10}{c_id_str:<14}{c_name:<22}{status:<10}")

print("-" * 122)
print(f"Total instances in DsPCBSD+: {total_all}")
print(f"Retained in 6-class model: {retained_total} ({retained_total/total_all*100:.2f}%)")
print(f"Excluded from 6-class model: {excluded_total} ({excluded_total/total_all*100:.2f}%)")
