# CLASS DEFINITION AND TAXONOMY AUDIT REPORT
*EdgePCB-QC — Bharat AI-SoC Challenge 2026–27 | PS5*

## 1. Executive Summary & Objective

In accordance with **Rule 2** ("Audit class definitions"), this investigation examines whether `pin_hole`, `missing_pad`, and `hole_breakout` are distinct, reliably labeled defect classes across the three source datasets (DeepPCB, DsPCBSD+, PCB-Defect V2), or whether splitting them creates domain-confounded labels.

---

## 2. Source Dataset Class Inventory

| Dataset | Total Images | Total Defects | Annular / Hole Class Labeled | Annotation Count | Median Size (px) | Aspect Ratio | Label Exclusivity |
|---|---|---|---|---|---|---|---|
| **DeepPCB** | 1,500 pairs | 10,013 | `pin-hole` (Class 6) | **1,501** | $33.0 \times 32.0$ | 1.14 (compact) | $100\%$ exclusive to DeepPCB |
| **DsPCBSD+** | 10,259 images | 20,276 | `HB` (Hole Breakout) | **2,883** | $40.0 \times 40.0$ | 1.03 (circular) | $100\%$ exclusive to DsPCBSD+ |
| **PCB-Defect V2** | 230 boards | 2,506 | `missing_pad` (Class 0) | **329** | $69.0 \times 69.0$ | 1.09 (compact) | $100\%$ exclusive to PCB-Defect V2 |

### Key Physical & Annotation Findings:
1. **Zero Intra-Dataset Class Coexistence:**
   - DeepPCB contains **zero** annotations for `missing_pad` or `hole_breakout`.
   - DsPCBSD+ contains **zero** annotations for `pin_hole` or `missing_pad`.
   - PCB-Defect V2 contains **zero** annotations for `pin_hole` or `hole_breakout`.
2. **Confounding with Image Modality:**
   - `pin-hole` exists *only* in synthetic, binarized/grayscale scanner images.
   - `HB` (hole breakout) exists *only* in high-contrast optical scanner colour crops.
   - `missing_pad` exists *only* in high-resolution, un-scanned whole-board camera photographs.
3. **Physical / Industrial IPC-A-600 / IPC-6012 Standard Definitions:**
   - **Pin-Hole (IPC-A-600 Section 3.1.2):** A void or pit within copper plating or solder mask exposing the laminate or underlying copper. Typically circular or irregular with $AR \approx 1.0$.
   - **Missing Pad (IPC-A-600 Section 3.3.1):** Total or severe absence of an SMT solder land or annular ring surrounding a drilled hole.
   - **Hole Breakout (IPC-A-600 Section 3.3.2):** A drilled through-hole positioned off-center such that the hole edge intersects or penetrates the external boundary of the annular ring (90° or 180° breakout violating minimum annular ring requirements).

---

## 3. The Clean-Board False Alarm Root Cause

Because all three classes represent circular copper voids or drilled annular ring geometries:
- A normal, defect-free component through-hole via or SMT circular pad possesses a circular drill hole surrounded by a concentric copper ring.
- In standard object detection training without negative examples, the visual feature representation of a concentric circle strongly overlaps with `HB` (drill hole + ring), `missing_pad` (drill hole without ring), and `pin-hole` (circular dark spot).
- In our baseline evaluation on 195 verified defect-free clean boards, **$68.9\%$ of all clean-board false alarms (246/357 boxes) were predicted as `pin_hole_missing_pad`**.
- The model lacked negative background training to learn that a well-centered through-hole with an intact copper ring is a **normal feature**, not a defect.

---

## 4. Evaluation Strategy: Two Taxonomies Compared

To strictly adhere to Rule 2 without inventing annotations, we evaluate two formal taxonomies in controlled training:

### Taxonomy 1: 6-Class Unified Canonical Taxonomy (Baseline)
- Class 0: `open_circuit`
- Class 1: `short_circuit`
- Class 2: `mouse_bite`
- Class 3: `spur`
- Class 4: `spurious_copper`
- Class 5: `pin_hole_missing_pad` (unifying all circular / annular ring / pad defects)

### Taxonomy 2: 8-Class Split Taxonomy (Experiment B)
- Class 0: `open_circuit`
- Class 1: `short_circuit`
- Class 2: `mouse_bite`
- Class 3: `spur`
- Class 4: `spurious_copper`
- Class 5: `pin_hole` (DeepPCB instances only)
- Class 6: `hole_breakout` (DsPCBSD+ instances only)
- Class 7: `missing_pad` (PCB-Defect V2 instances only)

Both taxonomies will be trained under identical budgets (50 epochs, seed 42) and evaluated against validation boards and clean boards to measure whether splitting provides real discriminative power or causes domain-aliasing errors.
