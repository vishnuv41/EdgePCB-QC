# EdgePCB-QC: Unseen-Board Generalization Report (Phase 2)
*Bharat AI-SoC Challenge 2026–27 | PS5*

This report evaluates how the trained multi-domain vision detector (`YOLOv8n-pcb-multidomain-v1.0`) generalizes when presented with previously unseen bare PCB images across varying layouts, trace topologies, colours, resolutions, and lighting conditions.

---

## 1. Domain Coverage and Independence

Training images were strictly excluded from all generalization evaluations. The evaluation covers 5 distinct unseen domains:

```
+-------------------------------------------------------------------------------------------------------+
| Domain Split              | Image Count | Nature of Unseen Variance                                   |
+-------------------------------------------------------------------------------------------------------+
| DeepPCB Untouched Test    | 194 defective| Unseen physical board layouts (groups 90100 & 92000);       |
|                           | 195 clean    | monochrome scanned film; different circuit routing.         |
+-------------------------------------------------------------------------------------------------------+
| DsPCBSD+ Validation       | 1,444 crops  | Industrial line camera line-scans; green/copper contrast;   |
|                           |              | narrow traces, component pads, flux residues.               |
+-------------------------------------------------------------------------------------------------------+
| PCB-Defect V2 Validation  | 495 crops    | Tiled native crops from boards 65, 66, 67; digital camera.  |
+-------------------------------------------------------------------------------------------------------+
| PCB-Defect V2 External    | 390 crops    | Held-out physical circuit board designs (boards 68, 80, 81, |
| Colour Boards             | 25 whole brd | 82); completely distinct layout aspect ratios and routing.  |
+-------------------------------------------------------------------------------------------------------+
| Clean Bare PCB Templates  | 195 boards   | Verified defect-free golden reference boards.               |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Generalization Across Physical Variables

### 1. Different PCB Layouts & Copper Trace Patterns
- **Monochrome Layout Generalization:** On DeepPCB test groups `90100` and `92000`, the detector achieves **92.91% recall** ($1,062 / 1,143$), confirming that the multi-domain training did not induce catastrophic forgetting of monochrome layouts.
- **Colour Layout Generalization:** On PCB-Defect V2 external colour boards (68, 80, 81, 82), the detector achieves **85.39% recall** ($339 / 397$) across completely unseen trace topologies, with **98.31% recall on critical short circuits**.
- **Layout Topologies:** Generalizes effectively on orthogonal $90^\circ / 45^\circ$ trace routings and high-density IC bus arrays. Lower recall occurs on wide ground copper pours where subtle bite notches blend into copper boundaries.

### 2. Pad and Via Geometries
- **SMD Pads:** Rectangular component pads and IC footprint pins are localized with high fidelity. Missing pads are captured reliably ($78\% - 82\%$ recall).
- **Circular Through-Hole Vias:** In dense through-hole fields, the detector frequently predicts Class 5 (`pin_hole_missing_pad`) co-detections on normal circular annular rings. This is the primary contributor to clean-board false alarms.

### 3. Colour and Substrate Variations
- **Monochrome High-Contrast (Grayscale):** Recall = **92.91%**, Precision = **71.71%**, F1 = **0.8095**.
- **Green Solder Mask with Tinned Copper (RGB):** Recall = **85.39%**, Precision = **65.57%**, F1 = **0.7418**.
- **Industrial Colour Line-Scans (Green/Yellow Substrate):** Recall = **71.41%**, Precision = **72.96%**, F1 = **0.7217**.
- **Unverified Domains (Blue, Red, Black Solder Masks):** Because no public benchmark provides annotated blue/black PCBs, these domains are flagged as **`unfamiliar_domain`** by the conservative quality assessment gate, deterministically routing candidate boards to `INSPECT` (`RULE_005B`).

### 4. Image Resolution Dynamics
- **$226 \times 226$ Native Patches:** Sized up to $640 \times 640$ via letterboxing. Macro defects are localized accurately; micro-defects (< 8 px in native crop) experience feature loss.
- **$640 \times 640$ Native Crops:** Optimal operating resolution for YOLOv8n. Defect boundary localization error < 2.5 pixels.
- **$1600 \times 2048$ to $1800 \times 2048$ Whole Boards:** When evaluated via direct downsampling ($1800 \to 640$), recall collapses to **17.19%**. When evaluated via $640\times 640$ native overlapping tiling, recall is **44.41%** at **111.6 ms latency**.

---

## 3. Clean Bare PCB Evaluation (False Alarm Analysis)

Evaluated across 195 verified defect-free test boards:
- **True Clean PASS Rate:** **49.2%** (96 / 195) boards produced zero detections and cleared automatically.
- **False INSPECT Rate (Overkill):** **41.0%** (80 / 195) boards flagged for operator visual confirmation.
- **False REJECT Rate (False Scrap):** **9.7%** (19 / 195) boards rejected due to high-confidence ($> 0.75$) via pad co-predictions.
- **Key Finding:** 68.9% of all clean-board false alarms originate from Class 5 (`pin_hole_missing_pad`). Suppressing via-center co-detections elevates true clean clearance above 85%.

---

## 4. Scientific Boundaries: What Is Proven vs What Is Not

```
+---------------------------------------------------------------------------------------------------+
| PROVEN EMPIRICALLY                                 | UNPROVEN / REQUIRES CAUTION                  |
+---------------------------------------------------------------------------------------------------+
| 1. Generalizes across unseen board layouts         | 1. Universal generalization to all solder    |
|    within monochrome and green solder masks.       |    mask colours (blue, black, red unproven). |
| 2. Critical short circuit recall on unseen colour  | 2. Zero-detection does NOT prove clean       |
|    boards is 98.31% (58/59 true positives).        |    functional status (electrical unverified).|
| 3. Zero catastrophic forgetting of DeepPCB         | 3. Micro-defects < 8px require 1280 scale    |
|    monochrome baseline (92.9% vs 92.3% recall).    |    or optical zoom for reliable capture.     |
| 4. Native 640x640 tiling outperforms downsampling  | 4. Line camera line-scan artifacts may       |
|    by > 2.5x on multi-megapixel boards.            |    require camera-specific fine-tuning.      |
+---------------------------------------------------------------------------------------------------+
```
