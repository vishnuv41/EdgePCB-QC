# EdgePCB-QC: Authoritative Dataset Audit Report (Phase 1)
*Bharat AI-SoC Challenge 2026–27 | PS5*

This audit report provides an exhaustive, evidence-based quantification of all raw and processed datasets utilized in the EdgePCB-QC quality-control pipeline.

---

## 1. Unified Dataset Split Summary

All images are consolidated in [`datasets/processed/unified_yolo/`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/datasets/processed/unified_yolo/) with zero data leakage across splits.

| Split Name | Image Count | Label Files | Total Bounding Boxes | Empty-Label Images | DeepPCB Samples | DsPCBSD+ Samples | PCB-Defect V2 Samples | Dominant Color Space |
|---|---|---|---|---|---|---|---|---|
| **`train`** | **9,307** | 9,307 | 22,066 | 172 (1.8%) | 1,120 | 5,811 | 2,376 | 88% RGB / 12% Monochrome |
| **`val`** | **2,125** | 2,125 | 4,431 | 33 (1.6%) | 186 | 1,444 | 495 | 91% RGB / 9% Monochrome |
| **`test`** (Untouched) | **194** | 194 | 1,143 | 0 (0.0%) | 194 | 0 | 0 | 100% Monochrome Grayscale |
| **`external_color_test`** | **390** | 390 | 397 | 49 (12.6%) | 0 | 0 | 390 | 100% RGB (Boards 68, 80, 81, 82) |
| **TOTAL** | **12,016** | **12,016** | **28,037** | **254 (2.1%)** | **1,500** | **7,255** | **3,261** | **Multi-Domain Unified** |

*Note on Empty-Label Images:* Empty labels represent clean board margins or background tiles generated during overlapping native tiling. They act as natural negative examples during training.

---

## 2. Canonical Class Mapping & Annotation Count

To resolve semantic conflicts between raw datasets, EdgePCB-QC maps raw labels to 6 canonical defect classes:

| Class ID | Canonical Defect Mode | Severity | DeepPCB Source Label | DsPCBSD+ Source Code | PCB-Defect V2 Label | Train BBoxes | Val BBoxes | Test BBoxes | Ext Color Test BBoxes | Total Instances | Training Sufficiency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **`0`** | `open_circuit` | **Critical** | `open` | `OP` | `open_circuit` | 3,381 | 584 | 247 | 59 | **4,271** | **Sufficient** (>3k) |
| **`1`** | `short_circuit` | **Critical** | `short` | `SH` | `short` | 2,297 | 403 | 183 | 59 | **2,942** | **Sufficient** (>2k) |
| **`2`** | `mouse_bite` | **Major** | `mousebite` | `MB` | `mouse_bite` | 4,161 | 839 | 179 | 80 | **5,259** | **Sufficient** (>4k) |
| **`3`** | `spur` | **Major** | `spur` | `SP` | `spur` | 5,406 | 1,178 | 184 | 76 | **6,844** | **Sufficient** (>5k) |
| **`4`** | `spurious_copper` | **Major** | `copper` | `SC` | `spurious_copper` | 2,939 | 521 | 165 | 64 | **3,689** | **Sufficient** (>2.5k) |
| **`5`** | `pin_hole_missing_pad` | **Critical** | `pin-hole` | `HB` (Hole Breakout) | `missing_pad` | 3,882 | 906 | 185 | 59 | **5,032** | **Sufficient** (>3.5k) |
| **TOTAL**| — | — | — | — | — | **22,066**| **4,431** | **1,143** | **397** | **28,037** | **All $\ge 2,000$** |

---

## 3. Image Dimensions, Aspect Ratios, and Colour Formats

```
+---------------------------------------------------------------------------------------------------------+
| Dataset Source | Native Image Dimensions      | Channels & Colour Space     | Lighting / Capture Method |
+---------------------------------------------------------------------------------------------------------+
| DeepPCB        | 640 x 640 (1.00 aspect)      | 1-channel / Grayscale       | Factory flatbed scanner   |
| DsPCBSD+       | 226 x 226 (1.00 aspect)      | 3-channel RGB (Industrial)  | Automated AOI line camera |
| PCB-Defect V2  | 1168x2048 to 1792x2048 (HiRes)| 3-channel RGB (Green mask)  | High-res digital camera   |
+---------------------------------------------------------------------------------------------------------+
```

### Tiling Strategy for Whole Boards (PCB-Defect V2):
- Whole-board high-resolution images ($1168\times 2048$ to $1792\times 2048$) cannot be downscaled directly to $640\times 640$ without collapsing fine $0.15\text{ mm}$ traces into 1–2 pixels.
- The training pipeline slices whole boards into overlapping $640\times 640$ windows with a **480-pixel stride ($25\%$ overlap)** at native pixel resolution.
- Bounding boxes are dynamically intersected with tile boundaries; boxes whose area inside the crop exceeds $30\%$ of their original area are retained and clipped.

---

## 4. Discarded and Excluded Annotations

A total of **6,002 annotations** from the raw DsPCBSD+ dataset were deliberately excluded from the 6-class YOLO detector:
- `CS` (Conductor Scratch): **2,490 annotations**
- `CFO` (Conductor Foreign Object): **1,832 annotations**
- `BMFO` (Base Material Foreign Object): **1,680 annotations**

### Engineering Justification:
1. **IPC-A-600 / IPC-6012 Physical Criteria:** Surface scratches (`CS`) and dust/contaminants (`CFO`, `BMFO`) are surface anomalies rather than copper etching defects. Under IPC-6012, a scratch only constitutes a functional rejection if it reduces the conductive trace thickness by $>10\%$ or breaches minimum electrical clearance.
2. **Avoiding Bounding-Box Detector Confusion:** Debris and superficial surface abrasions exhibit arbitrary textures and lack consistent geometric boundary prongs or necks. Training a bounding-box model to localize every microscopic dust particle causes catastrophic precision degradation on clean traces.
3. **Reference Comparison Routing:** In EdgePCB-QC, particulate debris and unclassified surface anomalies are routed to the Golden Reference Comparison pipeline ([`inspection/reference_compare.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/inspection/reference_compare.py)), which flags them as `unknown anomaly` -> `INSPECT`.

---

## 5. Source-Board Separation & Leakage Audit

### 1. DeepPCB (11 Board Groups)
- Partitioned strictly by physical board group:
  - **Train (7 groups):** `group00041`, `group12000`, `group12100`, `group12300`, `group13000`, `group20085`, `group44000` (1,120 test pairs).
  - **Validation (2 groups):** `group50600`, `group77000` (186 test pairs).
  - **Untouched Test (2 groups):** `group90100`, `group92000` (194 test pairs).
- **Leakage Status:** **Zero board leakage**. No layout in the test split appears in training.

### 2. PCB-Defect V2 (22 Physical Boards)
- Partitioned strictly by physical board number prefix:
  - **Train (15 boards):** Boards `50` through `64` (2,376 native crops).
  - **Validation (3 boards):** Boards `65`, `66`, `67` (495 native crops).
  - **External Colour Test (4 boards):** Boards `68`, `80`, `81`, `82` (390 native crops).
- **Leakage Status:** **Zero board leakage**. Boards `68, 80, 81, 82` have cross-correlation $< 0.05$ with training boards and feature completely distinct trace routing.

### 3. DsPCBSD+ (10,259 Patches)
- Partitioned according to authors' official split: `train2017` (8,208 images) and `val2017` (2,051 images).
- **Leakage Status:** Verified via [`scripts/check_dspcbsd_leakage.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/scripts/check_dspcbsd_leakage.py).
  - Exact filename overlap: **0 / 10,259**.
  - Bitwise SHA-256 duplicate overlap: **0 / 10,259**.
  - Dataset limitation: The original Figshare deposit does not record physical panel or board IDs. Therefore, DsPCBSD+ validates cross-patch generalization under factory scanner conditions, while board-level independence is proven using PCB-Defect V2 boards 68, 80, 81, 82.
