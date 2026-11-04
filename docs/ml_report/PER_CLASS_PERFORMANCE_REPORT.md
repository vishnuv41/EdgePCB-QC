# EdgePCB-QC: Per-Class Performance and Defect Mode Report
*Bharat AI-SoC Challenge 2026–27 | PS5*

This report details the quantitative detection performance of the current multi-domain checkpoint (`pcb_multidomain_experiment/weights/best.pt`) evaluated across all unseen domains.

---

## 1. Per-Class Performance Matrix Across Unseen Evaluation Domains

Evaluated at operating point $\text{conf} = 0.25$, $\text{IoU} \ge 0.50$ matching:

### Table 1: Defect Recall ($n / N$, %) Across Unseen Domains

| Canonical Defect Mode | Severity | DeepPCB Test (194 Monochrome Boards) | DsPCBSD+ Val (1,444 Colour Patches) | PCB-Defect V2 Val (495 Tiled Crops) | External Colour Test (Boards 68, 80, 81, 82) | Unseen Colour Average Recall |
|---|---|---|---|---|---|---|
| **`short_circuit`** | **Critical** | 77.60% (142 / 183) | **83.43%** (141 / 169) | 75.00% (66 / 88) | **98.31%** (58 / 59) | **85.58%** |
| **`open_circuit`** | **Critical** | **97.17%** (240 / 247) | **81.36%** (275 / 338) | 84.72% (61 / 72) | **79.66%** (47 / 59) | **81.91%** |
| **`pin_hole_missing_pad`** | **Critical** | **98.38%** (182 / 185) | **98.03%** (596 / 608) | 82.57% (90 / 109) | **77.97%** (46 / 59) | **86.19%** |
| **`mouse_bite`** | **Major** | **93.30%** (167 / 179) | 66.48% (363 / 546) | 69.30% (79 / 114) | **88.75%** (71 / 80) | **74.84%** |
| **`spur`** | **Major** | **93.48%** (172 / 184) | 51.45% (478 / 929) | 74.47% (70 / 94) | **85.53%** (65 / 76) | **70.48%** |
| **`spurious_copper`** | **Major** | **96.36%** (159 / 165) | 70.18% (200 / 285) | 88.76% (79 / 89) | **81.25%** (52 / 64) | **80.06%** |
| **OVERALL** | — | **92.91%** (1,062 / 1,143) | **71.41%** (2,053 / 2,875) | **78.62%** (445 / 566) | **85.39%** (339 / 397) | **78.47%** |

---

### Table 2: Precision and F1 Score Across Unseen Domains

| Canonical Defect Mode | DeepPCB Test (Prec / F1) | DsPCBSD+ Val (Prec / F1) | PCB-Defect V2 Val (Prec / F1) | Ext Colour Test (Prec / F1) | Macro F1 Across Domains |
|---|---|---|---|---|---|
| **`short_circuit`** | 73.20% / **0.7533** | 77.90% / **0.8057** | 86.84% / **0.8049** | 92.06% / **0.9508** | **0.8287** |
| **`open_circuit`** | 81.36% / **0.8856** | 69.80% / **0.7514** | 63.54% / **0.7262** | 65.28% / **0.7176** | **0.7702** |
| **`pin_hole_missing_pad`** | 42.82% / 0.5967* | 82.43% / **0.8956** | 90.00% / **0.8612** | 63.89% / **0.7023** | **0.7640** |
| **`mouse_bite`** | 79.52% / **0.8586** | 65.76% / **0.6612** | 48.17% / 0.5683 | 52.59% / **0.6605** | **0.6872** |
| **`spur`** | 92.97% / **0.9322** | 69.28% / 0.5905 | 64.81% / **0.6931** | 87.84% / **0.8667** | **0.7706** |
| **`spurious_copper`** | 92.44% / **0.9436** | 72.99% / **0.7156** | 51.63% / 0.6529 | 51.49% / 0.6303 | **0.7356** |
| **MACRO AVERAGE** | **77.05% / 0.8284** | **73.03% / 0.7367** | **67.50% / 0.7178** | **68.86% / 0.7547** | **0.7594** |

*\*Precision Caveat on DeepPCB:* Class 5 precision (42.82%) reflects 243 false-positive boxes on normal drilled through-hole vias in scanned monochrome films.

---

## 2. In-Depth Defect Mode Analysis

### 1. Critical Short Circuit (`short_circuit`)
- **Generalization:** Highest precision (92.06%) and F1 (0.9508) on unseen colour boards (58/59 true positives).
- **Physical Signature:** Conductive copper bridges between distinct trace conductors exhibit high edge contrast and anomalous geometry that the detector localizes reliably.
- **Electrical Implication:** Triggers `RULE_001` or `RULE_003` -> mandatory in-circuit continuity testing prior to panel release.

### 2. Critical Open Circuit (`open_circuit`)
- **Generalization:** Consistently high recall (97.17% on monochrome, 81.36% on DsPCBSD+, 79.66% on colour boards).
- **Physical Failure Mechanism:** Trace discontinuity or complete trace gap.
- **Missed Defects:** Misses occur predominantly on hairline trace cracks (< 3 px width) where downsampling collapses the gap.

### 3. Critical Pin Hole & Missing Pad (`pin_hole_missing_pad`)
- **Generalization:** High recall (> 86% average), but prone to over-detection.
- **Failure Mode:** On clean boards with dense circular through-hole via arrays, the model frequently predicts low-confidence Class 5 co-detections on normal annular rings.
- **Remedy:** Hard-negative mining with legitimate clean PCB images and circular via suppression.

### 4. Major Trace Irregularities (`mouse_bite` and `spur`)
- **Generalization:** `spur` exhibits 51.45% recall on DsPCBSD+ vs 85.53% on whole colour boards and 93.48% on DeepPCB.
- **Scale Dynamics:** In DsPCBSD+, many spurs are micro-prongs (< 8 px). When downsampled, the prong blends into the trace edge, escaping P3/P4 anchor strides.
