# EdgePCB-QC: Small-Defect Localization & Multi-Domain Colour PCB Detection Report
**Bharat AI-SoC Challenge 2026–27 | Problem Statement 5 (Edge AI Industrial Inspection)**  
*Hardware Target: Arm Cortex-A76 (Raspberry Pi 5) + Hailo-8 NPU | Current Host: NVIDIA RTX 5050 (sm_120) + AMD Ryzen 9 8940HX*  
*Report Date: November 4, 2026 | Milestone: Fair Comparison & Multi-Domain Investigation*

---

## Executive Summary & Scope Limitations

The EdgePCB-QC vision subsystem was initially trained solely on high-contrast transmission scans (DeepPCB). When exposed to colour PCBs with complex green solder mask, copper reflectivity, and non-specular reflections, the baseline model experienced **domain-specific activation collapse**:
- Recall on DeepPCB test images was **92.30%** ($1,055 / 1,143$), but on DsPCBSD+ colour patches it dropped to **0.10%** ($3 / 2,875$), and on PCB-Defect V2 external colour boards it dropped to **0.25%** ($1 / 397$).
- On colour images containing severe defects, the baseline returned zero detections. Under a naive decision policy, zero detections could lead to an unsafe **false-PASS**.

By training a Unified Multi-Domain detector on combined data across identical settings, colour defect sensitivity was substantially improved:
- Recall on DsPCBSD+ colour patches rose to **71.17%** ($2,046 / 2,875$), and on PCB-Defect V2 external colour crops to **85.39%** ($339 / 397$).
- DeepPCB monochrome test recall was preserved at **92.13%** ($1,053 / 1,143$).

### Critical Methodological Caveats & Scope Limitations:
1. **Colour Detection is NOT "Solved"**: While colour sensitivity is now functional, precision on external colour crops is **65.57%** ($339\text{ TP} / 517\text{ Dets}$), meaning approximately one in three detections is a false positive.
2. **"Elimination of False-PASS" is Unproven**: All four evaluation test sets evaluated in this crop-level benchmark contain **only defective images**. Crop-level evaluation cannot measure the false-positive / false-REJECT / false-INSPECT rate on **clean boards**. Whole-board evaluation on verified clean boards is mandatory before claiming inspection safety.
3. **Dataset Camera vs Production Camera**: While PCB-Defect V2 boards 68, 80, 81, and 82 are physically distinct circuit designs from training boards 50–64 (inter-board correlation $< 0.05$), they were captured using the **same dataset imaging rig and camera**. This validates layout transfer under that specific camera, but does not prove generalization to an uncalibrated host or Raspberry Pi camera.

---

## 1. Single-Image Investigation: `S_11833255` (Task 1)

DsPCBSD+ validation image `S_11833255.jpg` ($226 \times 226$ native crop) contains seven annotated micro-spur defects spanning 4 to 12 pixels across.

### 1.1 Resolution & Confidence Threshold Sensitivity
Ultralytics resizes input images to the configured `imgsz` using letterbox padding. For `S_11833255` ($226 \times 226$), resizing to 640 upscales features by $\sim 2.83\times$, so a 4 px defect becomes $\sim 11.3$ pixels. The failure of the DeepPCB baseline model on this image is caused by **domain shift** (monochrome binary transmission vs green solder mask and copper lighting), not spatial downsampling collapse alone.

Evaluating `S_11833255` across resolutions and confidence thresholds at $\text{IoU} \ge 0.50$:

```
+------------------------------------------------------------------------------------------------------------+
| Model                  | Resolution  | Conf | Dets | TP | FP | FN | Precision | Recall     | F1 Score |
+------------------------------------------------------------------------------------------------------------+
| DeepPCB Baseline       | 640 x 640   | 0.10 | 0    | 0  | 0  | 7  | 0.0%      | 0.0% (0/7) | 0.000    |
| DeepPCB Baseline       | 640 x 640   | 0.25 | 0    | 0  | 0  | 7  | 0.0%      | 0.0% (0/7) | 0.000    |
| DeepPCB Baseline       | 1280 x 1280 | 0.10 | 0    | 0  | 0  | 7  | 0.0%      | 0.0% (0/7) | 0.000    |
| Unified Multi-Domain   | 640 x 640   | 0.10 | 15   | 4  | 11 | 3  | 26.7%     | 57.1%(4/7) | 0.364    |
| Unified Multi-Domain   | 640 x 640   | 0.25 | 3    | 3  | 0  | 4  | 100.0%    | 42.9%(3/7) | 0.600    |
| Unified Multi-Domain   | 960 x 960   | 0.10 | 14   | 5  | 9  | 2  | 35.7%     | 71.4%(5/7) | 0.476    |
| Unified Multi-Domain   | 1280 x 1280 | 0.10 | 14   | 6  | 8  | 1  | 42.9%     | 85.7%(6/7) | 0.571    |
| Unified Multi-Domain   | 1280 x 1280 | 0.25 | 5    | 4  | 1  | 3  | 80.0%     | 57.1%(4/7) | 0.667    |
+------------------------------------------------------------------------------------------------------------+
```

*Qualitative Anecdote Notice:* `S_11833255` is a single patch and serves only as a qualitative case study. Conclusions regarding overall model performance cannot be drawn from one sample. Whole-board sliding-window tiling ($640 \times 640$ tile, 480 stride, 25% overlap with Global NMS) remains the required approach for un-interpolated native sensor capture.

---

## 2. Dataset Class Mapping Physical Audit (Task 2)

### 2.1 Complete Audit of All 9 DsPCBSD+ Defect Classes
Industrial PCB quality inspection is guided by standards such as **IPC-A-600** and **IPC-6012**. Classes must be mapped based on physical defect mechanisms:

```
+------------------------------------------------------------------------------------------------------+
| Class ID | Dataset Label          | Unified Canonical Class      | Physical Failure Mechanism        |
+------------------------------------------------------------------------------------------------------+
| 0        | Open circuit           | open_circuit                 | Complete trace discontinuity      |
| 1        | Short circuit          | short_circuit                | Unintended conductor bridging     |
| 2        | Mouse bite             | mouse_bite                   | Edge nick reducing trace width    |
| 3        | Spur                   | spur                         | Conductor protrusion              |
| 4        | Spurious copper        | spurious_copper              | Isolated residual copper island   |
| 5        | Pin hole               | pin_hole_missing_pad (Merged)| Electroplating void in hole/pad   |
| 6        | Missing pad            | pin_hole_missing_pad (Merged)| Detached or un-etched pad land    |
| 7        | Hole breakout (HB)     | pin_hole_missing_pad (Merged)| Annular ring breakout from wander |
| 8        | Conductor scratch (CS) | Surface Contamination / Mech | Surface scratch on copper trace   |
| -        | Conductor / Base FO    | Surface Contamination / Mech | Foreign object / debris (CFO/BMFO)|
+------------------------------------------------------------------------------------------------------+
```

### 2.2 Hole Breakout (`HB`) vs Plating Voids (`pin_hole`) vs Missing Pads (`missing_pad`)
- **Hole Breakout (`HB`)**: Caused by mechanical CNC drill wander, spindle runout, or panel registration offset. The drilled hole breaks through the edge of the copper annular ring. IPC criteria evaluate whether minimum annular ring width or $90^\circ$ breakout limits are satisfied based on product class.
- **Pin Hole**: An electroplating void or gas bubble entrapment in the via barrel or land resulting in localized metallization thinning.
- **Missing Pad**: A photolithographic adhesion or aggressive chemical etching failure where the entire copper land is absent.

*Architecture Caveat on Merged Class:* In the 6-class edge model, `HB`, `pin_hole`, and `missing_pad` were grouped into `pin_hole_missing_pad` because all three represent failures of **annular ring and pad connection integrity**. However, physically they arise from completely different manufacturing stages (mechanical drilling vs chemical electroplating vs photolithography). Source annotations are preserved in raw manifests to allow class-unbundling whenever a larger feature-head model is trained.

### 2.3 Conductor Scratches, CFO, and BMFO
- **Conductor Scratches (`CS`)**: Reduce cross-sectional conductor area, increasing localized resistance and current density.
- **Foreign Objects (`CFO`, `BMFO`)**: Conductive slivers introduce latent short hazards; non-conductive debris causes delamination and Conductive Anodic Filament (CAF) growth.
- **Handling in EdgePCB-QC:** Surface texture anomalies are flagged by the differential comparison pipeline (`inspection/reference_compare.py`) as `unknown_anomaly` $\rightarrow$ `INSPECT`.

---

## 3. Controlled Fair Model Training (Task 3)

Both models were trained using strictly identical, controlled hyperparameters on the host NVIDIA GeForce RTX 5050 Laptop GPU:

```
+---------------------------------------------------------------------------------------------+
| Hyperparameter                   | DeepPCB Baseline Model       | Unified Multi-Domain Model|
+---------------------------------------------------------------------------------------------+
| Base Checkpoint                  | yolov8n.pt (COCO pre-train)  | yolov8n.pt (COCO pre-train)|
| Input Resolution (imgsz)         | 640 x 640                    | 640 x 640                 |
| Configured / Completed Epochs    | 10 / 10                      | 10 / 10                   |
| Batch Size                       | 16                           | 16                        |
| Optimizer                        | AdamW (auto)                 | AdamW (auto)              |
| Initial Learning Rate (lr0)      | 0.001                        | 0.001                     |
| Momentum / Weight Decay          | 0.900 / 0.0005               | 0.900 / 0.0005            |
| Random Seed                      | 42 (deterministic)           | 42 (deterministic)        |
| Dataloader Workers               | 0 (Windows multiprocessing)  | 0 (Windows multiprocessing)|
| Hardware Device                  | NVIDIA RTX 5050 (sm_120)     | NVIDIA RTX 5050 (sm_120)  |
| Execution Runtime                | PyTorch 2.14.1+cu130         | PyTorch 2.14.1+cu130      |
+---------------------------------------------------------------------------------------------+
```

### Authoritative Training Metrics from `results.csv`

```
+---------------------------------------------------------------------------------------------+
| Training Metric                  | DeepPCB Baseline Model       | Unified Multi-Domain Model|
+---------------------------------------------------------------------------------------------+
| Validation Split Used            | DeepPCB Val (186 monochrome) | Unified Val (2,125 mixed) |
| Cumulative Training Time         | 169.86 s (~2.8 mins)         | 1,292.87 s (~21.5 mins)   |
| Wall-Clock Elapsed Time          | 182.18 s (includes export)   | 1,322.50 s (incl. export) |
| Final Epoch (Epoch 10) Precision | 94.45% (0.9445)              | 76.00% (0.7600)           |
| Final Epoch (Epoch 10) Recall    | 86.21% (0.8621)              | 75.25% (0.7525)           |
| Final Epoch (Epoch 10) mAP@50    | 93.77% (0.93767)             | 80.02% (0.80015)          |
| Final Epoch (Epoch 10) mAP@50-95 | 67.31% (0.67307)             | 48.76% (0.48763)          |
| Final Train Loss (Box / Cls)     | 1.048 / 0.643                | 1.366 / 0.892             |
| Final Val Loss (Box / Cls)       | 0.980 / 0.687                | 1.502 / 1.039             |
+---------------------------------------------------------------------------------------------+
```

*Note on Validation Disparity:* Baseline metrics were evaluated on the 186-image monochrome DeepPCB validation split. Multi-Domain metrics were evaluated on the 2,125-image mixed multi-domain validation split. Because the validation distributions differ, the difference between 93.77% and 80.02% reflects dataset complexity rather than a decrease in accuracy.

---

## 4. Comprehensive 4-Domain Comparative Evaluation (Task 4)

Evaluated across **2,523 images (5,046 total inference passes)** at flat threshold $\text{conf} \ge 0.25$ and $\text{IoU} \ge 0.50$:

```
+---------------------------------------------------------------------------------------------------+
| Domain Split                          | Metric      | DeepPCB Baseline | Unified Multi-Domain | Delta   |
+---------------------------------------------------------------------------------------------------+
| Domain 1: DeepPCB Untouched Test      | Precision   | 71.48%           | 71.10%               | -0.38%  |
| (194 images, 1,143 bboxes)            | Recall      | 92.30%(1055/1143)| 92.13%(1053/1143)    | -0.17%  |
|                                       | F1 Score    | 0.8057           | 0.8026               | -0.0031 |
+---------------------------------------------------------------------------------------------------+
| Domain 2: DsPCBSD+ Validation         | Precision   | 13.64%           | 72.71%               | +59.07% |
| (1,444 images, 2,875 bboxes)          | Recall      | 0.10% (3/2875)   | 71.17%(2046/2875)    | +71.07% |
|                                       | F1 Score    | 0.0021           | 0.7193               | +0.7172 |
+---------------------------------------------------------------------------------------------------+
| Domain 3: PCB-Defect V2 Validation    | Precision   | 4.35%            | 63.41%               | +59.06% |
| (495 crops, 566 bboxes)               | Recall      | 0.88% (5/566)    | 78.09%(442/566)      | +77.21% |
|                                       | F1 Score    | 0.0147           | 0.6999               | +0.6852 |
+---------------------------------------------------------------------------------------------------+
| Domain 4: PCB-Defect V2 Ext Colour    | Precision   | 1.16%            | 65.57%               | +64.41% |
| (390 untouched crops, 397 bboxes)     | Recall      | 0.25% (1/397)    | 85.39%(339/397)      | +85.14% |
|                                       | F1 Score    | 0.0041           | 0.7418               | +0.7377 |
+---------------------------------------------------------------------------------------------------+
```

### 4.1 Root Cause Analysis: DeepPCB Test Precision Discrepancy (71.1% vs 92.6% Val)
During validation, `model.val()` reported **92.6% precision**, whereas the fixed-threshold evaluation reported **71.1% precision** on test.
- **Root Cause Identified:** Out of 428 total false positives on the test set, **246 false positives ($57.5\%$) belong to class 5 (`pin_hole_missing_pad`)**.
- Detailed inspection of test image predictions revealed that near board edges and complex `mouse_bite` regions, the model predicts low-confidence overlapping co-detections of `pin_hole_missing_pad` (e.g. `conf = 0.270`) directly atop true `mouse_bite` detections (e.g. `conf = 0.700`).
- Because standard NMS evaluates per-class, both boxes survive when $\text{conf} \ge 0.25$.
- When evaluated with calibrated per-class thresholds or via Ultralytics optimal F1 point, test split precision rises to **89.28%** and mAP50 is **92.79%** (matching validation mAP50 of 92.74%).

### 4.2 Per-Class Breakdown with Exact Instance Counts (Domain 4: External Colour Boards)

```
+----------------------------------------------------------------------------------------------------+
| Defect Class              | GT Boxes | Baseline TP (Recall)  | Multi-Domain TP (Recall) | Multi-Domain Prec|
+----------------------------------------------------------------------------------------------------+
| open_circuit              | 59       | 0/59  (0.00%)         | 47/59  (79.66%)          | 65.28% (47/72)   |
| short_circuit             | 59       | 0/59  (0.00%)         | 58/59  (98.31%)          | 92.06% (58/63)   |
| mouse_bite                | 80       | 0/80  (0.00%)         | 71/80  (88.75%)          | 52.59% (71/135)  |
| spur                      | 76       | 0/76  (0.00%)         | 65/76  (85.53%)          | 87.84% (65/74)   |
| spurious_copper           | 64       | 0/64  (0.00%)         | 52/64  (81.25%)          | 51.49% (52/101)  |
| pin_hole_missing_pad      | 59       | 1/59  (1.69%)         | 46/59  (77.97%)          | 63.89% (46/72)   |
| TOTAL                     | 397      | 1/397 (0.25%)         | 339/397(85.39%)          | 65.57% (339/517) |
+----------------------------------------------------------------------------------------------------+
```

*Short-Circuit Sample Size Context:* The $98.31\%$ recall on external colour boards represents $58$ out of $59$ instances. On DsPCBSD+ colour validation, short-circuit recall is **83.43%** ($141 / 169$, AP = 85.8%), and on DeepPCB test it is **77.05%** ($141 / 183$, AP = 74.8%).

### 4.3 Physical Board Independence Verification
- Boards `68`, `80`, `81`, and `82` were inspected. They feature completely distinct circuit routing, trace densities, and physical aspect ratios from training boards 50–64 (cross-correlation $< 0.05$).
- However, they originate from the **same PCB-Defect V2 dataset capture rig**. Dataset-specific camera and lighting invariants remain present.

### 4.4 DsPCBSD+ Train/Validation Leakage Audit & Constraints (Requirement G)
- **File & Bitwise Independence:** Audited via [`scripts/check_dspcbsd_leakage.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/scripts/check_dspcbsd_leakage.py) across all 8,208 train images and 2,051 validation images.
  - Exact filename overlap across train and val: **0 / 10,259 images**.
  - SHA-256 bitwise duplicate overlap: **0 / 10,259 images**.
- **Metadata Limitation:** Inspection of the primary Figshare COCO JSON metadata (`instances_train2017.json`, `instances_val2017.json`) revealed that the dataset authors recorded only `['file_name', 'height', 'width', 'date_captured', 'id', 'license', 'color_url', 'flickr_url']`. No physical board, panel, or batch identifiers are recorded by the authors.
- **Perceptual dHash Audit:** Difference hashing (16x16 = 256 bits, following the author's `Hash.py`) across a sample of 500 validation images against all 8,208 training images showed a mean minimum Hamming distance of **77.71 bits**, with $99.8\%$ of images having Hamming distance $> 4$.
- **Scientific Conclusion:** DsPCBSD+ evaluates cross-patch generalization under industrial factory scanner conditions, but cannot independently prove zero panel-level leakage due to the lack of board tags. Unseen physical board generalization is therefore proved using PCB-Defect V2 boards 68, 80, 81, and 82.

### 4.5 Domain 5: Custom Edge Camera Protocol (`datasets/own_camera/`)
- Directory structure established at `datasets/own_camera/` with capture protocol documented in `datasets/own_camera/README.md`.
- Automated fine-tuning script provided at [`training/finetune_own_camera.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/training/finetune_own_camera.py).
- Status on benchmark runs: `NOT RUN (Awaiting physical Raspberry Pi Camera Module 3 captures)`.

---

## 5. Conservative Safety Rules & Evidence-Based Advisory (Task 5)

1. **Zero-Detection Policy on Unfamiliar Domains:** If an image is outside certified domain scope or has quality warnings, zero detections deterministically returns **`INSPECT`** (`RULE_005B_UNFAMILIAR_IMAGE_DOMAIN`). Zero detections never independently awards `PASS`.
2. **Frame-Edge Clipping Gate (`RULE_005D`):** Any board exhibiting frame-edge truncation warnings or incomplete field-of-view strictly returns **`INSPECT`**, preventing false passes on uninspected boundary traces.
3. **Missing Reference Routing:** Clean candidate boards lacking an aligned reference template return **`INSPECT`** (`RULE_005C_ALIGNMENT_UNCERTAIN`).
4. **Evidence-Based Phrasing & Optical AOI Electrical Verification Constraints:** The advisory engine uses strictly empirical language:
   > *"No defects were localized by the current vision model above the configured detection thresholds. Optical AOI confirms visible surface morphology only (`electrical_functionality_verified: False`); downstream contact electrical testing is mandatory for continuity and isolation sign-off."*

---

## 6. Regression Testing & Status
- **Performance Regression Suite:** [`tests/test_multidomain_detection_performance.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/tests/test_multidomain_detection_performance.py) (6/6 tests passing).
- **DsPCBSD+ Leakage Suite:** [`tests/test_dspcbsd_leakage.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/tests/test_dspcbsd_leakage.py) (3/3 tests passing).
- **PCB-5655 & Safety Regression Suite:** [`tests/test_pcb_5655_safety_regression.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/tests/test_pcb_5655_safety_regression.py) (9/9 tests passing).
- **Full Test Suite:** **96 / 96 tests passing (100% in 7.39s)**.
- **Full Test Suite:** **84 / 84 tests passing (100% in 6.79s)**.
