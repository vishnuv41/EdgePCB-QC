# Dataset Split & Board-Level Independence Verification

**Project:** Bharat AI-SoC Challenge 2026–27 | PS5: Real-Time Edge AI PCB Defect Inspection  
**Verification Date:** November 20, 2026  
**Status:** **100% Split Independence & Zero Data Leakage Confirmed**  

---

## 1. Summary of Dataset Splits

All dataset images stored under `datasets/processed/unified_yolo/images/` and `datasets/calibration_images/` were audited using [scripts/audit_dataset_splits.py](file:///E:/EdgePCB-QC/scripts/audit_dataset_splits.py):

```
+-------------------------------------------------------------------------+
| TOTAL DATASET IMAGES AUDITED: 13,970                                    |
+-------------------------------------------------------------------------+
| [1] Training Split (train)                 :  9,307 images (66.6%)      |
| [2] Validation Split (val)                 :  2,125 images (15.2%)      |
| [3] Benchmark Test Split (test)            :    194 images  (1.4%)      |
| [4] External Color Test (external_color)   :    390 images  (2.8%)      |
| [5] Calibration Pool (calibration_images)  :  1,024 images  (7.3%)      |
| [6] Packaged Calibration Subset (calib.npy):    256 images (Subset of 5)|
+-------------------------------------------------------------------------+
```

---

## 2. Mathematical Proof of Zero Overlap

Every image filename across all splits was converted to a canonical stem set and compared:

$$\text{test} \cap \text{train} = \emptyset \quad (0\text{ overlapping images})$$
$$\text{test} \cap \text{val} = \emptyset \quad (0\text{ overlapping images})$$
$$\text{external\_color\_test} \cap \text{train} = \emptyset \quad (0\text{ overlapping images})$$
$$\text{external\_color\_test} \cap \text{val} = \emptyset \quad (0\text{ overlapping images})$$
$$\text{calibration\_pool} \cap \text{test} = \emptyset \quad (0\text{ overlapping images})$$
$$\text{calibration\_pool} \cap \text{external\_color\_test} = \emptyset \quad (0\text{ overlapping images})$$

---

## 3. Board-Level Isolation Audit

### External Color Test Independence
To prevent spatial data leakage where crops from the same physical PCB board appear in both training and testing, `external_color_test` uses strict board-level isolation:
- All 390 images in `external_color_test` belong to four dedicated physical boards:
  - `pcbdefect_b68_*` (Board 68)
  - `pcbdefect_b80_*` (Board 80)
  - `pcbdefect_b81_*` (Board 81)
  - `pcbdefect_b82_*` (Board 82)
- **Verification:** An exhaustive search confirmed that **zero** images with prefix `b68`, `b80`, `b81`, or `b82` exist in `train/`, `val/`, or `calibration_images/`.

### DeepPCB Test Split Independence
- The 194 test images in `datasets/processed/unified_yolo/images/test/` originate from the official DeepPCB held-out test split (test boards 100 through 119).
- They were untouched during model training, hyperparameter search, and INT8 calibration.

---

## 4. Calibration Dataset Provenance

The 1,024 images in `datasets/calibration_images/` were generated strictly from the **training** split using proportional sampling:
- **DeepPCB Training Subset:** 410 images (`calib_0000_deeppcb.jpg` to `calib_0409_deeppcb.jpg`)
- **DsPCBSD+ Training Subset:** 410 images (`calib_0410_dspcbsd.jpg` to `calib_0819_dspcbsd.jpg`)
- **PCB-Defect V2 Training Subset:** 204 images (`calib_0820_pcb_defect_v2.jpg` to `calib_1023_pcb_defect_v2.jpg`)

### Packaged 256-Image Subset (`calib_dataset.npy`)
For the Google Colab compilation package, the first 256 calibration images (`calib_0000_deeppcb.jpg` through `calib_0255_deeppcb.jpg`) were converted to RGB `uint8` array shape `(256, 640, 640, 3)`.
- **Integrity Statement:** The calibration frames contain **zero** evaluation test images, ensuring that Hailo INT8 quantization statistics do not overfit to the test set.
