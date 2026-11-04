# EdgePCB-QC: False-Positive and False-Negative Visual Investigation (Phase 3)
*Bharat AI-SoC Challenge 2026–27 | PS5*

This report investigates the physical and algorithmic failure modes that cause false-positive (FP) alarms, false-negative (FN) missed defects, and duplicate detections across unseen evaluation sets.

---

## 1. False Negative (Missed Defect) Analysis

### Scale Breakdown of Missed Defects:

```
+---------------------------------------------------------------------------------------------------+
| Evaluation Split           | Defect Scale               | Ground Truth | True Positives | Missed (FN) |
+---------------------------------------------------------------------------------------------------+
| DsPCBSD+ Validation        | Small (< 32x32 px)         | 2,067        | 1,313 (63.5%)  | 754 (36.5%) |
| (1,444 colour patches)     | Medium (32 to 96 px)       | 806          | 739 (91.7%)    | 67 (8.3%)   |
|                            | Large (> 96x96 px)         | 2            | 1 (50.0%)      | 1 (50.0%)   |
+---------------------------------------------------------------------------------------------------+
| PCB-Defect V2 External     | Small (< 32x32 px)         | 3            | 2 (66.7%)      | 1 (33.3%)   |
| (390 colour crops)         | Medium (32 to 96 px)       | 283          | 243 (85.9%)    | 40 (14.1%)  |
|                            | Large (> 96x96 px)         | 111          | 94 (84.7%)     | 17 (15.3%)  |
+---------------------------------------------------------------------------------------------------+
| DeepPCB Untouched Test     | Small (< 32x32 px)         | 116          | 114 (98.3%)    | 2 (1.7%)    |
| (194 monochrome boards)    | Medium (32 to 96 px)       | 1,023        | 944 (92.3%)    | 79 (7.7%)   |
|                            | Large (> 96x96 px)         | 4            | 4 (100.0%)     | 0 (0.0%)    |
+---------------------------------------------------------------------------------------------------+
```

### Key Finding on Missed Defects:
1. **$91.7\%$ of all missed defects in DsPCBSD+ (754 out of 822) are SMALL ($< 32\times 32\text{ px}$)**.
2. In contrast, medium defects ($32\text{ to }96\text{ px}$) achieve **$91.7\%$ recall**.
3. **Physical Causes of False Negatives:**
   - **Hairline Open Cracks:** Microscopic trace cracks (< 3 px wide) on high-density traces get smoothed out during convolutional stride operations.
   - **Shallow Spurs:** Trace edge bumps extending < 4 px from the trace boundary do not trigger anchor activations on the P3 feature map ($80\times 80$ grid).
   - **Low Solder Mask Contrast:** Dark green solder mask regions where copper traces reflect low ambient light reduce edge contrast, causing confidence to fall below $0.25$.

---

## 2. False Positive (False Alarm) Analysis

### Breakdown of False Positives:
Across the 194 test images in DeepPCB (419 FP boxes) and 195 clean reference boards (357 FP boxes):
- **Class 5 (`pin_hole_missing_pad`) accounts for $58.0\%$ of test FPs and $68.9\%$ of clean-board FPs**.
- **Physical Cause:** Legitimate circular component through-hole vias have high-contrast concentric ring edges. In monochrome images, a normal through-hole looks virtually identical to an etched pin hole void or missing pad annular ring.
- **Mouse Bite vs Trace Neckdown Confusion:** In PCB-Defect V2 external crops (64 FPs on mouse bite), designed trace neckdowns at IC pad entries are occasionally flagged as `mouse_bite`.

---

## 3. Duplicate Detection Analysis

When sliding-window tiling is applied across whole boards:
- Defect instances spanning tile boundaries receive overlapping detections from adjacent tiles.
- In `scripts/compare_whole_board_inference.py`:
  - With standard IoU fusion threshold ($0.45$), duplicate detections accounted for **$18.3\%$ of total detections**.
  - Tightening the fusion IoU threshold to **$0.30$** reduced false-positive duplicates by **$19\%$** without reducing defect recall.

---

## 4. Annotated Comparison Visualizations

Representative audit images with color-coded bounding boxes are saved in [`benchmarks/results/generalization_audit/`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/benchmarks/results/generalization_audit/):
- **Green Bounding Box:** Ground Truth Defect.
- **Cyan / Yellow Bounding Box:** True Positive Prediction (Model matched GT at $\text{IoU} \ge 0.50$).
- **Red Bounding Box:** False Positive (Prediction without corresponding GT).
- **Magenta Bounding Box:** False Negative (Missed Ground Truth Defect).

### Key Inspection Images:
1. `pcb-defect_v2_external_colour_(untouched_boards)_pcbdefect_b68_pcb_defect_206_crop002_x960_y0_audit.jpg`:
   - Demonstrates sub-millimeter short-circuit localization ($\text{IoU} = 0.825$) on unseen green solder mask trace routing.
2. `deeppcb_untouched_test_(monochrome)_deeppcb_group90100_90100000_audit.jpg`:
   - Demonstrates high-recall monochrome localization (open, short, spur captured with 100% precision on true defects).
3. `dspcbsd+_validation_(colour_patches)_dspcbsd_0032075_audit.jpg`:
   - Visualizes small spur and hole breakout detection under factory line scan lighting.
