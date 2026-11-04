# EdgePCB-QC: Recommended Training Improvement Plan (Phase 5)
*Bharat AI-SoC Challenge 2026–27 | PS5*

This document formulates an evidence-based, scientifically grounded training improvement plan to resolve the failure modes identified during the multi-domain audit.

> [!IMPORTANT]
> In accordance with the engineering directives, **the existing checkpoint must not be modified or overwritten** until this training improvement plan is reviewed and approved.

---

## 1. Key Failure Modes Identified in Audit

1. **Micro-Defect Scale Collapse (DsPCBSD+):**
   - $91.7\%$ of all false negatives (754 of 822 missed defects) are small defects (< 32x32 px).
   - Standard YOLOv8 P3 stride (8 px) downsamples fine 3–8 px spurs and hairline cracks into single-pixel activations.
2. **Normal Through-Hole Via Hallucinations (Class 5):**
   - Class 5 (`pin_hole_missing_pad`) accounts for $68.9\%$ of false alarms on clean boards and $58.0\%$ on DeepPCB test.
   - The detector cannot reliably distinguish a legitimate component via from a plating pin hole or missing pad.
3. **Domain Sampling Imbalance:**
   - Training dataset consists of 5,811 DsPCBSD+ industrial crops (62.4%), 2,376 PCB-Defect V2 crops (25.5%), and only 1,120 DeepPCB monochrome crops (12.1%).
4. **Tile Boundary Duplicates:**
   - Overlapping tiles generate duplicate predictions on border defects that survive naive class-specific NMS.

---

## 2. Recommended Training Experiments

### Experiment 1: Hard-Negative Mining on Clean Bare PCB Images
- **Objective:** Eliminate the $68.9\%$ false-alarm rate caused by Class 5 hallucinations on circular component vias.
- **Implementation:**
  - Ingest the 1,120 clean template images from DeepPCB train groups and 15 clean whole boards from PCB-Defect V2 as **unannotated background images (empty `.txt` label files)**.
  - In Ultralytics YOLO, background images without bounding boxes force the loss function to penalize false-positive activations on legitimate circular vias, annular rings, and IC solder pads.
- **Expected Outcome:** Clean-board False Alarm Rate (FAR) drops from $50.8\%$ to $< 15\%$, while preserving True Clean PASS rate $> 85\%$.

### Experiment 2: Domain-Balanced Re-Sampling
- **Objective:** Prevent high-frequency DsPCBSD+ industrial patches from dominating the gradient updates over monochrome and whole-board green solder mask domains.
- **Implementation:**
  - Create a weighted dataset sampler or balance each training batch:
    - 4 images: DeepPCB monochrome scanned films (25%)
    - 6 images: PCB-Defect V2 high-res colour crops (37.5%)
    - 6 images: DsPCBSD+ industrial line-scan patches (37.5%)
  - Batch size = 16. Total virtual epoch = 4,000 steps per epoch.
- **Expected Outcome:** Equalizes gradient flow, preventing trace-edge sensitivity collapse on green solder masks.

### Experiment 3: Micro-Defect Preservation & Scale-Aware Training
- **Objective:** Recover the $36.5\%$ missed small defects (< 32 px) identified in DsPCBSD+.
- **Implementation:**
  - **Copy-Paste / Small-Object Augmentation:** Specifically crop true micro-spurs and hairline opens and paste them with random scaling ($0.8\times - 1.5\times$) onto clean copper traces during training.
  - **High-Resolution Tile Training:** Train on $640\times 640$ crops extracted from $2\times$ super-resolved or $960\times 960$ native tiles with a P2 detection head ($160\times 160$ stride-4 feature map) if Hailo DFC layer budget permits.
- **Expected Outcome:** Small-defect recall in DsPCBSD+ increases from $63.5\%$ to $> 80\%$.

### Experiment 4: Controlled Illumination and Photometric Augmentation
- **Objective:** Ensure robust cross-camera transfer across varying LED lighting angles without altering copper geometry.
- **Implementation:**
  - HSV-Value (Brightness) jitter: $\pm 30\%$ (`hsv_v = 0.30`).
  - HSV-Saturation jitter: $\pm 40\%$ (`hsv_s = 0.40`).
  - HSV-Hue jitter: strictly constrained to $\pm 1.5\%$ (`hsv_h = 0.015`) to prevent unnatural inversion of green mask into purple/red while permitting FR4 yellow/brown variance.
  - Contrast Limited Adaptive Histogram Equalization (CLAHE) variant: evaluate whether applying CLAHE during preprocessing normalizes specular reflections from shiny solder.
- **Hard Constraint:** Zero non-orthogonal geometric shearing or skewing; trace clearances must maintain true physical aspect ratios.

### Experiment 5: Domain 5 Custom Camera Fine-Tuning
- **Objective:** Final stage edge calibration using Raspberry Pi Camera Module 3 bench captures.
- **Implementation:**
  - Execute [`training/finetune_own_camera.py`](file:///c:/Users/VISHNU/Downloads/EdgePCB-QC/training/finetune_own_camera.py) with 30 epochs, frozen backbone (`freeze=10`), and learning rate $\text{lr0} = 0.001$.

---

## 3. Strict Engineering Constraints (Non-Negotiable)

1. **No Retraining Prior to Plan Signoff:** The existing `pcb_multidomain_experiment/weights/best.pt` remains the active production checkpoint.
2. **No Merging of Distinct Defect Modes:**
   - `open_circuit` and `short_circuit` must remain strictly independent output classes due to opposing electrical failure modes (loss of continuity vs dielectric breakdown).
   - `pin_hole_missing_pad` must remain flagged with its documented caveat regarding through-hole annular rings.
3. **No Unqualified PASS Output:** Optical detection alone can never verify electrical impedance or continuity. All PASS outputs must carry qualified visual clearance statements.
