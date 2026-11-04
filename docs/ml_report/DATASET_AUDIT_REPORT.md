# Dataset Usage and Training Pipeline Audit Report
**Generated:** 2026-11-04 | **EdgePCB-QC Project**

## 1. Physical Verification of Datasets Available in Workspace

| Dataset Name | Total Images Available | Annotated Defect Instances | Annotation Format | Defect Classes & Semantics | Images Used in Training (`best.pt`) | Images Used in Validation | Images Used in Testing | Licence & Attribution Source |
|---|---|---|---|---|---|---|---|---|
| **DeepPCB** | **1,500 pairs** (1,500 test + 1,500 template = 3,000 files) | **10,013** | Coordinate text files (`x1 y1 x2 y2 type`) normalized to YOLO | 6 classes: `open` (1,942), `short` (1,506), `mousebite` (1,965), `spur` (1,625), `copper` (1,474), `pin-hole` (1,501) | **1,120** (7 board groups) | **186** (2 board groups) | **194** (2 unseen board groups) | **MIT License**, Tang et al. (GitHub `tangsanli5201/DeepPCB`) |
| **DsPCBSD+** | **10,259 images** (8,207 train + 2,052 val) | **20,276** (16,184 train + 4,092 val) | Dual: COCO JSON (`instances_train2017.json`) & YOLO `.txt` | 9 industrial classes: `SH` (915), `SP` (4,584), `SC` (1,593), `OP` (1,770), `MB` (2,529), `HB` (2,883), `CS` (2,490), `CFO` (1,832), `BMFO` (1,680) | **0 (NOT USED IN `best.pt`)** | **0** | **0** | **CC BY 4.0**, Lv et al., *Nature Scientific Data* 11, 794 (2024), DOI: `10.1038/s41597-024-03656-8` |
| **PCB-Defect V2** | **230 high-res board images** (230 v1 + 230 v2) | **1,704** (v1 COCO HBB) / **2,506** (v2 OBB) | Version 1: COCO JSON (`_annotations.coco.json`); Version 2: YOLO OBB polygon (`data.yaml`) | 6 classes: `missing_pad` (329), `mouse_bite` (830), `open_circuit` (304), `short` (289), `spur` (490), `spurious_copper` (264) | **0 (NOT USED IN `best.pt`)** | **0** | **0** | **CC BY 4.0**, Rashid et al., *Data in Brief* 64 (2026) 112296, DOI: `10.1016/j.dib.2025.112296` |

---

## 2. Key Audit Findings & Training Pipeline Facts

1. **Exact Training Composition of `best.pt`:**
   - Active Checkpoint: `models/checkpoints/pcb_smoke_experiment/weights/best.pt`
   - Config file `args.yaml` points directly to: `data: datasets/processed/deeppcb_yolo/data.yaml`
   - **DeepPCB was the SOLE dataset used to train the current model.**
   - Exactly **1,120 training images** (from 7 DeepPCB board groups: `group00041`, `group12000`, `group12100`, `group12300`, `group13000`, `group20085`, `group44000`) were fed to the detector.
   - **Zero images from DsPCBSD+ or PCB-Defect V2 were ever used in the training of `best.pt`.**

2. **Image Domain & Sensor Differences:**
   - **DeepPCB:** 640x640 synthetic/monochrome binarized-style optical scans (black conductor traces on high-contrast white/cream substrate). Zero color variance.
   - **DsPCBSD+:** 226x226 cropped patches extracted from real automated optical inspection (AOI) production lines. True color photographs showing yellow/brown copper, FR-4 laminate, and industrial reflections.
   - **PCB-Defect V2:** 230 whole-board, ultra-high-resolution (median 2639 x 2538 px, up to 29.97 MP) real camera photographs of physical printed circuit boards with green solder mask, copper traces, silkscreen, and natural workshop lighting.

3. **Missing / Incompatible Annotations Analysis:**
   - **DeepPCB:** Only covers 6 classes. No annotation for surface scratches (`CS`) or foreign object debris (`CFO`, `BMFO`).
   - **DsPCBSD+:** Contains cropped patches rather than full boards. Context of whole board layout is lost.
   - **PCB-Defect V2:** Version 1 provides horizontal bounding boxes (1,704 instances). Version 2 provides oriented bounding boxes (2,506 instances). To train a standard YOLO detector, either Version 1 COCO boxes must be converted to YOLO format, or Version 2 OBB polygons must be converted to enclosing axis-aligned boxes.
   - **Semantic Disparity on Pad/Hole Defects:**
     - DeepPCB labels `pin-hole` (void or small hole in copper trace/pad).
     - PCB-Defect V2 labels `missing_pad` (completely missing solder pad).
     - DsPCBSD+ labels `HB` (Hole Breakout: hole center deviates and breaks annular ring).
     - **These are physically distinct manufacturing defect mechanisms and must not be conflated as identical without explicit qualification.**

