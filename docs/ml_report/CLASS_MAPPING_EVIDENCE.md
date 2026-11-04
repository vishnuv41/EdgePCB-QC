# Scientific Defect Class Mapping & Annotation Evidence
**EdgePCB-QC Project** | *Bharat AI-SoC Challenge 2026–27*  
**Document Revision:** 2.0 (Authoritative Cross-Dataset Reconciled)

---

## 1. Primary Sources & Peer-Reviewed Literature

To ensure zero invented facts and strict adherence to industrial standards (IPC-A-600 / IPC-A-610), every defect class is grounded in official dataset publication records:

1. **DeepPCB Dataset:**
   - **Source:** Tang, S., He, F., Huang, X., & Yang, J. (2019). *Online PCB Defect Detector on a New Dataset*. arXiv:1902.06197. GitHub repository: `tangsanli5201/DeepPCB` (MIT License).
   - **Imaging Domain:** Paired 640×640 monochrome high-contrast binarized optical scans.
   - **Annotations:** Coordinate text files (`x1 y1 x2 y2 type`).

2. **DsPCBSD+ Dataset:**
   - **Source:** Lv, X., et al. (2024). *DsPCBSD+: A comprehensive dataset for printed circuit board surface defect detection*. **Nature Scientific Data** 11, 794. DOI: [`10.1038/s41597-024-03656-8`](https://doi.org/10.1038/s41597-024-03656-8). Figshare deposit ID: `24970329.v1` (CC BY 4.0).
   - **Imaging Domain:** 226×226 cropped patches from industrial automated optical inspection (AOI) lines in a volume PCB production plant under natural factory illumination.
   - **Annotations:** COCO JSON (`instances_train2017.json`, `instances_val2017.json`) and normalized YOLO `.txt` files.

3. **PCB-Defect V2 Dataset:**
   - **Source:** Rashid, A. J., Ullah, M. A., Isfara, A., Ahmed, N., Mian, M. M., & Shalehin, M. M. (2026). *PCB-defect: An annotated dataset for surface defect detection in printed circuit boards*. **Data in Brief** 64, 112296. DOI: [`10.1016/j.dib.2025.112296`](https://doi.org/10.1016/j.dib.2025.112296). Mendeley Data deposit ID: `vdj74sngvn.2` (CC BY 4.0).
   - **Imaging Domain:** 230 whole-board, ultra-high-resolution (2639×2538 median, up to 29.97 MP) real camera photographs with green solder mask, copper traces, silkscreen, and industrial lighting.
   - **Annotations:** Version 1: COCO JSON (`_annotations.coco.json`, 1,704 horizontal bounding boxes); Version 2: YOLO OBB polygon coordinates (2,506 instances).

---

## 2. Authoritative Defect Class Mapping Table

| Canonical ID | Canonical Class Name | Severity Tier | DeepPCB Label & Meaning | DsPCBSD+ Code & Meaning | PCB-Defect V2 Label & Meaning | Semantic Compatibility Assessment |
|---|---|---|---|---|---|---|
| **0** | `open_circuit` | **Critical** | `open` (ID 1): Complete discontinuity or break in conductive trace path. | `OP` (ID 4): Break in conductor path causing electrical open circuit. | `open_circuit` (ID 3): Complete conductor trace break. | **100% Identical Semantics.** Verified break in electrical continuity across all 3 datasets. |
| **1** | `short_circuit` | **Critical** | `short` (ID 2): Unintended copper bridge between adjacent conductors. | `SH` (ID 1): Unintended connection / bridging between conductors. | `short` (ID 4): Conductive bridge shorting adjacent circuits. | **100% Identical Semantics.** Verified unintended conductive bridge across all 3 datasets. |
| **2** | `mouse_bite` | **Major** | `mousebite` (ID 3): Edge void / notch taking a bite out of conductor edge. | `MB` (ID 5): Small notch or crack reducing conductor cross-section. | `mouse_bite` (ID 2): Conductor margin void / notch defect. | **100% Identical Semantics.** Edge void reducing trace ampacity across all 3 datasets. |
| **3** | `spur` | **Major** | `spur` (ID 4): Sharp projection or prong protruding from trace edge. | `SP` (ID 2): Sharp protrusion off conductor edge toward adjacent trace. | `spur` (ID 5): Conductor edge protrusion narrowing dielectric spacing. | **100% Identical Semantics.** Edge protrusion degrading dielectric clearance across all 3 datasets. |
| **4** | `spurious_copper` | **Major** | `copper` (ID 5): Isolated residual copper fleck on dielectric substrate. | `SC` (ID 3): Unwanted residual copper fleck / splash in non-conductive area. | `spurious_copper` (ID 6): Isolated copper splash on substrate. | **100% Identical Semantics.** Isolated residual conductor fleck across all 3 datasets. |
| **5** | `pin_hole_missing_pad` | **Critical** | `pin-hole` (ID 6): Void or pin-hole in copper trace or annular pad. | `HB` (ID 6): Hole Breakout (drilled hole deviates, breaking annular ring). | `missing_pad` (ID 1): Unetched or completely absent copper landing pad. | **Subtype Agglomeration with Documented Physical Distinctions** (see Section 3). |

---

## 3. Physical Distinction in Hole & Pad Anomalies

As required by scientific integrity guidelines, we do **not** claim that a pin-hole, a missing pad, and an annular breakout are identical physical defects:

1. **DeepPCB `pin-hole`:**
   - **Physical Mechanism:** Micro-void formed during copper plating or photoresist pinholing. The annular ring is present, but contains a localized hole or void through the copper foil.
   - **IPC Standard:** IPC-A-600 Section 2.5 (Voids in Conductive Pattern).
2. **PCB-Defect V2 `missing_pad`:**
   - **Physical Mechanism:** Severe photolithographic over-etching or mechanical delamination resulting in the complete absence of the copper solder pad / via land.
   - **IPC Standard:** IPC-A-600 Section 3.1 (Unsupported Holes & Annular Rings).
3. **DsPCBSD+ `HB` (Hole Breakout):**
   - **Physical Mechanism:** Drill misregistration where the via hole center shifts significantly relative to the copper landing pad, causing the drilled circumference to break out past the external boundary of the annular ring ($90^\circ$ or $180^\circ$ breakout).
   - **IPC Standard:** IPC-A-600 Section 3.3 (Annular Ring Breakout).

**Engineering Unification Justification:**
In a 6-class industrial edge detector, all three defects represent **annular ring and pad structural integrity failures**. All three defects directly compromise solder joint reliability and via interconnection, justifying their classification as **Critical (Severity Tier 1)** requiring operator inspection or rejection.

---

## 4. Excluded Classes from DsPCBSD+

The DsPCBSD+ dataset contains three additional classes that are deliberately excluded from the 6-class circuit geometry detector:

| DsPCBSD+ Code | Defect Name | Annotation Count | Justification for Exclusion |
|---|---|---|---|
| `CS` | Conductor Scratch | 2,490 | Superficial surface scratch on copper surface. In bare PCB inspection, shallow scratches that do not penetrate base laminate or violate minimum conductor thickness (IPC-A-600 2.2.1) do not warrant board rejection. Conflating scratches with circuit breaks would elevate false-reject rates. |
| `CFO` | Conductor Foreign Object | 1,832 | Particulate or fibrous contamination resting on conductive traces. These are removable surface contaminants rather than permanent copper etching defects; they are handled via ionized air cleaning. |
| `BMFO` | Base Material Foreign Object | 1,680 | Debris, particulate, or discoloration resting on bare FR-4 substrate. Non-conductive laminate discoloration that does not bridge traces does not violate circuit connectivity. |

By excluding `CS`, `CFO`, and `BMFO`, the 6-class model focuses exclusively on **true physical trace and pad defects** that directly affect electrical continuity and circuit isolation.
