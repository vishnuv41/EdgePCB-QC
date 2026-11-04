# EdgePCB-QC — Dataset Class Mapping Audit & Physical Defect Analysis
*Bharat AI-SoC Challenge 2026–27 | PS5*
*Authoritative Technical Investigation on Defect Semantics, IPC Standards, and Cross-Dataset Reconciliation*

---

## 1. Executive Summary

A critical challenge in automated optical inspection (AOI) for printed circuit boards (PCBs) is the reconciliation of disparate annotation ontologies across factory lines. This audit provides an evidence-based, scientifically grounded review of all nine defect classes in **DsPCBSD+** (DOI: `10.1038/s41597-024-03656-8`), evaluates the physical implications of merging **Hole Breakout (`HB`)** into **Pin Hole / Missing Pad**, and rigorously investigates the industrial significance of **Conductor Scratches (`CS`)**, **Conductor Foreign Objects (`CFO`)**, and **Base Material Foreign Objects (`BMFO`)**.

---

## 2. Complete Audit of All Nine DsPCBSD+ Defect Classes

The primary annotations in `datasets/raw/dspcbsd/Data_YOLO/labels/` encompass **20,276 defect instances** across train and validation splits:

| Raw ID | Code | Source Defect Class | Train Count | Val Count | Total Count | Relative Frequency | Canonical 6-Class ID | Canonical Name | 6-Class Status |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **0** | `SH` | Short Circuit | 746 | 169 | 915 | 4.51% | 1 | `short_circuit` | **RETAINED** |
| **1** | `SP` | Spur | 3,655 | 929 | 4,584 | 22.61% | 3 | `spur` | **RETAINED** |
| **2** | `SC` | Spurious Copper | 1,308 | 285 | 1,593 | 7.86% | 4 | `spurious_copper` | **RETAINED** |
| **3** | `OP` | Open Circuit | 1,432 | 338 | 1,770 | 8.73% | 0 | `open_circuit` | **RETAINED** |
| **4** | `MB` | Mouse Bite | 1,983 | 546 | 2,529 | 12.47% | 2 | `mouse_bite` | **RETAINED** |
| **5** | `HB` | Hole Breakout | 2,275 | 608 | 2,883 | 14.22% | 5 | `pin_hole_missing_pad` | **RETAINED** |
| **6** | `CS` | Conductor Scratch | 2,042 | 448 | 2,490 | 12.28% | N/A | *Extended Class 6* | **EXCLUDED (Parity)** |
| **7** | `CFO` | Conductor Foreign Object | 1,409 | 423 | 1,832 | 9.04% | N/A | *Extended Class 7* | **EXCLUDED (Parity)** |
| **8** | `BMFO` | Base Material Foreign Object | 1,334 | 346 | 1,680 | 8.29% | N/A | *Extended Class 8* | **EXCLUDED (Parity)** |
| **TOTAL**| — | — | **16,184** | **4,092** | **20,276** | **100.00%** | — | — | **14,274 Retained (70.40%)** |

---

## 3. Physical Distinction & Reassessment of Hole Breakout (`HB`)

In the baseline 6-class model, DsPCBSD+ `HB` was merged into canonical class 5 (`pin_hole_missing_pad`). This section examines the physical and manufacturing distinctions:

### A. Failure Mechanics and Manufacturing Origin

1. **Pin Hole (`deeppcb: pin-hole`):**
   - **Physical Mechanism:** Metallization void or gaseous blowout occurring during chemical/electrolytic copper plating of plated through-hole (PTH) barrels or surface lands.
   - **Manufacturing Stage:** Chemical wet-processing / electroplating bath.
   - **Reliability Risk:** High-current thermal cycling induces barrel cracking or intermittent electrical opens under thermal shock (IPC-TM-650 2.6.8).

2. **Missing Pad (`pcb_defect_v2: missing pad`):**
   - **Physical Mechanism:** Complete or near-complete absence of a copper annular pad around a via or component lead hole. Caused by photoresist detachment, over-etching, or chemical undercut.
   - **Manufacturing Stage:** Photolithography imaging, developing, or etching.
   - **Reliability Risk:** Complete loss of solder joint mechanical anchor; inability to form a solder fillet (IPC-A-610 Class 3).

3. **Hole Breakout (`dspcbsd: HB`):**
   - **Physical Mechanism:** Annular ring breakout where the drilled hole perimeter breaches the outer edge of the copper pad. Caused by mechanical CNC drill spindle runout, drill bit wander in dense laminates, or layer-to-layer optical misregistration.
   - **Manufacturing Stage:** Mechanical CNC drilling or panel alignment.
   - **Reliability Risk:** Severely reduced conductor cross-section at the pad-trace junction; catastrophic failure under IPC Class 3 rules where breakout exceeding $90^\circ$ or $180^\circ$ is unacceptable.

```
   [Pin Hole]                  [Missing Pad]               [Hole Breakout (HB)]
 +---------------+           +---------------+           +---------------+
 |   (Pad Land)  |           | (FR-4 Substrate|          |   (Pad Land)  |
 |     +---+     |           |  Only Bare FR4|           |  +---+        |
 |  O  |   |     |           |     +---+     |           |  |   | (Drill |
 |(Void)|   |     |           |     |   |     |           |  | O | Wander)|
 |     +---+     |           |     +---+     |           |  +---+        |
 +---------------+           +---------------+           +---------------+
  Plating Void in Land        Absence of Copper Pad       Drill Breaks Annular Ring
```

### B. Engineering Justification for 6-Class vs 9-Class Ontology

- **Unified 6-Class Model:** All three defects represent **annular pad / land structural integrity failures** preventing compliant component termination or via continuity. Grouping them ensures 1:1 topological parity with DeepPCB and PCB-Defect V2, enabling cross-dataset transfer learning without ungrounded synthetic heads.
- **Dedicated 9-Class Model:** In an operational factory floor with statistical process control (SPC), separating `HB` from `pin_hole` is vital: `HB` flags CNC drill spindle wear or optical registration offset, whereas `pin_hole` flags chemical bath contaminant imbalance.

---

## 4. Industrial Reality: Conductor Scratches & Foreign Objects

Contrary to academic simplifications, **Conductor Scratch (`CS`)**, **Conductor Foreign Object (`CFO`)**, and **Base Material Foreign Object (`BMFO`)** are **NOT** irrelevant in industrial electronics manufacturing.

### A. Conductor Scratch (`CS`, 2,490 instances)
- **Industrial Standards:** Under **IPC-A-600 Class 3** and **IPC-6012 Table 3-5**, scratches across conductor surfaces are strictly regulated:
  - If a scratch reduces conductor width or thickness by more than **10%** of design requirements, it is a **strict REJECT**.
  - Surface scratches that penetrate the protective solder mask expose underlying copper to galvanic oxidation and moisture ingress.
  - Under high-current power traces, a deep scratch produces localized current crowding and thermal hot spots ($I^2R$ dissipation), precipitating trace burnout.
- **AOI Strategy:** Scratches must be flagged for optical measurement. If copper thickness remains intact per design allowance, standard operating procedure (SOP) allows approved selective recoating per IPC-7711/7721 rework guidelines.

### B. Conductor Foreign Object (`CFO`, 1,832 instances)
- **Industrial Standards:** Conductive foreign material (solder splashes, copper burrs, metal wire trimmings) resting across circuit traces is classified as an **acute latent defect**.
- Under operating conditions (vibration, thermal cycling), loose metallic particulate migrates across dielectric gaps, causing intermittent short circuits, arcing, or catastrophic power rail collapse.
- **AOI Strategy:** Mandatory quarantine. Parts must be blown with filtered ionized air, solvent-wiped, and visually re-inspected.

### C. Base Material Foreign Object (`BMFO`, 1,680 instances)
- **Industrial Standards:** Debris, resin dust, or particulate resting on bare laminate substrate.
- Non-conductive particles can become entrapped beneath protective solder mask, causing solder mask blistering, micro-voiding, and moisture entrapment that accelerates Conductive Anodic Filament (CAF) growth under high bias voltage.
- **AOI Strategy:** Surface cleaning via ESD-safe solvent wipes followed by optical re-inspection.

### Summary: Physical Classification Taxonomy

```
                          ┌─────────────────────────────┐
                          │   PCB Visual Inspection     │
                          └──────────────┬──────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
  ┌──────────────┴──────────────┐                 ┌──────────────┴──────────────┐
  │  Photolithographic Etching  │                 │   Surface Contamination     │
  │     & Drill Geometry        │                 │    & Physical Abrasion      │
  │   (Core 6-Class Model)      │                 │    (Extended Classes 6-8)   │
  ├─────────────────────────────┤                 ├─────────────────────────────┤
  │ 0: Open Circuit (OP)        │                 │ 6: Conductor Scratch (CS)   │
  │ 1: Short Circuit (SH)       │                 │ 7: Cond. Foreign Obj (CFO)  │
  │ 2: Mouse Bite (MB)          │                 │ 8: Base Mat. For. Obj(BMFO) │
  │ 3: Spur (SP)                │                 └─────────────────────────────┘
  │ 4: Spurious Copper (SC)     │
  │ 5: Hole Breakout / Pad (HB) │
  └─────────────────────────────┘
```

---

## 5. Conclusion & Actionable Recommendations

1. **Core 6-Class Detection (Current):** Provides universal cross-dataset parity (DeepPCB, DsPCBSD+, PCB-Defect V2), maximizing training data volume (9,307 train images) for robust feature generalization across both monochrome scans and colour substrates.
2. **Safety Policy Alignment:** Even when classes 6–8 are not in the primary 6-class detector head, any unusual visual disturbance caused by scratches or heavy debris that causes optical anomaly or low confidence is caught by the conservative quality gate (`RULE_005A_DEGRADED_IMAGE_QUALITY`, `RULE_007_UNCERTAIN_SURFACE_ANOMALIES`) and routed to `INSPECT`. Zero false passes are permitted.
