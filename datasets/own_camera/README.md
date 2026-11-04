# Own-Camera Dataset Protocol (Domain 5)
Bharat AI-SoC Challenge 2026-27 | PS5

This directory (`datasets/own_camera/`) accommodates physical custom PCB captures from edge cameras (Raspberry Pi Camera Module 3 or bench digital cameras).

## 1. Directory Structure

```text
datasets/own_camera/
├── images/          # RGB captures (.jpg, .png)
│   ├── train/       # Custom training images (tiled 640x640 or whole-board)
│   └── val/         # Custom validation images
├── labels/          # YOLO formatted bounding box annotations (.txt)
│   ├── train/       # class_id x_center y_center width height (normalized)
│   └── val/
└── README.md
```

## 2. Image Capture Protocol

To minimize false positives and maintain detection consistency across domains:

1. **Camera Alignment:**
   - Mount camera strictly perpendicular ($90^\circ \pm 2^\circ$) to the PCB surface using a rigid test stand or tripod.
   - Recommended standoff: 15–25 cm for standard Raspberry Pi Camera Module 3 (autofocus locked at target distance).

2. **Illumination:**
   - Use diffuse, indirect LED ring lighting or dual cross-polarized side lamps (color temperature 5000K–6500K).
   - Avoid direct overhead spotlights that cause specular blinding glares on tinned copper pads and solder joints.

3. **Resolution & Framing:**
   - Capture at native camera resolution ($\ge 1920 \times 1080$).
   - Framing should encompass the board perimeter with $\sim 5\%$ border margin to avoid frame-edge clipping (`RULE_005D`).
   - For high-density SMD traces, utilize the EdgePCB-QC `TiledInferenceRunner` with overlapping $640 \times 640$ patches (stride 480 px, 25% overlap).

4. **Annotation Classes:**
   Annotations must follow the unified 6-class format in `training/class_map.yaml`:
   - `0`: `open_circuit` (Critical)
   - `1`: `short_circuit` (Critical)
   - `2`: `mouse_bite` (Major)
   - `3`: `spur` (Major)
   - `4`: `spurious_copper` (Major)
   - `5`: `pin_hole_missing_pad` (Critical/Major cavity or missing pad)

## 3. Fine-Tuning Execution

When local captures are annotated and placed into `images/` and `labels/`:
```powershell
python training/finetune_own_camera.py --epochs 30 --batch 16 --lr 0.001
```
If this directory contains zero annotated images, all multi-domain evaluation scripts report `Domain 5 (own_camera): NOT RUN (Dataset empty - hardware pending)`.
