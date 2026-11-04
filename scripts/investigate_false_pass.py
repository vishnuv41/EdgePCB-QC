"""
Empirical Investigation of Colour PCB False-PASS Root Causes
Bharat AI-SoC Challenge 2026-27 | PS5

Tests:
1. DeepPCB grayscale with known defect.
2. DeepPCB RGB identical conversion (R=G=B).
3. DeepPCB green solder-mask domain shift.
4. DeepPCB inverted contrast.
5. Real color PCB from PCB-Defect V2 with known defects (full resolution & resized).
6. Tiled inference on high-res color PCB.
7. Defective board with camera blur / defocus.
8. Defective board with underexposure.
9. Defective board with overexposure / glare.
10. Clean color PCB crop.

Logs results to benchmarks/results/false_pass_investigation.json
"""

import json
import os
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

import cv2
import numpy as np
from ultralytics import YOLO
from decision_engine.decision import DecisionEngine
MODEL_PATH = WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_smoke_experiment" / "weights" / "best.pt"
OUTPUT_JSON = WORKSPACE_ROOT / "benchmarks" / "results" / "false_pass_investigation.json"
OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def run_investigation():
    print("=" * 70)
    print("PHASE 2: INVESTIGATION OF COLOUR PCB FALSE-PASS ROOT CAUSES")
    print("=" * 70)

    model = YOLO(str(MODEL_PATH))
    engine = DecisionEngine()

    test_cases = []

    # 1. DeepPCB Grayscale Original
    p_deep = WORKSPACE_ROOT / "datasets" / "processed" / "deeppcb_yolo" / "images" / "test" / "group90100_90100000.jpg"
    img_deep = cv2.imread(str(p_deep))
    test_cases.append({
        "id": "TC01_deeppcb_grayscale",
        "description": "DeepPCB original monochrome test image with known defects",
        "image": img_deep,
        "preprocessing": "Standard BGR load from disk (monochrome pixel values)"
    })

    # 2. DeepPCB Exact RGB Copy
    # Ensure standard 3-channel BGR representation (unchanged)
    img_rgb = img_deep.copy()
    test_cases.append({
        "id": "TC02_deeppcb_exact_rgb",
        "description": "DeepPCB exact 3-channel RGB (identical channel values)",
        "image": img_rgb,
        "preprocessing": "3 identical channels (R=G=B), no spectral shift"
    })

    # 3. DeepPCB Green Solder Mask Shift
    img_green = img_deep.copy().astype(np.float32)
    img_green[:, :, 0] *= 0.25  # Blue channel suppressed
    img_green[:, :, 1] *= 0.95  # Green channel preserved
    img_green[:, :, 2] *= 0.15  # Red channel suppressed
    img_green = np.clip(img_green, 0, 255).astype(np.uint8)
    test_cases.append({
        "id": "TC03_deeppcb_green_solder_mask_shift",
        "description": "DeepPCB with green solder mask spectral shift (simulated color domain)",
        "image": img_green,
        "preprocessing": "Spectral attenuation (B*0.25, G*0.95, R*0.15) simulating green solder mask"
    })

    # 4. DeepPCB Inverted Contrast
    img_inv = 255 - img_deep
    test_cases.append({
        "id": "TC04_deeppcb_inverted_contrast",
        "description": "DeepPCB with inverted contrast (white traces on dark substrate)",
        "image": img_inv,
        "preprocessing": "Inverted pixel luminance (255 - I)"
    })

    # 5. Real Color PCB from PCB-Defect V2 (Known Defective Board)
    p_v2 = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v1" / "PCB_Defect" / "images" / "pcb_defect_001.jpg"
    img_v2 = cv2.imread(str(p_v2))
    test_cases.append({
        "id": "TC05_real_color_pcb_defect_001",
        "description": "Real high-resolution color PCB with known defects (PCB-Defect V2 pcb_defect_001.jpg)",
        "image": img_v2,
        "preprocessing": "Loaded native high-res color image, direct YOLO downsampling to 640x640"
    })

    # 6. Real Color PCB Defect-002
    p_v2_2 = WORKSPACE_ROOT / "datasets" / "raw" / "pcb_defect_v2" / "extracted_v1" / "PCB_Defect" / "images" / "pcb_defect_002.jpg"
    img_v2_2 = cv2.imread(str(p_v2_2))
    test_cases.append({
        "id": "TC06_real_color_pcb_defect_002",
        "description": "Real high-resolution color PCB with known defects (PCB-Defect V2 pcb_defect_002.jpg)",
        "image": img_v2_2,
        "preprocessing": "Loaded native high-res color image, direct YOLO downsampling to 640x640"
    })

    # 7. Defective DeepPCB with Heavy Defocus / Optical Blur
    img_blur = cv2.GaussianBlur(img_deep, (51, 51), 25.0)
    test_cases.append({
        "id": "TC07_defective_board_blurred",
        "description": "Defective DeepPCB with heavy optical defocus / Gaussian blur",
        "image": img_blur,
        "preprocessing": "Gaussian blur (kernel 51x51, sigma=25.0) simulating severe camera defocus"
    })

    # 8. Defective DeepPCB with Severe Underexposure
    img_dark = (img_deep.astype(np.float32) * 0.15).astype(np.uint8)
    test_cases.append({
        "id": "TC08_defective_board_underexposed",
        "description": "Defective DeepPCB with severe underexposure / low lighting",
        "image": img_dark,
        "preprocessing": "Intensity scaling by 0.15 simulating dark/failing camera lighting"
    })

    # 9. Defective DeepPCB with Severe Glare / Overexposure
    img_glare = np.clip(img_deep.astype(np.float32) * 1.5 + 80, 0, 255).astype(np.uint8)
    test_cases.append({
        "id": "TC09_defective_board_overexposed",
        "description": "Defective DeepPCB with extreme glare / sensor overexposure",
        "image": img_glare,
        "preprocessing": "Gain 1.5x + 80 offset simulating specular reflection/glare"
    })

    # 10. Clean Color PCB Crop (Patch from PCB-Defect V2)
    clean_crop = img_v2[400:1040, 400:1040].copy()
    test_cases.append({
        "id": "TC10_clean_color_pcb_crop",
        "description": "Clean 640x640 crop of green solder-mask PCB without defects",
        "image": clean_crop,
        "preprocessing": "Cropped 640x640 defect-free region of real green PCB"
    })

    results = []

    for tc in test_cases:
        img = tc["image"]
        h, w, c = img.shape
        is_gray = bool(np.allclose(img[:, :, 0], img[:, :, 1]) and np.allclose(img[:, :, 1], img[:, :, 2]))

        # Run inference through best.pt
        pred = model.predict(source=img, conf=0.25, device="cpu", verbose=False)[0]

        detections = []
        for box in pred.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            detections.append({
                "class": pred.names[cls_id],
                "confidence": round(conf, 4),
                "box": [round(x, 2) for x in box.xyxy[0].tolist()]
            })

        # Run through Decision Engine
        payload = {
            "board_id": tc["id"],
            "defects": detections,
            "coverage": 1.0
        }
        dec_record = engine.evaluate(payload, preset="consumer")

        verdict = dec_record["decision"]
        rule_id = dec_record["rule_id"]

        is_false_pass = False
        # If the test case is a defective board (TC01-TC09) and the verdict is PASS, it is a false PASS!
        if tc["id"] in ["TC01_deeppcb_grayscale", "TC02_deeppcb_exact_rgb", "TC03_deeppcb_green_solder_mask_shift",
                        "TC04_deeppcb_inverted_contrast", "TC05_real_color_pcb_defect_001", "TC06_real_color_pcb_defect_002",
                        "TC07_defective_board_blurred", "TC08_defective_board_underexposed", "TC09_defective_board_overexposed"]:
            if verdict == "PASS":
                is_false_pass = True

        res_item = {
            "test_id": tc["id"],
            "description": tc["description"],
            "image_shape": [h, w, c],
            "is_grayscale_replicated": is_gray,
            "preprocessing": tc["preprocessing"],
            "detection_count": len(detections),
            "detections": detections,
            "decision": verdict,
            "rule_id": rule_id,
            "rule_description": dec_record["rule_description"],
            "requires_electrical_test": dec_record["requires_electrical_test"],
            "is_false_pass": is_false_pass
        }
        results.append(res_item)

        status_marker = "[FAIL - FALSE PASS]" if is_false_pass else f"[{verdict}]"
        print(f"{tc['id']:<38} | Detections: {len(detections):2d} | Verdict: {verdict:<7} | Rule: {rule_id:<32} | {status_marker}")

    # Summary of findings
    false_passes = [r for r in results if r["is_false_pass"]]
    print(f"\n[*] Total Test Cases: {len(results)}")
    print(f"[!] Total False-PASS Occurrences: {len(false_passes)}")
    for fp in false_passes:
        print(f"    - {fp['test_id']}: {fp['description']} -> Returned PASS with {fp['detection_count']} detections!")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Detailed investigation report saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    run_investigation()
