"""
Unseen Bare PCB Image Evaluation Pipeline
Bharat AI-SoC Challenge 2026-27 | PS5

Inspects previously unseen bare PCB images and reports:
1. Domain validation status (colour space, contrast, histogram).
2. Image quality assessment (Laplacian sharpness, blur, clipping).
3. Optical inspection coverage and uninspected/clipped regions.
4. Localized defect manifest (bounding boxes, class, confidence, quadrant).
5. Conservative industrial decision disposition (PASS / INSPECT / REJECT).
6. Evidence-based electrical verification notices.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from decision_engine.decision import DecisionEngine
from inspection.cpu_inference import CPUInspector
from inspection.tiling import TiledInferenceRunner
from inspection.quality_assessment import assess_inspection_image
from inspection.coverage import calculate_inspection_coverage

DEFAULT_MODEL_PATH = (
    WORKSPACE_ROOT / "models" / "checkpoints" / "pcb_multidomain_experiment" / "weights" / "best.pt"
)
UNSEEN_DIR = WORKSPACE_ROOT / "datasets" / "unseen_evaluation"
OUTPUT_DIR = UNSEEN_DIR / "results"


def inspect_single_board(
    image_path: Path,
    inspector: CPUInspector,
    tiled_runner: TiledInferenceRunner,
    decision_engine: DecisionEngine,
    preset: str = "consumer",
    reference_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Runs the complete inspection pipeline on an unseen bare PCB image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Failed to read image at: {image_path}")

    h, w = img.shape[:2]

    # Optional reference image
    ref_img = None
    if reference_path and reference_path.exists():
        ref_img = cv2.imread(str(reference_path))

    # 1. Optical Coverage & Frame-Edge Clipping Analysis
    coverage_info = calculate_inspection_coverage(img, reference_image=ref_img)
    coverage_val = coverage_info["coverage"]
    clipping_warnings = coverage_info.get("clipping_warnings", 0)
    clipped_edges = coverage_info.get("clipped_edges", 0)

    # 2. Image Quality & Domain Scope Assessment
    model_meta = inspector.get_model_metadata()
    supported_domains = model_meta.get("supported_domains", ["colour_rgb_green_mask", "colour_industrial_patches", "monochrome_high_contrast"])
    quality_assessment = assess_inspection_image(
        img,
        coverage=coverage_val,
        min_coverage_required=0.90,
        supported_domains=supported_domains,
        has_valid_reference=(ref_img is not None)
    )

    # 3. Vision Detection (Tiled inference for high-res boards, single-frame for standard sizes)
    t0 = time.perf_counter()
    if h > 640 or w > 640:
        defects = tiled_runner.inspect_tiled(img)
    else:
        defects = inspector.inspect(img)
    vision_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Decision Engine Evaluation
    payload = {
        "board_id": image_path.stem,
        "defects": defects,
        "coverage": coverage_val,
        "clipping_warnings": clipping_warnings,
        "quality_assessment": quality_assessment
    }
    decision_record = decision_engine.evaluate(payload, preset=preset)

    # 5. Render Annotated Image
    annotated = img.copy()
    for d in defects:
        bbox = d.get("bbox", [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cls_name = d.get("class", "defect")
            conf = d.get("confidence", 0.0)
            sev = d.get("severity", "major")
            color = (0, 0, 255) if sev == "critical" else ((0, 165, 255) if sev == "major" else (0, 255, 255))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, f"{cls_name} ({conf:.2f}) [{sev.upper()}]", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_img_path = OUTPUT_DIR / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(out_img_path), annotated)

    result = {
        "board_id": image_path.stem,
        "image_file": str(image_path),
        "annotated_output_image": str(out_img_path),
        "dimensions": {"width": w, "height": h},
        "domain_status": decision_record["domain_status"],
        "is_domain_validated": (decision_record["domain_status"] == "within_domain"),
        "image_quality_status": decision_record["image_quality_status"],
        "is_quality_acceptable": (decision_record["image_quality_status"] == "acceptable"),
        "optical_coverage_percentage": round(coverage_val * 100, 2),
        "clipping_warnings_count": clipping_warnings,
        "clipped_edges_count": clipped_edges,
        "uninspected_regions": coverage_info.get("clipped_edge_names", []),
        "defects_localized_count": len(defects),
        "defect_classes_found": sorted(list({d.get("class") for d in defects})),
        "decision": decision_record["decision"],
        "rule_id": decision_record["rule_id"],
        "rule_description": decision_record["rule_description"],
        "requires_manual_inspection": (decision_record["decision"] == "INSPECT"),
        "reason_for_inspect": decision_record.get("reason_for_inspect"),
        "requires_electrical_test_for_suspected_defect": decision_record["requires_electrical_test_for_suspected_defect"],
        "electrical_functionality_verified": False,
        "electrical_test_required_for_final_signoff": True,
        "vision_latency_ms": round(vision_ms, 2),
        "defects_manifest": defects
    }

    out_json_path = OUTPUT_DIR / f"{image_path.stem}_report.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def print_board_report(res: Dict[str, Any]):
    print("\n" + "=" * 70)
    print(f"INSPECTION DISPOSITION REPORT: {res['board_id']}")
    print("=" * 70)
    print(f"[*] Resolution        : {res['dimensions']['width']} x {res['dimensions']['height']}")
    print(f"[*] Domain Validation : {res['domain_status'].upper()} (Validated: {res['is_domain_validated']})")
    print(f"[*] Image Quality     : {res['image_quality_status'].upper()} (Acceptable: {res['is_quality_acceptable']})")
    print(f"[*] Optical Coverage  : {res['optical_coverage_percentage']}%")
    print(f"[*] Clipping Warnings : {res['clipping_warnings_count']} (Clipped Edges: {res['uninspected_regions']})")
    print(f"[*] Defects Found     : {res['defects_localized_count']} localized {res['defect_classes_found']}")
    print(f"[*] FINAL DECISION    : {res['decision']}")
    print(f"[*] Triggered Rule    : {res['rule_id']}")
    print(f"[*] Description       : {res['rule_description']}")
    if res['requires_manual_inspection']:
        print(f"[*] Reason for Review : {res['reason_for_inspect']}")
    print(f"[*] Electrical Status : UNVERIFIED BY OPTICAL AOI (Electrical Test Required for Signoff: True)")
    print(f"[*] Annotated Image   : {res['annotated_output_image']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unseen bare PCB images with EdgePCB-QC")
    parser.add_argument("--image", type=str, default=None, help="Path to single image or folder of images")
    parser.add_argument("--reference", type=str, default=None, help="Path to golden reference image")
    parser.add_argument("--preset", type=str, default="consumer", choices=["consumer", "industrial", "automotive"])
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.40)
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Initialize subsystems
    inspector = CPUInspector(model_path=args.model_path, conf_threshold=0.25, device=args.device)
    tiled_runner = TiledInferenceRunner(
        inspector=inspector,
        tile_size=args.tile_size,
        overlap=args.overlap,
        iou_threshold=0.30,
        edge_padding=True,
        class_agnostic_nms=False
    )
    decision_engine = DecisionEngine()

    target_path = Path(args.image) if args.image else (UNSEEN_DIR / "images")
    if not target_path.exists():
        print(f"[!] Path does not exist: {target_path}")
        return 1

    if target_path.is_file():
        image_files = [target_path]
    else:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = sorted([f for f in target_path.glob("*.*") if f.suffix.lower() in valid_exts])

    if not image_files:
        print(f"[!] No valid PCB image files found in: {target_path}")
        print(f"[!] Place test images in {UNSEEN_DIR / 'images'} and rerun.")
        return 0

    print(f"[*] Found {len(image_files)} unseen image(s) to evaluate.")
    ref_p = Path(args.reference) if args.reference else None

    for img_p in image_files:
        res = inspect_single_board(
            img_p, inspector, tiled_runner, decision_engine,
            preset=args.preset, reference_path=ref_p
        )
        print_board_report(res)

    return 0


if __name__ == "__main__":
    sys.exit(main())
