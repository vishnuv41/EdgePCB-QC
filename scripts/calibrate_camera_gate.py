"""
Camera and Image Quality Gate Calibration Utility
EdgePCB-QC (Bharat AI-SoC Challenge 2026-27 | PS5)

Analyzes an empirical corpus of inspection images to calibrate:
- Blur threshold (Laplacian variance 5th percentile with safety margin)
- Underexposure threshold (Mean brightness minimum)
- Dynamic contrast threshold (Luminance standard deviation minimum)
- Glare tolerance & color saturation thresholds

Can update decision_engine/config.json directly.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np


def compute_frame_metrics(img: np.ndarray) -> Dict[str, float]:
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1]))
    else:
        gray = img.copy()
        sat = 0.0

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(np.mean(gray))
    std_contrast = float(np.std(gray))
    glare_pct = float(np.sum(gray >= 252)) / float(gray.size)

    return {
        "blur_score": lap_var,
        "mean_brightness": mean_bright,
        "contrast_score": std_contrast,
        "glare_ratio": glare_pct,
        "saturation_score": sat
    }


def calibrate_image_quality_gate(
    images_dir: Path,
    margin_factor: float = 0.70,
    max_images: int = 200
) -> Dict[str, Any]:
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_files = [p for p in images_dir.rglob("*") if p.suffix.lower() in valid_exts][:max_images]

    if not img_files:
        raise ValueError(f"No valid images found in {images_dir}")

    print(f"[*] Calibrating quality gate on {len(img_files)} images from: {images_dir}")

    blur_vals, bright_vals, contrast_vals, glare_vals, sat_vals = [], [], [], [], []

    for p in img_files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        m = compute_frame_metrics(img)
        blur_vals.append(m["blur_score"])
        bright_vals.append(m["mean_brightness"])
        contrast_vals.append(m["contrast_score"])
        glare_vals.append(m["glare_ratio"])
        sat_vals.append(m["saturation_score"])

    # Compute percentiles
    calibrated = {
        "sample_count": len(blur_vals),
        "blur": {
            "p5": float(np.percentile(blur_vals, 5)),
            "median": float(np.median(blur_vals)),
            "recommended_threshold": round(float(np.percentile(blur_vals, 5)) * margin_factor, 1)
        },
        "brightness": {
            "p5": float(np.percentile(bright_vals, 5)),
            "p95": float(np.percentile(bright_vals, 95)),
            "recommended_underexposure": round(max(15.0, float(np.percentile(bright_vals, 5)) * margin_factor), 1),
            "recommended_overexposure": round(min(252.0, float(np.percentile(bright_vals, 95)) * 1.05), 1)
        },
        "contrast": {
            "p5": float(np.percentile(contrast_vals, 5)),
            "median": float(np.median(contrast_vals)),
            "recommended_min_contrast": round(max(10.0, float(np.percentile(contrast_vals, 5)) * margin_factor), 1)
        },
        "glare": {
            "p95": float(np.percentile(glare_vals, 95)),
            "recommended_max_glare": round(float(np.percentile(glare_vals, 95)) * 1.5, 3)
        },
        "saturation": {
            "median": float(np.median(sat_vals)),
            "p95": float(np.percentile(sat_vals, 95))
        }
    }

    return calibrated


def main():
    parser = argparse.ArgumentParser(description="Calibrate EdgePCB-QC Camera Image Quality Gate")
    parser.add_argument("--images-dir", type=str, default="datasets/processed/deeppcb_yolo/images/val",
                        help="Directory containing representative nominal camera/inspection images")
    parser.add_argument("--update-config", action="store_true",
                        help="Update decision_engine/config.json with calibrated thresholds")
    parser.add_argument("--output", type=str, default="benchmarks/results/camera_calibration_report.json",
                        help="Path to save calibration report JSON")
    args = parser.parse_args()

    img_dir = Path(args.images_dir)
    report = calibrate_image_quality_gate(img_dir)

    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS & RECOMMENDED QUALITY THRESHOLDS")
    print("=" * 60)
    print(f"Sample images analyzed:    {report['sample_count']}")
    print(f"Blur Threshold (Laplacian): {report['blur']['recommended_threshold']} (p5: {report['blur']['p5']:.1f})")
    print(f"Min Brightness (Luminance):{report['brightness']['recommended_underexposure']} (p5: {report['brightness']['p5']:.1f})")
    print(f"Max Brightness (Luminance):{report['brightness']['recommended_overexposure']} (p95: {report['brightness']['p95']:.1f})")
    print(f"Min Contrast (StdDev):     {report['contrast']['recommended_min_contrast']} (p5: {report['contrast']['p5']:.1f})")
    print(f"Max Specular Glare Ratio:  {report['glare']['recommended_max_glare']}")
    print("=" * 60)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[+] Calibration report saved to: {out_p}")

    if args.update_config:
        cfg_path = Path("decision_engine/config.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if "image_quality_gate" not in cfg:
            cfg["image_quality_gate"] = {}

        cfg["image_quality_gate"]["blur_threshold"] = report["blur"]["recommended_threshold"]
        cfg["image_quality_gate"]["underexposure_threshold"] = report["brightness"]["recommended_underexposure"]
        cfg["image_quality_gate"]["overexposure_mean_threshold"] = report["brightness"]["recommended_overexposure"]
        cfg["image_quality_gate"]["min_contrast_threshold"] = report["contrast"]["recommended_min_contrast"]

        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[+] Successfully updated {cfg_path} with calibrated thresholds.")


if __name__ == "__main__":
    main()
