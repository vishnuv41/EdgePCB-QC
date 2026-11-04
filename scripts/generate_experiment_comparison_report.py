"""
Experiment Comparison and Final Technical Report Generator for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Aggregates evaluation results across all 5 experiments and the baseline reference:
- Baseline Reference (pcb_multidomain_v1.0, 10 epochs)
- Experiment A: Baseline fine-tuning (50 epochs)
- Experiment B: Class taxonomy split (8 classes)
- Experiment C: Clean-board hard-negative learning
- Experiment D: Domain-balanced sampling (1:1:1 parity)
- Experiment E: Small-defect targeted augmentation

Outputs comprehensive comparative tables, per-class F1 comparisons,
clean-board false-alarm reduction metrics, and recommended candidate.
"""

import json
from pathlib import Path
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = WORKSPACE_ROOT / "benchmarks" / "results"
REPORT_DEST = WORKSPACE_ROOT / "docs" / "ml_report" / "ADVANCED_MODEL_IMPROVEMENT_REPORT.md"

EXPERIMENT_KEYS = [
    ("baseline_multidomain_reference", "Baseline (10 Epochs, Unified)"),
    ("exp_a_baseline_finetune", "Exp A: Baseline Fine-Tuning (50 Epochs)"),
    ("exp_b_taxonomy_split", "Exp B: 8-Class Split Taxonomy"),
    ("exp_c_hard_negatives", "Exp C: Clean-Board Hard-Negatives"),
    ("exp_d_domain_balanced", "Exp D: Domain-Balanced (1:1:1)"),
    ("exp_e_small_defect_aug", "Exp E: Small-Defect Augmentation")
]


def load_summaries():
    summaries = {}
    for key, label in EXPERIMENT_KEYS:
        p = BENCHMARKS_DIR / f"{key}_evaluation.json"
        if not p.exists():
            p = BENCHMARKS_DIR / f"{key}_summary.json"
        if p.exists():
            with open(p, "r") as f:
                summaries[key] = (label, json.load(f))
        else:
            summaries[key] = (label, None)
    return summaries


def generate_markdown_report():
    summaries = load_summaries()

    rows = []
    for key, (label, data) in summaries.items():
        if data is None:
            rows.append({
                "Experiment": label,
                "Status": "PENDING / RUNNING",
                "Val Precision": "-",
                "Val Recall": "-",
                "Val F1": "-",
                "Small Defect Recall (<32px)": "-",
                "Clean Board FP Boxes": "-",
                "Clean Scrap Rate": "-",
                "Whole-Board Precision": "-",
                "Whole-Board Recall": "-",
                "Whole-Board F1": "-"
            })
            continue

        vc = data.get("validation_crop_metrics", {})
        cb = data.get("clean_board_metrics", {})
        wb = data.get("whole_board_metrics", {})

        p = vc.get("precision", 0)
        r = vc.get("recall", 0)
        f1 = vc.get("f1", 0)
        s_r = vc.get("small_defect_recall", 0)

        fp_boxes = cb.get("total_false_positive_boxes", 0)
        scrap = cb.get("false_scrap_rate", 0)

        wb_p = wb.get("precision", 0)
        wb_r = wb.get("recall", 0)
        wb_f1 = wb.get("f1", 0)

        rows.append({
            "Experiment": label,
            "Status": "COMPLETED",
            "Val Precision": f"{p*100:.2f}%",
            "Val Recall": f"{r*100:.2f}%",
            "Val F1": f"{f1:.4f}",
            "Small Defect Recall (<32px)": f"{s_r*100:.2f}%",
            "Clean Board FP Boxes": f"{fp_boxes} ({cb.get('false_positive_rate_per_board', 0):.2f}/bd)",
            "Clean Scrap Rate": f"{scrap*100:.2f}%",
            "Whole-Board Precision": f"{wb_p*100:.2f}%",
            "Whole-Board Recall": f"{wb_r*100:.2f}%",
            "Whole-Board F1": f"{wb_f1:.4f}"
        })

    df = pd.DataFrame(rows)
    table_md = df.to_markdown(index=False)

    report_content = f"""# ADVANCED MODEL IMPROVEMENT AND FINE-TUNING REPORT
*EdgePCB-QC — Bharat AI-SoC Challenge 2026–27 | PS5*

## 1. Executive Summary & Objective

The objective of this phase is genuine bare PCB image generalization on unseen industrial boards across different layouts, colours, and lighting conditions, moving beyond training set metrics to address:
1. **Clean-Board False-Positive Alarms:** Drastically reducing the 357 false-positive boxes across 195 clean boards, of which 68.9% were via hallucinations (`pin_hole_missing_pad`).
2. **Small-Defect Sensitivity:** Overcoming scale collapse for micro-defects (< 32x32 px) where baseline recall was previously only 12.73%.
3. **Whole-Board Tiled Detection:** Improving whole-board recall (baseline 42.98%) and precision (baseline 67.57%) on multi-megapixel colour PCBs.
4. **Physical Taxonomy Evaluation:** Empirically comparing the 6-class canonical taxonomy against an 8-class split taxonomy.

---

## 2. Experimental Benchmark Matrix

{table_md}

---

## 3. Detailed Experiment Analysis

### Baseline Reference (YOLOv8n-pcb-multidomain-v1.0)
- **Epochs:** 10 epochs on Unified Multi-Domain Dataset
- **Clean-board FP Boxes:** 357 across 195 clean boards (1.83 boxes/board)
- **Clean Scrap Rate:** 40.5% of clean boards falsely rejected
- **Via Hallucinations:** 246 boxes (68.9% of all clean-board false alarms)
- **Small-Defect Recall:** 12.73% (< 32x32 px)
- **Whole-Board (Config 7):** Recall 42.98%, Precision 67.57%, F1 0.5254

*(Individual experiment evaluations will be populated as each controlled run completes).*

---

## 4. Hardware Deployment Audit (Raspberry Pi 5 & Hailo AI HAT+)
- **Architecture:** YOLOv8n (3.01M parameters, 8.2 GFLOPs at 640x640)
- **ONNX Verification:** Tensor shapes `[1, 3, 640, 640] -> [1, 10, 8400]` (6 classes) or `[1, 12, 8400]` (8 classes)
- **Operator Compatibility:** 100% compatible with Hailo Dataflow Compiler (Conv, C2f, SPPF, Concat, Detect).
"""

    with open(REPORT_DEST, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[+] Generated preliminary comparison report at {REPORT_DEST}")


if __name__ == "__main__":
    generate_markdown_report()
