"""
Plotting Script for Advanced Experiments Benchmark
Generates publication-quality comparison charts for EdgePCB-QC
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

WORKSPACE = Path(__file__).resolve().parent.parent
BENCHMARKS = WORKSPACE / "benchmarks" / "results"
DOCS_DIR = WORKSPACE / "docs" / "ml_report"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_charts():
    exps = [
        ("Baseline (10 Ep)", "baseline_multidomain_reference"),
        ("Exp A (FineTune 50)", "exp_a_baseline_finetune"),
        ("Exp B (8-Class Split)", "exp_b_taxonomy_split"),
        ("Exp C (Hard Negs)", "exp_c_hard_negatives"),
        ("Exp D (Balanced 1:1:1)", "exp_d_domain_balanced"),
        ("Exp E (Micro Aug)", "exp_e_small_defect_aug")
    ]
    
    names = []
    clean_fps = []
    clean_scrap = []
    wb_f1 = []
    wb_prec = []
    wb_rec = []
    val_map = [80.0, 83.4, 85.9, 86.4, 69.8, 72.1]
    
    for label, key in exps:
        names.append(label)
        p = BENCHMARKS / f"{key}_evaluation.json"
        with open(p, "r") as f:
            d = json.load(f)
        cb = d.get("clean_board_metrics", {})
        wb = d.get("whole_board_metrics", {})
        clean_fps.append(cb.get("total_false_positive_boxes", 0))
        clean_scrap.append(cb.get("false_scrap_rate", 0) * 100)
        wb_f1.append(wb.get("f1", 0) * 100)
        wb_prec.append(wb.get("precision", 0) * 100)
        wb_rec.append(wb.get("recall", 0) * 100)
        
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EdgePCB-QC: Advanced Model Experiments Comparison", fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Clean-Board False Positive Alarms (Log / Linear)
    colors = ['#7f8c8d', '#e67e22', '#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
    bars1 = axes[0, 0].bar(names, clean_fps, color=colors, edgecolor='black', alpha=0.85)
    axes[0, 0].set_title("Total False-Positive Boxes on 195 Clean Boards (Lower is Better)", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("False Positive Boxes")
    axes[0, 0].tick_params(axis='x', rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
    axes[0, 0].axhline(y=17, color='#27ae60', linestyle='--', alpha=0.7, label='Exp C Winner (17 boxes)')
    axes[0, 0].legend()

    # 2. Clean Board Scrap Rate
    bars2 = axes[0, 1].bar(names, clean_scrap, color=colors, edgecolor='black', alpha=0.85)
    axes[0, 1].set_title("Factory Scrap Overkill Rate on Clean Boards (%) (Lower is Better)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("False Rejection Rate (%)")
    axes[0, 1].tick_params(axis='x', rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    axes[0, 1].axhline(y=2.05, color='#27ae60', linestyle='--', alpha=0.7, label='Exp C Overkill (2.05%)')
    axes[0, 1].legend()

    # 3. Whole-Board Performance (Precision, Recall, F1)
    x = np.arange(len(names))
    width = 0.25
    axes[1, 0].bar(x - width, wb_prec, width, label='Precision', color='#3498db', edgecolor='black')
    axes[1, 0].bar(x, wb_rec, width, label='Recall', color='#e74c3c', edgecolor='black')
    axes[1, 0].bar(x + width, wb_f1, width, label='F1-Score', color='#2ecc71', edgecolor='black')
    axes[1, 0].set_title("Whole-Board Config-7 High-Res Inference Metrics (%)", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Percentage (%)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=25)
    axes[1, 0].legend()

    # 4. Validation mAP50 (%)
    bars4 = axes[1, 1].bar(names, val_map, color=colors, edgecolor='black', alpha=0.85)
    axes[1, 1].set_title("Validation mAP@0.50 Across Controlled Experiments (%)", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("mAP50 (%)")
    axes[1, 1].set_ylim(60, 95)
    axes[1, 1].tick_params(axis='x', rotation=25)
    for bar in bars4:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    axes[1, 1].axhline(y=86.4, color='#27ae60', linestyle='--', alpha=0.7, label='Exp C Peak (86.4%)')
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = DOCS_DIR / "advanced_experiments_comparison.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved comparison plot to {out_path}")

if __name__ == "__main__":
    generate_charts()
