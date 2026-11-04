"""
Plotting Script for Clean-Board False Alarm & Via Hallucination Breakdown
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
DOCS_DIR = WORKSPACE / "docs" / "ml_report"

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def generate_fp_charts():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Total False Positive Boxes and Scrap Rate
    exps = ["Baseline", "Exp A (FineTune)", "Exp B (Taxonomy)", "Exp C (Hard Negs)", "Exp D (Balanced)", "Exp E (Micro Aug)"]
    fp_boxes = [357, 382, 316, 17, 313, 410]
    scrap_rate = [40.51, 40.51, 37.95, 2.05, 37.44, 42.56]
    
    colors = ['#95a5a6', '#e67e22', '#3498db', '#27ae60', '#9b59b6', '#e74c3c']
    
    bars1 = axes[0].bar(exps, fp_boxes, color=colors, edgecolor='black', alpha=0.85)
    axes[0].set_title("Clean-Board False Positive Boxes (195 Boards)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("False Positive Boxes")
    axes[0].tick_params(axis='x', rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2.0, yval + 5, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
        
    # Annotate 95.2% drop on Exp C
    axes[0].annotate('-95.2% Drop\n(17 boxes)', xy=(3, 17), xytext=(2.8, 160),
                     arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
                     fontweight='bold', color='#1e8449', ha='center',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1', edgecolor='#27ae60'))

    # 2. Via Hallucinations specifically
    via_hallucinations = [246, 259, 234, 1, 252, 262]
    bars2 = axes[1].bar(exps, via_hallucinations, color=colors, edgecolor='black', alpha=0.85)
    axes[1].set_title("Via/Hole Hallucinations on Clean Boards (pin_hole_missing_pad)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Hallucinated Bounding Boxes")
    axes[1].tick_params(axis='x', rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.0, yval + 4, f"{int(yval)}", ha='center', va='bottom', fontweight='bold')
        
    axes[1].annotate('99.6% Reduction\n(Only 1 box)', xy=(3, 1), xytext=(2.8, 120),
                     arrowprops=dict(facecolor='#27ae60', shrink=0.08, width=2, headwidth=8),
                     fontweight='bold', color='#1e8449', ha='center',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1', edgecolor='#27ae60'))

    plt.tight_layout()
    out_path = DOCS_DIR / "clean_board_false_alarm_reduction.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[+] Saved clean board false alarm reduction plot to {out_path}")

if __name__ == "__main__":
    generate_fp_charts()
