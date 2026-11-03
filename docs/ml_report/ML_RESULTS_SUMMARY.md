# EdgePCB-QC — Machine Learning Results Summary
## Presentation-Ready Technical Executive Briefing
**Bharat AI-SoC Challenge 2026–27 | Problem Statement PS5**

---

### Key Technical Scorecard (Untouched Test Split Evaluation)

| Metric / Parameter | Value | Benchmark Specification & Notes |
|---|---|---|
| **Model Architecture** | **YOLOv8n (Ultralytics 8.4.201)** | 130 fused layers, 3,139,218 parameters, 8.8 GFLOPs (@640x640) |
| **Model Binary Size** | **6.20 MB** (`.pt`) / **12.19 MB** (`.onnx`) | Extremely compact footprint for edge ARM / NPU deployment |
| **Dataset Scale** | **1,500 Image Pairs (10,013 BBoxes)** | 1,120 train (7,880 bboxes), 186 val (990 bboxes), 194 test (1,143 bboxes) |
| **Data Split Methodology** | **Zero-Leakage Group Partitioning** | 11 distinct board groups partitioned at the physical board layout root |
| **Training Duration** | **3 Epochs in 12.58 Minutes** | 251.65 s/epoch on host AMD Ryzen 9 8940HX CPU |
| **Mean Average Precision (mAP@50)** | **71.53%** (`0.7153`) | Primary competition metric evaluated on unseen board groups |
| **Mean Average Precision (mAP@50-95)** | **41.29%** (`0.4129`) | COCO strict localization threshold average |
| **Macro Precision** | **71.36%** (`0.7136`) | High precision across copper surface features |
| **Macro Recall** | **67.39%** (`0.6739`) | Overall defect sensitivity across 1,143 test instances |
| **Macro F1-Score** | **69.32%** (`0.6932`) | Harmonic balance between defect detection and false alarms |
| **Inference Latency & FPS** | **27.21 ms / 34.4 FPS** (PyTorch CPU) | Real-time industrial line rate (>30 FPS); ONNX: 26.97 ms / 33.4 FPS |
| **Peak RAM Footprint** | **714.5 MB** | Lightweight memory utilization on CPU |
| **Best Performing Class** | **`spurious_copper` (93.15% AP@50)** | Precision: 91.46%, Recall: 84.34%, F1: 87.76% (high contrast on substrate) |
| **Worst Performing Class** | **`short_circuit` (39.45% AP@50)** | Recall: 34.97% (hairline copper bridging requires tiling / differencing) |

---

### Per-Class Detection Accuracy Summary (Test Set: 194 Images, 1,143 Defect Instances)

```
┌───────────────────────────┬───────────────┬────────────┬───────────┬──────────┬──────────┬───────────┐
│ Defect Class              │ Severity Tier │ Instances  │ Precision │  Recall  │ F1-Score │  mAP@50   │
├───────────────────────────┼───────────────┼────────────┼───────────┼──────────┼──────────┼───────────┤
│ spurious_copper           │ Major         │    165     │   91.46%  │  84.34%  │  87.76%  │  93.15%   │
│ pin_hole_missing_pad      │ Critical      │    185     │   46.00%  │  94.05%  │  61.78%  │  80.59%   │
│ mouse_bite                │ Major         │    179     │   83.41%  │  67.04%  │  74.33%  │  76.91%   │
│ spur                      │ Major         │    184     │   90.83%  │  50.54%  │  64.95%  │  69.78%   │
│ open_circuit              │ Critical      │    247     │   57.14%  │  73.40%  │  64.26%  │  69.31%   │
│ short_circuit             │ Critical      │    183     │   59.30%  │  34.97%  │  44.00%  │  39.45%   │
├───────────────────────────┼───────────────┼────────────┼───────────┼──────────┼──────────┼───────────┤
│ OVERALL MACRO AVERAGE     │ Multi-Tier    │   1,143    │   71.36%  │  67.39%  │  69.32%  │  71.53%   │
└───────────────────────────┴───────────────┴────────────┴───────────┴──────────┴──────────┴───────────┘
```

---

### Core Engineering Takeaways for ECE & Challenge Evaluators

1. **Zero-Leakage Integrity:**  
   Unlike standard random train/val splits that leak identical board layouts between sets, EdgePCB-QC rigorously evaluated `group90100` and `group92000` as unseen test boards. The 71.53% mAP50 represents genuine generalization to novel printed circuit topologies.
2. **Deterministic Quality Decoupling:**  
   Optical inspection is decoupled from pass/fail decisions. The vision engine detects defects in **29.07 ms**; the deterministic decision engine evaluates rules in **0.08 ms**; and the local Small Language Model (Llama 3.2 1B Instruct) provides operator reasoning without hallucinating quality verdicts.
3. **Electrical Verification Enforcement:**  
   Because hairline shorts have a 35% optical recall due to camera resolution bounds, any suspected open or short circuit automatically flags `requires_electrical_test = True`. Visual AOI screens the board, while contact test equipment (Flying-Probe / DMM) certifies electrical isolation.

---

### Deliverable File Locations
- **Complete Technical Report:** [`docs/ml_report/COMPLETE_ML_TRAINING_REPORT.md`](file:///C:/Users/VISHNU/Downloads/EdgePCB-QC/docs/ml_report/COMPLETE_ML_TRAINING_REPORT.md)
- **Training Parameters CSV:** [`docs/ml_report/training_parameters.csv`](file:///C:/Users/VISHNU/Downloads/EdgePCB-QC/docs/ml_report/training_parameters.csv)
- **Per-Class Metrics CSV:** [`docs/ml_report/per_class_metrics.csv`](file:///C:/Users/VISHNU/Downloads/EdgePCB-QC/docs/ml_report/per_class_metrics.csv)
- **Test Metrics JSON:** [`docs/ml_report/test_metrics.json`](file:///C:/Users/VISHNU/Downloads/EdgePCB-QC/docs/ml_report/test_metrics.json)
- **Computational Cost JSON:** [`benchmarks/results/vision_computational_cost.json`](file:///C:/Users/VISHNU/Downloads/EdgePCB-QC/benchmarks/results/vision_computational_cost.json)
- **19 High-Resolution Charts:** [`docs/ml_report/charts/`](file:///C:/Users/VISHNU/Downloads/EdgePCB-QC/docs/ml_report/charts/)
