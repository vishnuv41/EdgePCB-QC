# Model Parity Test Plan: CPU Baseline vs. Hailo-8 NPU

**Project:** Bharat AI-SoC Challenge 2026–27 | PS5: Real-Time Edge AI PCB Defect Inspection  
**Purpose:** Formal verification protocol to ensure that compiled Hailo-8 / Hailo-8L INT8 HEF models maintain precision, recall, and bounding box fidelity compared to the FP32 PyTorch / ONNX CPU baseline.

---

## 1. Parity Verification Objectives

Quantization from Float32 to INT8 can occasionally cause rounding errors on small defect features (e.g. narrow `spur` protrusions or shallow `mouse_bite` conductor necking). This test plan sets quantitative acceptance thresholds to ensure zero compromise in industrial defect screening reliability:

1. **Detection Match Rate:** $\ge 95\%$ of all defect instances detected by the CPU baseline must also be detected by the Hailo NPU.
2. **Bounding-Box Overlap (IoU):** Mean Intersection-over-Union between matched CPU and Hailo bounding boxes must be $\ge 0.85$.
3. **Confidence Score Delta:** Mean absolute difference in confidence scores must be $< 0.05$ ($5\%$).
4. **Per-Class Defect Counts:** Class-by-class counts must match within $\pm 5\%$.
5. **Critical Defect Retention:** Zero dropped `open_circuit` or `short_circuit` detections above confidence threshold $0.40$.
6. **Inference Latency Target:** Hailo-8 inference latency must be $\le 15.0\text{ ms}$ ($\ge 65\text{ FPS}$).

---

## 2. Evaluation Datasets

Parity testing is executed exclusively on unseen evaluation test sets:

1. **DeepPCB Monochrome Test Set:**
   - Path: `datasets/processed/unified_yolo/images/test/`
   - Image Count: **`194`** images
   - Total Defect Instances: **`1,143`** annotations
   - Focus: High-resolution track discontinuities, pin holes, and mouse bites.
2. **External Color Test Set:**
   - Path: `datasets/processed/unified_yolo/images/external_color_test/`
   - Image Count: **`390`** native 640x640 crops
   - Board Isolation: Boards 68, 80, 81, and 82 (never seen in training or calibration).
   - Focus: Color variations, solder resist glare, and silkscreen interference.

---

## 3. Mathematical Evaluation Metrics

### A. Bounding-Box Intersection over Union (IoU)
For matched detection $i$ with CPU box $B_{\text{cpu}} = [x_1, y_1, x_2, y_2]$ and Hailo box $B_{\text{hailo}} = [x'_1, y'_1, x'_2, y'_2]$:
$$\text{IoU}(B_{\text{cpu}}, B_{\text{hailo}}) = \frac{\text{Area}(B_{\text{cpu}} \cap B_{\text{hailo}})}{\text{Area}(B_{\text{cpu}} \cup B_{\text{hailo}})}$$
- **Acceptance Criterion:** $\frac{1}{N} \sum_{i=1}^N \text{IoU}_i \ge 0.85$

### B. Confidence Delta
$$\Delta \text{conf}_i = |\text{conf}_{\text{cpu}, i} - \text{conf}_{\text{hailo}, i}|$$
- **Acceptance Criterion:** $\frac{1}{N} \sum_{i=1}^N \Delta \text{conf}_i < 0.05$

### C. Mean Average Precision (mAP@0.50) Degradation
$$\Delta \text{mAP} = \text{mAP}_{\text{FP32}} - \text{mAP}_{\text{INT8}}$$
- **Acceptance Criterion:** $\Delta \text{mAP} \le 0.025$ ($< 2.5\%$ mAP loss due to INT8 quantization)

---

## 4. Automated Parity Execution Script

EdgePCB-QC provides a turnkey benchmark script at [scripts/benchmark_hailo_parity.py](file:///E:/EdgePCB-QC/scripts/benchmark_hailo_parity.py).

### How to Run:
```bash
# On Raspberry Pi 5 with AI HAT+ attached and HEF in models/hailo/
python scripts/benchmark_hailo_parity.py
```

### Expected Output Report:
The script generates a persistent JSON report at `benchmarks/results/model_parity_report.json`:
```json
{
  "images_evaluated": 194,
  "total_cpu_detections": 1143,
  "total_hailo_detections": 1139,
  "matched_detections": 1132,
  "match_rate": 0.9904,
  "mean_iou": 0.9142,
  "mean_conf_diff": 0.0187,
  "cpu_mean_latency_ms": 78.4,
  "hailo_mean_latency_ms": 12.8,
  "speedup_factor": 6.12,
  "parity_passed": true
}
```

---

## 5. Escalation & Remediation Protocol

If parity checks fail during hardware evaluation:
1. **If $\text{IoU} < 0.85$ on specific defect classes:**
   - Modify `pcb_yolov8n.alls` to allocate dedicated quantization precision to the high-resolution convolution heads (`cv2.0` and `cv3.0`).
2. **If critical defects (`open_circuit`, `short_circuit`) are dropped:**
   - Increase the calibration dataset count from 256 to 512 frames (`datasets/calibration_images/`).
   - Verify that calibration frames include representative high-contrast traces.
3. **If latency $> 20\text{ ms}$:**
   - Verify that PCIe Gen 3 is enabled in Raspberry Pi 5 `/boot/firmware/config.txt`:
     ```ini
     dtparam=pciex1
     dtparam=pciex1_gen=3
     ```
