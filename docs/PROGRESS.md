# PROGRESS — EdgePCB-QC
*Bharat AI-SoC Challenge 2026–27 | PS5*

## IMMEDIATE HUMAN ACTIONS REQUIRED (NEEDS HUMAN)
1. **Accept Llama 3.2 1B Licence on Hugging Face:**
   - *Link:* https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - *Action:* Once Meta approves your submitted access request, generate a read-access token at https://huggingface.co/settings/tokens, and paste it into `.env` as `HF_TOKEN=hf_...`
   - *Status:* Official export pipeline `advisory/export_llama_pte.py` is ready and verified. Deterministic fallback advisory (`advisory/fallback.py`) is fully active, unit-tested, and providing instant IPC-compliant reasoning.
2. **Hardware Procurement & Registration (Phase 2):**
   - Pi 5 8GB, Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L), Camera Module 3 (+ macro lens), diffused LED lighting, 27W USB-C PSU.
   - Register at https://hailo.ai/developer-zone/ for HailoRT and Dataflow Compiler documentation.

---

## Detailed Task Status

| Task | Status | Progress / Details |
|---|---|---|
| **T1 Environment** | **DONE & VERIFIED** | Python 3.11 `.venv` configured. Git repo initialized on branch `main`. PyTorch 2.14.1 (cu126), OpenCV, ExecuTorch 1.5.1, Streamlit, Plotly, SQLAlchemy, and Scikit-learn installed. Hardware status clearly distinguished: RTX 5050 (sm_120) active at driver level, CPU fallback verified as primary execution engine (~28 FPS). |
| **T2 Datasets** | **DONE & VERIFIED** | **All 3 datasets acquired & cataloged:**<br>1. **DsPCBSD+**: 10,259 images, 20,276 bboxes, 9 classes.<br>2. **DeepPCB**: 1,500 pairs across 11 board groups, 10,013 bboxes, 6 classes.<br>3. **PCB-Defect V2**: 460 images, 1,704 bboxes (COCO) & 2,506 oriented (v2).<br>Zero-leakage group split implemented in `training/prepare_data.py`. `datasets/reports/dataset_summary.csv` and `class_distribution.png` generated. `tests/test_splits.py` passed 100%. |
| **T3 Llama + ExecuTorch** | **TESTED & READY** | ExecuTorch 1.5.1 installed with official Windows wheels. Schema contracts (`advisory/schema.py`), Llama 3 prompt engineering (`advisory/prompt.py`), and grounded fallback engine (`advisory/fallback.py`) built. `tests/test_advisory.py` passed 100%. Transparently labeled in UI/docs as deterministic fallback pending HF license token. |
| **T4 Vision model** | **DONE & VERIFIED** | **Reconciled Authoritative Evaluation on Unseen Boards (194 images, 1,143 defect instances):**<br>- Reconciled conflicting 200-image vs 194-image figures: 200-image was a pre-canonical legacy split; 194 images is the true, leak-free test split (`group90100` [74 imgs] + `group92000` [120 imgs]).<br>- Checkpoint: `models/checkpoints/pcb_smoke_experiment/weights/best.pt`<br>- Exported ONNX: `models/onnx/pcb_smoke_experiment.onnx` (12.2 MB)<br>- **Authoritative Test Metrics:** Precision **71.36%**, Recall **67.39%**, mAP50 **71.53%**, mAP50-95 **41.29%**.<br>- **Critical Classes:** Open Circuit Recall **73.40%** (Precision 57.14%, mAP50 69.31%), Short Circuit Recall **34.97%** (Precision 59.30%, mAP50 39.45%), Pin Hole / Missing Pad Recall **94.05%**.<br>- CPU Inference Speed: **35.9 ms/frame (~28 FPS)**.<br>Results logged in `benchmarks/results/eval_unseen_boards.json`. |
| **T5 Decision engine** | **DONE & VERIFIED** | Deterministic quality engine built (`severity.py`, `rules.py`, `decision.py`, `config.json`). Implements rules `RULE_001` through `RULE_008` across Consumer, Industrial, Automotive presets. Operator audit trail & override implemented. Completely independent of LLM. `tests/test_decision_engine.py` passed 100%. |
| **T6 Application** | **DONE & VERIFIED** | Streamlit industrial dashboard active at `http://localhost:8501`. Fixed deprecation warnings, added Localized Defect Manifest table with bounding box coordinates, added prominent Electrical Integrity Warning, updated hardware specifications panel, and integrated `TiledInferenceRunner` for high-resolution panels. |
| **T7 Reference comparison** | **DONE & VERIFIED** | `inspection/reference_compare.py` built using OpenCV ORB alignment, RANSAC homography, difference mapping, and morphological anomaly extraction. `tests/test_reference_compare.py` passed 100%. |
| **T8 Hailo preparation** | **DOCUMENTED** | `docs/HAILO_SETUP.md` created with full compilation pipeline (ONNX -> HAR -> Int8 PTQ -> HEF), installation commands for Raspberry Pi 5, and HailoRT Python bindings. Execution cleanly decoupled with fallback. |
| **T9 Benchmarks & Reliability** | **DONE & VERIFIED** | `scripts/test_reliability.py` executed across nominal, glare, dim, defocus, and noise conditions. Fused 36 defects across 1280x1280 multi-panel composite via Global NMS. Fail-safe quality gate verified (zero false-negative PASS verdicts). **30/30 unit & integration tests passing (100%)**. |
