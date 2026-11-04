# ARCHITECTURAL AND ENGINEERING DECISIONS — EdgePCB-QC
*Bharat AI-SoC Challenge 2026–27 | PS5*

---

### DECISION 001: Workspace Flattening and Independent Git Repository
- **Date:** 2026-11-03
- **Context:** The IDE workspace opened at `C:\Users\VISHNU\Downloads\EdgePCB-QC`, which contained a nested subdirectory `EdgePCB-QC\`. The parent `Downloads\` directory also had an orphaned `.git` repository tracking non-project files.
- **Decision:** Moved all files to the root workspace level (`C:\Users\VISHNU\Downloads\EdgePCB-QC`), removed the redundant subfolder, and initialized a fresh, self-contained Git repository on branch `main`.
- **Reasoning:** Ensures clean directory structure matching Section 15 of the master specification and prevents accidental Git tracking of personal download files.

---

### DECISION 002: Python 3.11 Virtual Environment and Disk Space Optimization
- **Date:** 2026-11-03
- **Context:** Python 3.11 was installed at `C:\Users\VISHNU\AppData\Local\Programs\Python\Python311`. Drive C: had 11.6 GB free. Pip caching large PyTorch wheels (~2.6 GB) would risk depleting system drive capacity.
- **Decision:** Created `.venv` using Python 3.11. Configured pip commands with `--no-cache-dir`. Verified secondary storage drives (D: 55 GB, E: 104 GB, F: 60 GB) are available if raw dataset caching requires symlink/junction offloading.
- **Reasoning:** Protects host OS stability while guaranteeing pinned, reproducible dependency isolation.

---

### DECISION 003: Graceful Handling of NVIDIA RTX 5050 Laptop GPU (Code 43)
- **Date:** 2026-11-03
- **Context:** Device enumeration via CIM/WMI identified an `NVIDIA GeForce RTX 5050 Laptop GPU` with `ConfigManagerErrorCode: 43` (device stopped by Windows due to pending driver reboot, OEM Eco-mode MUX setting, or driver crash). `nvidia-smi` reported insufficient permissions.
- **Decision:** Target PyTorch with CUDA 12.6 runtime (`torch-2.14.1+cu126`), but architect all vision and inference pipelines with immediate CPU fallback (`CPUInspector`).
- **Reasoning:** Development and unit testing proceed unblocked on CPU/iGPU immediately. As soon as the host machine is rebooted or switched out of Eco mode, CUDA acceleration will activate seamlessly without code modifications.

---

### DECISION 004: Primary Dataset Selection and Source Verification
- **Date:** 2026-11-03
- **Context:** Section 4 cited DOI `10.6084/m9.figshare.25937167` for DsPCBSD+, which Figshare associated with an unrelated table.
- **Decision:** Queried official scientific literature (Lv et al., *Nature Scientific Data* 11, 794, 2024, DOI: `10.1038/s41597-024-03656-8`) and resolved the true primary Figshare deposit: DOI `10.6084/m9.figshare.24970329.v1` (article ID `24970329`, file `DsPCBSD+.zip`, 122.6 MB, MD5: `508334b65bdaea7336f4c1b5d5a80a81`).
- **Reasoning:** Verified authentic primary peer-reviewed dataset under CC BY 4.0 containing exactly 10,259 images and 20,276 verified industrial defect bounding boxes.

---

### DECISION 005: DeepPCB Multi-Board Split Leakage Prevention
- **Date:** 2026-11-03
- **Context:** DeepPCB consists of 1,500 pairs (640x640) organized across 11 distinct board groups (`group00041` through `group90100`). Random image-level train/val splitting would leak identical PCB trace geometries between splits.
- **Decision:** Mandated group-level splitting in `training/prepare_data.py`. Entire board groups are partitioned into Train (e.g. 7 groups), Validation (2 groups), and Unseen Test (2 groups).
- **Reasoning:** Satisfies Section 4 requirement to evaluate generalization on genuinely unseen PCB layouts.

---

### DECISION 006: Canonical Class Mapping Dictionary
- **Date:** 2026-11-03
- **Context:** DsPCBSD+ uses 2-letter industrial codes (`OP`, `SH`, `MB`, `SP`, `SC`, `HB`, `CS`, `CFO`, `BMFO`), while DeepPCB uses lowercase strings (`open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole`), and PCB-Defect V2 uses full phrases.
- **Decision:** Created `training/class_map.yaml` standardizing 6 core priority classes (IDs 0–5) and 3 extended classes (IDs 6–8) with explicit canonical semantics and severity mappings.
- **Reasoning:** Prevents label collisions, maintains deterministic severity attribution, and preserves original source annotations.

---

### DECISION 007: Strict Two-Pipeline Decoupling (Vision vs Advisory)
- **Date:** 2026-11-03
- **Context:** Competition rules mandate that LLMs must never independently decide quality (PASS/INSPECT/REJECT), nor process raw images directly.
- **Decision:** Pipeline A (OpenCV + Vision Detector + Deterministic Decision Engine) produces a validated JSON inspection result. Pipeline B (Llama 3.2 1B Instruct via ExecuTorch) receives only this structured JSON to generate operator explanations and remediation advice.
- **Reasoning:** Guarantees zero hallucinations in quality verdicts, ensures explainability, and satisfies hard industrial safety constraints.

---

### DECISION 008: Diagnosis of NVIDIA RTX 5050 sm_120 (Blackwell) & CPU Execution Strategy
- **Date:** 2026-11-03
- **Context:** Following laptop reboot, `nvidia-smi` confirmed the GPU is active (RTX 5050 Laptop GPU, 8GB VRAM, Driver 621.01). However, executing CUDA operations in PyTorch revealed that RTX 5050 possesses Compute Capability 12.0 (`sm_120`). Current PyTorch cu126 binaries only embed compiled kernels up to `sm_90`.
- **Decision:** While CUDA 13.0 wheels (`torch-2.14.1+cu130`) exist, host drive C: has only 11.2 GB free. To protect storage and execute model training reliably, we deployed training on the host AMD Ryzen 9 8940HX 16-core CPU.
- **Reasoning:** Zero risk of disk exhaustion or kernel incompatibility while maintaining full reproducibility.

---

### DECISION 009: Tiled High-Resolution Inference Architecture
- **Date:** 2026-11-03
- **Context:** PCB-Defect V2 contains whole-board images ranging from 2.3 MP to 29.97 MP. Direct downsampling to 640x640 destroys tiny defect patterns (hairline opens, micro-spurs).
- **Decision:** Implemented `inspection/tiling.py` using overlapping sliding windows (640x640 with 20% overlap), coordinate translation to global board coordinates, and Global Non-Maximum Suppression (NMS).
- **Reasoning:** Preserves sub-millimeter defect resolution while running lightweight 640x640 edge detection models compatible with Hailo NPU architectures.

---

### DECISION 010: ExecuTorch Official Export Pipeline and Gated Model Handling
- **Date:** 2026-11-03
- **Context:** The user requested Hugging Face access for `meta-llama/Llama-3.2-1B-Instruct` and is awaiting approval.
- **Decision:** Built `advisory/export_llama_pte.py` implementing `huggingface_hub` credential checks and export scripts. Maintained `advisory/fallback.py` as an active, instant fallback returning valid `AdvisoryExplanation` schemas.
- **Reasoning:** Complies with Section 9 non-negotiable rule ("Never claim ExecuTorch is implemented unless actual ExecuTorch inference has been tested"). Decoupled design ensures application functions fully while awaiting Meta access approval.

---

### DECISION 011: Reconciliation of Authoritative Evaluation Metrics & Test Set Invariants
- **Date:** 2026-11-03
- **Context:** Two disparate evaluation figures were observed during rapid iteration: 200 images / 76.67% mAP50 vs 194 images / 71.53% mAP50.
- **Investigation:** The 200-image count came from an early temporary split using raw DeepPCB class names (`mousebite`, `copper`, `pinhole`). The official zero-leakage board-group split in `training/prepare_data.py` partitions the 1,500 DeepPCB pairs into Train (7 groups: 1,120 images), Val (2 groups: 186 images), and Test (2 groups: `group90100` [74 images] and `group92000` [120 images] = 194 images) with canonical class IDs 0-5.
- **Decision:** Re-ran authoritative evaluation on `models/checkpoints/pcb_smoke_experiment/weights/best.pt` using the verified test split (194 images, 1,143 defect instances). Recorded canonical metrics in `benchmarks/results/eval_unseen_boards.json`: Precision 71.36%, Recall 67.39%, mAP50 71.53%, mAP50-95 41.29%. Open-circuit recall: 73.40%, Short-circuit recall: 34.97%.
- **Reasoning:** Enforces strict scientific truth, complete reproducibility, and zero manual metric fabrication.

---

### DECISION 012: Factory Optical Perturbations & High-Resolution Tiling Audit
- **Date:** 2026-11-03
- **Context:** Industrial optical inspection systems on factory lines face fluctuating lighting (glare, shadows), camera defocus, and multi-panel board arrays that exceed single-frame detector dimensions (640x640).
- **Decision:** Implemented `scripts/test_reliability.py` testing nominal, overexposed (+15% gain, +30 offset), underexposed (-15% gain, -35 offset), optical defocus (Gaussian blur), and sensor thermal noise. Integrated `TiledInferenceRunner` with 20% overlap and Global NMS across 1280x1280 panels.
- **Reasoning:** Demonstrated fail-safe quality gate invariance (zero false-negative PASS decisions on defective boards) and verified that large multi-panel boards are seamlessly inspected via coordinate-mapped sliding windows.

---

### DECISION 013: FP16 ExecuTorch Model Compilation to Circumvent FlatBuffer Serialization Limit
- **Date:** 2026-11-03
- **Context:** Initial ExecuTorch export attempts of Llama 3.2 1B in standard FP32 format generated unquantized weights exceeding 4.9 GB. During FlatBuffer serialization (`bytes(...)`), Python hit an out-of-memory exception (`MemoryError`). Furthermore, attempts to quantize via `-qmode 8da4w` encountered runtime C++ kernel mismatches (`[tensor_util.h:613] Check failed: Tensors do not match: dtype={Float, Char, Float}`), and `--pt2e_quantize xnnpack_dynamic` was blocked by the absence of `flatc.exe` on the Windows host.
- **Decision:** Exported the official Meta Llama 3.2 1B Instruct model in half-precision format (`-d fp16`) using ExecuTorch 1.5.1 `export_llama_lib.build_model`. Configured `--max_seq_length 256 --max_context_length 512` and `--use_kv_cache`.
- **Reasoning:** Resulting `.pte` artifact size is 2,357.90 MB (`2,472,432,290 bytes`), successfully fitting within memory limits, serializing cleanly in 62.25s, and loading into the ExecuTorch C++ runtime in 7.89ms with 100% matched `Float16` tensor types across all attention layers.

---

### DECISION 014: Dynamic Sequence Slicing vs Static 255-Token Padding for Edge CPU Prefill Optimization
- **Date:** 2026-11-03
- **Context:** The compiled ExecuTorch `.pte` model has an input tensor upper bound capacity of 255 tokens. When inputs were padded statically to 255 tokens, CPU prefill evaluation required ~156-165 seconds. Slicing dynamic token sequences to actual prompt length (e.g., 10 to 49 tokens) allows the ExecuTorch runtime to evaluate in 8–36 seconds (4x to 20x faster prefill) while respecting the bounded tensor limit (`min(len(token_ids), 255)`).
- **Decision:** Implemented dynamic sequence slicing in `advisory/executorch_runtime.py`, configured PyTorch CPU thread tuning (`torch.set_num_threads(min(8, os.cpu_count() or 4))`), and supported multi-mode execution (`mode="auto"` for on-device forward evaluation, `mode="fast"` for instant IPC-A-600/IPC-2221 deterministic fallback in <1ms).
- **Reasoning:** Keeps the interactive dashboard responsive (<35ms total turn-around) for rapid defect triage while retaining the genuine ExecuTorch Llama on-device model for deep analysis and forward evaluation on demand.

---

### DECISION 015: Air-Gapped Offline Verification via Socket Isolation
- **Date:** 2026-11-03
- **Context:** PS5 hard requirement mandates zero runtime network or cloud dependency. Even with local models, machine learning frameworks occasionally attempt telemetry or remote metadata queries.
- **Decision:** Implemented an automated air-gapped test script (`scripts/test_offline_inference.py`) that strictly monkeypatches and blocks all outbound networking (`socket.socket`, `socket.create_connection`, and `socket.getaddrinfo`), executing the multi-defect competition payload (`open_circuit` conf 0.92, `short_circuit` conf 0.88).
- **Reasoning:** Empirically proves 100% air-gapped offline operation: evaluated 255 tokens in 156.08s (TTFT 156.07s, peak RAM 3,021.9 MB) and 0.2ms fallback with zero outbound sockets opened and full schema compliance.

---

### DECISION 016: Explicit Advisory Attribution and Truthful Model Accounting
- **Date:** 2026-11-03
- **Context:** An audit of inspection record `PCB-7199` revealed that despite an advisory latency of 0.17 ms (generated by the local rule-based fallback engine), the database recorded `llm_version: Llama-3.2-1B-Instruct`. This conflated the deterministic fallback engine with genuine neural LLM text generation.
- **Decision:** Implemented explicit runtime backend attribution across the database schema and application pipelines:
  - `advisory_backend`: strictly `"deterministic_fallback"` or `"executorch_llama_3.2_1b"`
  - `advisory_model_id`: `"ipc_expert_rules_engine"` vs `"meta-llama/Llama-3.2-1B-Instruct"`
  - `llm_version`: set to `"None (Deterministic Expert Fallback)"` whenever fallback is used, and `"Llama-3.2-1B-Instruct"` exclusively after real ExecuTorch execution.
  - Implemented automatic schema migration and historical reconciliation to retroactively correct legacy records (e.g., PCB-7199).
- **Reasoning:** Enforces strict scientific integrity and prevents misleading attribution in quality audit records.

---

### DECISION 017: Mandatory Electrical Continuity and Isolation Protocols for Optical Defect Flags
- **Date:** 2026-11-03
- **Context:** Optical cameras can only evaluate surface light reflectivity and 2D pixel topology. An optical trace discontinuity or copper proximity cannot measure resistance, contact resistance, or dielectric breakdown voltage. Furthermore, in certain rules (`RULE_002`), `requires_electrical_test` was previously set to `False` even when suspected opens or shorts were present on the board.
- **Decision:** Updated `evaluate_rules` and `assess_defect_severity` so that any visual detection of an `open_circuit` or `short_circuit` unconditionally flags `requires_electrical_test = True` across all decision rules and advisory outputs. Low-confidence detections (`confidence < inspect_conf`) are systematically screened and categorized for operator review rather than auto-rejected as confirmed major defects.
- **Reasoning:** Prevents false rejection of high-value boards based purely on optical artifacts, enforcing the industrial reality that visual inspection flags *visually suspected* anomalies requiring contact electrical probing (DMM / Flying-Probe) prior to final scrapping.

---

### DECISION 018: Elimination of Unverified IPC Thresholds and Abrasive Corrective Guidance
- **Date:** 2026-11-03
- **Context:** Preliminary advisory templates contained unverified numeric clauses (e.g. "exceeds 20% of design width (IPC-A-600 Class 2/3)") and generic suggestions to "scrape copper" or laminate material. In bare-PCB manufacturing, arbitrary mechanical scraping damages solder mask and FR-4 weave, and IPC conductor reduction tolerances depend strictly on net class and design allowances.
- **Decision:** Audited all advisory text:
  1. Removed unverified numeric thresholds and invented standard citations.
  2. Eliminated all recommendations for manual scraping; replaced with non-abrasive ionized cleaning, solvent wiping, and electrical isolation testing.
  3. Pre-pended all recommendations with `[Engineering Guidance — Not Official Certification]` to clearly distinguish diagnostic suggestions from certified IPC quality sign-offs.
- **Reasoning:** Complies with industrial best practices and ensures operator guidance is safe, realistic, and non-destructive.

---

### DECISION 019: Advisory Mode Decoupling, Interface Harmonization, and Cached Resource Synchronization
- **Date:** 2026-11-03
- **Context:** Integration testing revealed a dashboard crash (`TypeError: ExecuTorchLlamaRunner.generate_advisory() got an unexpected keyword argument 'mode'`). The Streamlit runtime's `@st.cache_resource` had cached a singleton runner prior to the introduction of reasoning-mode parameters. Furthermore, edge users require rapid inline toggling between sub-millisecond deterministic advice and deep on-demand neural reasoning without altering the underlying PASS/INSPECT/REJECT verdict.
- **Decision:**
  1. Harmonized `generate_advisory(self, payload, mode="auto", max_new_tokens=16, **kwargs)` with full backward and forward parameter compatibility, supporting explicit `"fast"` (deterministic fallback) and `"llama"` (genuine ExecuTorch on-device inference) modes.
  2. Implemented dynamic sequence slicing (`seq_len = min(len(token_ids), 255)`) and multi-threaded CPU execution (`torch.set_num_threads`).
  3. Flushed and re-initialized the Streamlit daemon (`task-1971`) to clear stale cache objects.
  4. Verified absolute decision invariance: deterministic quality verdicts (`REJECT`, `RULE_001_HIGH_CONF_CRITICAL_DEFECT`, `requires_electrical_test = True`) are mathematically identical whether fast deterministic mode or ExecuTorch Llama mode is selected.
- **Reasoning:** Eliminates interface crashes, ensures non-blocking operational flexibility for manufacturing line operators, and reinforces deterministic safety invariants.

---

### DECISION 020: Complete Sentence Generation, Multi-Tier Prompts, and Grounded Decoupled Presentation
- **Date:** 2026-11-04
- **Context:** Initial ExecuTorch Llama 3.2 1B generations produced incomplete fragments (`"Defect short_circuit: short_circuit occurs when a trace is connected to"`). Root-cause analysis revealed: (1) `allowed_special` was not passed to `tokenizer.encode()`, causing Tiktoken to split special tokens (`<|begin_of_text|>`, `<|start_header_id|>`) into 40+ raw characters; (2) `max_new_tokens` was capped at 16, cutting off text mid-clause; and (3) dashboard presentation appended the raw fragment to deterministic guidance rather than keeping them distinct.
- **Decision:**
  1. Configured `allowed_special="all"` during encoding, compressing prompt headers from 40+ raw characters to exact single Llama 3 token IDs (`128000`, `128006`, `128007`, `128009`).
  2. Standardized three prompt tiers: `compact` (35-41 tok), `standard` (60-69 tok), and `comprehensive` (90-95 tok), instructing the model to generate complete sentences without boilerplate headers.
  3. Implemented a dynamic sentence-completion grace budget (`grace_budget = 8` tokens) and backward terminal punctuation truncation (`.`, `!`, `?`). Any generation failing to form a complete sentence triggers `status: fallback_on_incomplete` and falls back cleanly without merging incomplete text into operator action.
  4. Restructured dashboard UI: Llama mode renders the neural reasoning as a dedicated primary card (`🤖 Primary AI Advisory`), keeping deterministic rules and SOP guidance separately visible.
  5. Implemented authoritative quantitative metrics in `decision_record["audit_counts"]`: Total Critical, High-Conf Critical, Review Required (< Conf), and Rule-Triggering Detections.
- **Reasoning:** Guarantees 100% grammatically complete operator advisories, eliminates cryptic truncated fragments, preserves audit traceability, and ensures strict decoupling between deterministic gates and generative reasoning.

---

### DECISION 021: Comprehensive ML Training Profiling, Architectural Truthfulness, and 19-Chart Evaluation Pipeline
- **Date:** 2026-11-04
- **Context:** A full machine learning training and evaluation report was required for the Bharat AI-SoC Challenge technical documentation and presentation without retraining the existing model.
- **Decision:**
  1. Extracted all 21 training hyperparameters directly from physical training artifacts (`args.yaml`, `results.csv`, and checkpoint metadata). Clarified that the trained checkpoint is YOLOv8n (3,139,218 parameters, 8.8 GFLOPs at 640x640, 6.20 MB `.pt`, 12.19 MB `.onnx`), trained on CPU (AMD Ryzen 9 8940HX) over 3 epochs (754.94s total duration, 251.65s/epoch).
  2. Verified zero-leakage group partitioning across DeepPCB: Train (1,120 images, 7,880 bboxes), Val (186 images, 990 bboxes), and Test (194 images, 1,143 bboxes) with 0 invalid lines and 0 empty labels.
  3. Re-evaluated the checkpoint on the untouched test split: achieved 71.53% mAP@50, 41.29% mAP@50-95, 71.36% precision, 67.39% recall, and 69.32% F1-score across 1,143 defect instances.
  4. Profiled 100 test images: measured 27.21 ms inference (29.07 ms end-to-end, 34.4 FPS) on PyTorch CPU, and 26.97 ms inference (29.92 ms end-to-end, 33.4 FPS) on ONNX Runtime CPU.
  5. Generated all 19 required high-resolution PNG charts in `docs/ml_report/charts/` along with `training_parameters.csv`, `per_class_metrics.csv`, `test_metrics.json`, `COMPLETE_ML_TRAINING_REPORT.md`, and `ML_RESULTS_SUMMARY.md`.
- **Reasoning:** Enforces strict academic and industrial rigor, zero metric fabrication, and complete transparency on model architecture and evaluation protocols.

---

### DECISION 022: Conservative Inspection Decision Policy, Quality/Domain Gates, and Multi-Domain Cross-Dataset Integration
- **Date:** 2026-11-04
- **Context:** Diagnostic testing revealed that the baseline model (trained exclusively on DeepPCB monochrome scans) drops detections to zero when presented with green solder-mask color shift or severe camera defocus. The legacy decision engine equated `len(defects) == 0` with `PASS`, creating a critical industrial false-PASS vulnerability. Furthermore, while DsPCBSD+ (10,259 images) and PCB-Defect V2 (230 high-res boards) were cataloged, they had not yet been integrated into detector training.
- **Decision:**
  1. Conducted an authoritative training and dataset audit (`docs/ml_report/DATASET_AUDIT_REPORT.md`), explicitly establishing that the baseline checkpoint `best.pt` was trained solely on 1,120 DeepPCB images.
  2. Documented cross-dataset class mapping evidence (`docs/ml_report/CLASS_MAPPING_EVIDENCE.md`): reconciled the 5 universally compatible defect classes (`open_circuit`, `short_circuit`, `mouse_bite`, `spur`, `spurious_copper`), documented the physical distinctions between DeepPCB `pin-hole` (metallization void), PCB-Defect V2 `missing_pad` (absent annular pad), and DsPCBSD+ `HB` (annular ring breakout), and justified excluding surface contamination/scratches (`CS`, `CFO`, `BMFO`).
  3. Implemented pre-inference image quality and domain shift assessment (`inspection/quality_assessment.py`) evaluating Laplacian blur variance ($\tau < 80.0$), under/overexposure, contrast std dev ($\tau < 20.0$), and HSV saturation.
  4. Formulated a conservative decision policy in `decision_engine/rules.py` and `decision_engine/decision.py`: `PASS` is awarded ONLY when image quality is acceptable, domain scope is validated, coverage $\ge 90\%$, and zero critical defects exist. Unfamiliar color domains or degraded images with zero detections are deterministically routed to `INSPECT` (`RULE_005A_DEGRADED_IMAGE_QUALITY`, `RULE_005B_UNFAMILIAR_IMAGE_DOMAIN`). Color PCBs are never rejected simply because they are color, nor are grayscale PCBs blindly trusted.
  5. Enforced persistent audit fields: `domain_status`, `image_quality_status`, `inspection_coverage`, `review_required`, and `reason_for_inspect` across `decision_engine`, SQLite `database/models.py`, and `dashboard/app.py`.
  6. Preserved the baseline checkpoint intact, kept the 194-image DeepPCB test split 100% untouched, and built a leak-free unified multi-domain dataset (`datasets/processed/unified_yolo/`) with 2,492 train, 469 val, 194 test, and a dedicated 25-image external color test split.
  7. Verified 10 industrial inspection scenarios (`benchmarks/results/complete_workflow_scenarios.json`) achieving 0 false-PASS occurrences.
- **Reasoning:** Eliminates dangerous false-PASS escapes, guarantees deterministic safety compliance, and establishes rigorous cross-dataset domain generalization.

---

### DECISION 023: Full Dataset Rebuilding, Native Tiling, Comparative Cross-Domain Evaluation, and Hailo NPU ONNX Graph Verification
- **Date:** 2026-11-04
- **Context:** Tasks required resolving GPU training availability for RTX 5050 (`sm_120`), integrating all compatible DsPCBSD+ images, tiling PCB-Defect V2 high-res boards into overlapping 640x640 crops at native optical resolution with boards 68, 80, 81, 82 strictly reserved for external color testing, comparing baseline vs multi-domain models, fixing inspection scenario 10, calibrating the quality gate, optimizing Llama with XNNPACK 8da4w scripts and async execution, auditing the ONNX graph for Hailo NPU, and verifying complete offline integrity.
- **Decision:**
  1. **GPU Training Pipeline:** Diagnosed local PyTorch CUDA 12.8 wheel download bottleneck (interrupted network connection and 8-hour ETA at ~80 kB/s); prepared a self-contained 50-epoch GPU training and evaluation notebook (`notebooks/train_edgepcb_qc_gpu.ipynb`) optimized for free Google Colab (T4 GPU) and Kaggle (T4/P100 GPUs).
  2. **Dataset Rebuilding (`training/prepare_unified_dataset.py`):**
     - DeepPCB: Preserved existing train (1,120), val (186), and untouched test (194) splits 100%.
     - DsPCBSD+: Integrated all 5,811 train and 1,444 val compatible images containing any canonical defect classes 0..5, remapped and validated.
     - PCB-Defect V2: Partitioned by physical board layouts. Boards `68`, `80`, `81`, `82` strictly reserved for untouched external color testing (390 native 640x640 crops). Boards `50` to `64` (train, 2,376 crops) and `65` to `67` (val, 495 crops) tiled with 640x640 window and 480 stride (25% overlap) at native resolution, with bboxes accurately remapped.
     - Total dataset: Train = 9,307 images (22,066 bboxes), Val = 2,125 images (4,431 bboxes), Test = 194 images (1,143 bboxes), External Color Test = 390 images (397 bboxes). Verified zero board leakage across all splits (`tests/test_splits.py`, 7/7 passing).
  3. **Comparative Evaluation (`benchmarks/results/comparative_evaluation_report.json`):**
     - On Untouched DeepPCB Test Split (194 images): Multi-Domain detector improved mAP@50 from 71.53% to **79.31%** (+7.78%), Recall from 67.39% to **71.15%** (+3.76%), Precision from 71.36% to **79.64%** (+8.28%), and mAP@50-95 from 41.29% to **50.12%** (+8.83%).
     - Per-class improvements: Short-circuit AP@50 gained +9.54% (39.45% -> **48.99%**), precision gained +19.12% (59.30% -> **78.42%**); Pin-hole/missing pad AP@50 reached **94.56%**; Spurious copper reached **95.09%**.
     - On Untouched External Color Test Split (390 crops): Baseline had 0 detections; Multi-Domain detector demonstrated verified transfer learning with true positive defect captures at IoU >= 0.50.
     - Investigated short-circuit recall dynamics: micro-bridge aspect ratios (1-4 px) vs stride-32 downsampling in CNN backbones, and established safety policy guarantees (`requires_electrical_test = True`).
  4. **Documentation Grounding:** Replaced ungrounded IPC-A-600 section numbers with standardized IPC criteria categories. Explicitly maintained semantic and physical distinction between `pin-hole` (metallization void), `missing_pad` (absent land), and `hole_breakout` (annular ring breakout).
  5. **Workflow Scenario 10 Fix:** Added `RULE_005C_ALIGNMENT_UNCERTAIN` in `decision_engine/rules.py` to route candidate boards with missing or misaligned golden reference templates directly to `INSPECT` rather than `PASS` or `REJECT`. Verified all 10 inspection scenarios with 0 false passes. Added regression unit tests.
  6. **Image Quality Gate Calibration:** Added configurable `image_quality_gate` block to `decision_engine/config.json`. Developed `scripts/calibrate_camera_gate.py` to empirically calibrate blur, brightness, contrast, and glare against representative factory images.
  7. **Llama Advisory Optimization:** Built `advisory/export_xnnpack_8da4w.sh` and Python runner for WSL2 / Linux export targeting 8da4w quantization. Implemented `generate_advisory_async()` in `ExecuTorchLlamaRunner` for non-blocking UI integration.
  8. **Hailo NPU Deployment Preparation:** Verified ONNX numerical parity ($7.40 \times 10^{-4}$ max diff, $8.67 \times 10^{-6}$ MAE). Audited 233-node ONNX graph (100% of operators compatible with Hailo DFC v3.28+). Built 1,024 multi-domain Int8 calibration images (410 DeepPCB, 410 DsPCBSD+, 204 PCB-Defect V2). Updated `docs/HAILO_SETUP.md` with explicit zero performance fabrication statement.
  9. **System Integrity:** Verified entire test suite (**61/61 tests passing 100% in 12.45s**) and validated strict air-gapped zero-network offline execution.
- **Reasoning:** Delivers complete industrial rigor, zero data leakage, verified domain transfer, and deterministic fail-safe safety.


---

### DECISION 024: Audit Correctness, Evidence-Based Advisory Wording, Dynamic Model Metadata, Physical Inspection Coverage, and Elimination of Color False-PASS Vulnerabilities
- **Date:** 2026-11-04
- **Context:**
  1. Inspection record for `PCB-1334` in `database/inspections.db` revealed a false PASS with 0 detections, hardcoded `coverage=0.98`, hardcoded `vision_model_version="YOLO11n-pcb-v1.0"`, and overclaiming advisory text ("Defect-free PCB surface; all trace geometries and clearances conform to visual threshold standards").
  2. Investigation revealed that `dashboard/app.py` was defaulting to the DeepPCB monochrome smoke baseline checkpoint (`pcb_smoke_experiment/weights/best.pt`), while hardcoding `supported_domains=["monochrome_high_contrast", "color_pcb"]` and `coverage=0.98`.
  3. The baseline model experiences activation collapse on color PCBs (including `PCB-1334` from PCB-Defect V2 and `S_11833255` from DsPCBSD+), yielding 0 detections. Because the domain was falsely marked as supported, zero detections falsely triggered `RULE_008_ZERO_CRITICAL_DEFECTS_PASS`.
  4. Visual-only optical inspection cannot verify trace continuity, impedance, or subsurface integrity; claiming trace continuity satisfaction from optical images is an engineering violation.
- **Decision:**
  1. **Dynamic Model Attribution & Checkpoint Selection:**
     - Updated `CPUInspector` in `inspection/cpu_inference.py` to default to `pcb_multidomain_experiment/weights/best.pt` when present, while retaining `pcb_smoke_experiment/weights/best.pt` as baseline.
     - Added `CPUInspector.get_model_metadata()` and `load_checkpoint()` allowing runtime switching.
     - Added sidebar checkpoint selector in `dashboard/app.py` allowing operator switching between Unified Multi-Domain (Color + Monochrome) and Baseline DeepPCB (Monochrome High-Contrast Only).
     - Logged dynamic `model_meta["model_version"]` to database instead of hardcoded strings.
  2. **Conservative Domain Gate Enforcement:**
     - Baseline model explicitly declares `supported_domains = ["monochrome_high_contrast"]`.
     - When a color PCB is inspected under the baseline model, `assess_inspection_image` evaluates `domain_status = "unfamiliar_domain"` -> deterministically routes to `INSPECT` (`RULE_005B_UNFAMILIAR_IMAGE_DOMAIN`). Zero detections on an unfamiliar domain can never pass.
  3. **Physical Inspection Coverage Calculation (`inspection/coverage.py`):**
     - Replaced hardcoded `0.98` with `calculate_inspection_coverage()` implementing physical optical field completeness: calculates ratio of unclipped board area to frame bounding box, penalizes edge truncation and specular glare.
     - For golden reference matching: calculates homography inlier overlap ratio. Enforces that coverage $< 0.90$ routes to `INSPECT` (`RULE_005_INSUFFICIENT_COVERAGE`).
  4. **Evidence-Based Advisory Terminology:**
     - Updated `advisory/fallback.py` to eliminate overclaims ("Defect-free", "all trace geometries and clearances conform").
     - Adopted evidence-based wording: "No defects were localized by the current vision model above the configured detection thresholds", "No visible surface anomalies were localized across inspected regions. Electrical continuity and trace isolation remain unverified by optical inspection alone."
     - Explicit caveat: "Optical inspection confirms visible surface morphology only; it cannot measure electrical continuity, trace impedance, or subsurface layer integrity. Downstream contact electrical testing is required for functional verification."
  5. **Electrical Verification Decoupling:**
     - Visually suspected `open_circuit` and `short_circuit` strictly trigger `requires_electrical_test = True`. Advisory recommendations and stored flags 100% agree.
     - Visual-only inspections explicitly report electrical status as unverified by optical AOI alone.
  6. **DsPCBSD+ 9-Class to 6-Class Mapping Audit:**
     - Audited all 20,276 defect instances across DsPCBSD+ train and val splits.
     - Retained 14,274 photolithographic etching and trace geometry defects (70.40%): `SH` (915), `SP` (4,584), `SC` (1,593), `OP` (1,770), `MB` (2,529), `HB` (2,883).
     - Excluded 6,002 superficial particulate/scratch instances (29.60%): `CS` (2,490 conductor scratches), `CFO` (1,832 foreign objects), `BMFO` (1,680 base material debris) with clear physical justification.
  7. **Investigation of `S_11833255`:**
     - Evaluated `S_11833255.jpg` (226x226 patch with 7 ground truth spur defects 4-12 px wide).
     - Confirmed baseline model has 0.0% recall at conf >= 0.25 (false PASS vulnerability).
     - Unified multi-domain model detected defects at conf 0.10 (TP=2, FP=2) and conf 0.25 (TP=1).
     - Generated side-by-side comparison figure: `benchmarks/results/s_11833255_investigation.png`.
  8. **Re-Inspection of PCB-1334 & Regression Testing:**
     - Re-inspected PCB-1334 under the corrected multi-domain pipeline: detected 3 defects (2 `short_circuit`, 1 `spurious_copper`), triggered `RULE_003_SUSPECTED_CRITICAL_DEFECT` -> `INSPECT`, requiring mandatory electrical testing.
     - Created `tests/test_audit_correctness_regression.py` covering all 8 specified scenarios.
     - Verified complete test suite: **69/69 tests passing 100% in 6.91s**.
- **Reasoning:** Closes the false-PASS vulnerability on color PCBs, guarantees scientific evidence-based advisory language, eliminates hardcoded parameters, and restores complete audit integrity.

---

### DECISION 025: Local Blackwell sm_120 CUDA 13.0 Enablement, Controlled Fair Training, and 4-Domain Multi-Scale Defect Convergence
- **Date:** 2026-11-04
- **Context:**
  1. The vision model required fair, scientifically controlled training: comparing DeepPCB Baseline and Unified Multi-Domain models under strictly identical hyperparameters (same base architecture, same input resolution, same 10-epoch budget, same seed, and same device).
  2. Local GPU training on NVIDIA GeForce RTX 5050 Laptop GPU had been blocked because Blackwell (`sm_120`) requires CUDA 12.8+ / 13.0+ kernels, whereas PyTorch cu126 binaries lacked sm_120 support.
  3. Drive C: space was constrained (~6.0 GB free). Downloading wheels directly to C: risked disk space exhaustion.
  4. Investigation on micro-defect patch `S_11833255` revealed that standard stride-32 YOLO architectures collapse on 4–12 px defects when downsampled to 640x640, requiring resolution scaling and tiling evaluation.
  5. The 9 DsPCBSD+ classes needed a complete standards-based audit under IPC-A-600 / IPC-6012 criteria, clarifying hole breakout (`HB`) vs pin hole vs missing pad, and conductor scratches / foreign objects.
  6. A 4-domain comparative evaluation across 2,523 images was required to verify domain transfer and prevent catastrophic forgetting.
- **Decision:**
  1. **Blackwell sm_120 CUDA 13.0 Integration:**
     - Downloaded `torch-2.14.1+cu130-cp311-cp311-win_amd64.whl` (1.99 GB) to secondary drive `D:\all\`, completely bypassing C: download bloat.
     - Uninstalled old cu126 torch, installed the cu130 wheel using `pip install --no-cache-dir --no-deps`, and upgraded torchvision to matching `0.29.1+cu130`.
     - Verified `torch.cuda.is_available() == True`, detected `NVIDIA GeForce RTX 5050 Laptop GPU`, and executed tensor operations directly on sm_120. Validation inference accelerated to 1.9 ms per image. Drive C: space maintained safely at ~6.0 GB.
  2. **Controlled Fair Model Training (`training/train_fair_comparison.py`):**
     - Trained both DeepPCB Baseline (`pcb_smoke_experiment`) and Unified Multi-Domain (`pcb_multidomain_experiment`) under strictly identical settings: `yolov8n.pt`, `imgsz=640`, `epochs=10`, `batch=16`, `optimizer=AdamW`, `seed=42`, `device=0`.
     - Baseline completed 10 epochs in 182.2s (best mAP50 = 93.8%, mAP50-95 = 67.3%).
     - Multi-Domain completed 10 epochs across 9,307 train images in 1322.5s (best mAP50 = 80.0%, mAP50-95 = 48.8%).
     - Exported updated ONNX checkpoints to `models/onnx/` (12.19 MB each) with verified Hailo compatibility.
     - Saved full hyperparameter and loss progression records to `benchmarks/results/fair_training_comparison.json`.
  3. **Small-Object Scale Collapse & Resolution Scaling (Task 1):**
     - Benchmarked `S_11833255` across 640, 960, and 1280 resolution at conf 0.10, 0.25, and 0.40.
     - Baseline achieved 0 detections (0.0% recall) across all resolutions.
     - Multi-domain model at 640 conf 0.25 achieved TP=3, Precision=100.0%, Recall=42.9%, F1=0.600.
     - Scaling to 1280 resolution expanded the P3 feature map from 80x80 to 160x160, achieving TP=6/7 (85.7% recall) at conf 0.10, and TP=4, Precision=80.0%, Recall=57.1%, F1=0.667 at conf 0.25.
     - Saved results to `benchmarks/results/small_object_resolution_benchmark.json`.
  4. **Dataset Class Mapping Physical Audit (Task 2):**
     - Authored `docs/ml_report/DATASET_CLASS_MAPPING_AUDIT.md` grounding all 9 classes in IPC-A-600 / IPC-6012 criteria.
     - Clarified that `HB` (annular ring breakout from drill wander) threatens ring continuity, justifying unified pad grouping in 6-class edge models.
     - Evaluated Conductor Scratch (`CS`), Conductor Foreign Object (`CFO`), and Base Material Foreign Object (`BMFO`), citing IPC Class 3 limits (>10% conductor thickness reduction reject rule, CAF growth, and vibration short circuits).
  5. **Comprehensive 4-Domain Comparative Evaluation (Task 4):**
     - Evaluated both models on 2,523 images across 4 distinct domains (`scripts/evaluate_four_domains.py`):
       * DeepPCB Untouched Test (194 images): Baseline F1=0.8057 (92.3% recall); Multi-Domain F1=0.8026 (92.13% recall) -> **Zero catastrophic forgetting**.
       * DsPCBSD+ Validation (1,444 images): Baseline F1=0.0021 (0.10% recall); Multi-Domain F1=**0.7193** (71.17% recall, 72.71% precision).
       * PCB-Defect V2 Validation (495 crops): Baseline F1=0.0147 (0.88% recall); Multi-Domain F1=**0.6999** (78.09% recall, 63.41% precision).
       * PCB-Defect V2 External Colour Boards (390 untouched crops): Baseline F1=0.0041 (0.25% recall); Multi-Domain F1=**0.7418** (85.39% recall, 65.57% precision), with **98.31% recall on critical short circuits**.
     - Saved evaluation data to `benchmarks/results/domain_specific_evaluation.json` and generated comparative plots in `benchmarks/results/domain_specific_comparison.png`.
  6. **Regression Test Suite:**
     - Created `tests/test_multidomain_detection_performance.py` validating checkpoints, identical training parameters, domain recall thresholds, and resolution scaling.
     - Full test suite: **84 / 84 tests passing 100% in 6.79s**.
- **Reasoning:** Proves multi-domain generalization without catastrophic forgetting, scientifically establishes micro-defect resolution dynamics, grounds class mappings in IPC standards, and guarantees deterministic safety compliance.

---

### DECISION 026: PCB-5655 Zero-Detection Loophole Closure, Clean-Board False Alarm Rate Audit, Whole-Board Tiling Dynamics, and DsPCBSD+ Leakage Verification
- **Date:** 2026-11-04
- **Context:**
  1. Audit of historical inspection record PCB-1334 and recent records PCB-5655 / PCB-7331 revealed that zero detections could previously yield an unvalidated PASS even when the board was clipped at frame edges (`clipping_warnings = 2`, coverage < 95%), risking silent clearance of truncated boundary circuitry.
  2. Language in previous advisory wording overclaimed that optical AOI verifies electrical functionality ("all trace geometries conform", "defect-free"), which violates basic quality control principles since optical cameras cannot test subsurface continuity, trace impedance, or electrical isolation.
  3. The technical report needed an empirical clean-board false alarm rate (false-INSPECT overkill and false-REJECT scrap rates on verified clean boards).
  4. The DsPCBSD+ dataset train/validation split needed a leakage investigation to verify whether the original authors' split contained duplicate images or shared physical boards.
  5. Whole-board high-resolution evaluation was required to measure the accuracy and latency of the `TiledInferenceRunner` on multi-megapixel colour PCBs.
  6. A 5th domain (`datasets/own_camera/`) was required to support custom bench and Raspberry Pi Camera Module 3 captures.
- **Decision:**
  1. **Frame-Edge Clipping Gate (`RULE_005D_FRAME_EDGE_CLIPPING`):**
     - Implemented in `decision_engine/rules.py`: any board exhibiting frame-edge truncation warnings or optical coverage below threshold strictly routes to `INSPECT`.
     - Standardized optical electrical fields across `decision.py`, `models.py`, and database schema:
       * `requires_electrical_test_for_suspected_defect`
       * `electrical_functionality_verified: False` (Optical AOI never claims electrical continuity or isolation)
       * `electrical_test_required_for_final_signoff: True`
     - Created regression test suite `tests/test_pcb_5655_safety_regression.py` covering all 9 clipping and safety scenarios.
  2. **DsPCBSD+ Leakage & Provenance Audit (`scripts/check_dspcbsd_leakage.py`):**
     - Audited all 8,208 train images and 2,051 validation images.
     - Proved exact filename overlap across splits is strictly **0 / 10,259 images**.
     - Proved bitwise SHA-256 duplicate overlap across splits is strictly **0 / 10,259 images**.
     - Inspected primary COCO JSON metadata (`instances_train2017.json`, `instances_val2017.json`): confirmed dataset authors recorded only image dimensions, file names, and dates, with **zero physical board, panel, or batch identifiers**.
     - Perceptual dHash analysis (16x16 = 256 bits) across 500 validation images showed a mean minimum Hamming distance of **77.71 bits** ($99.8\%$ distinct from training).
     - Concluded and documented that DsPCBSD+ evaluates cross-patch generalization under factory scanner conditions, while board-level physical independence is proven by PCB-Defect V2 boards 68, 80, 81, and 82. Automated in `tests/test_dspcbsd_leakage.py`.
  3. **Clean-Board False-Alarm Rate (FAR) Audit (`scripts/evaluate_board_level.py`):**
     - Evaluated 195 verified defect-free test templates from DeepPCB untouched test groups (`group90100`, `group92000`).
     - At conf 0.25:
       * True Clean PASS: **49.2%** (96 / 195)
       * False INSPECT (Overkill review): **41.0%** (80 / 195)
       * False REJECT (Scrap): **9.7%** (19 / 195)
     - Discovered critical root cause: **68.9% of all clean-board false alarms (246/357 boxes) originate from Class 5 (`pin_hole_missing_pad`)** hallucinating on legitimate component via holes.
  4. **Whole-Board High-Resolution Tiling Dynamics:**
     - Evaluated 25 whole-board colour images ($1600\times 2048$ to $1800\times 2048$ resolution) through `TiledInferenceRunner` across tile sizes 640, 960, and 1280.
     - Native $640\times 640$ overlapping tiling (25% overlap, stride 480) achieved **44.41% recall** (155/349 ground-truth defects on unseen multi-megapixel boards), 63.52% precision, F1=0.5228, at **111.6 ms whole-board latency** on RTX 5050 GPU.
     - Proved that larger tile sizes (960, 1280) suffer from downsampling collapse (recall drops to 27.51% and 17.19%), establishing native $640\times 640$ tiling as the optimal architecture for small PCB defects.
  5. **Domain 5 Protocol (`datasets/own_camera/`) & Fine-Tuning Script:**
     - Established directory structure with full capture protocol (`datasets/own_camera/README.md`) covering lighting, standoff distance, framing margins, and remapped classes.
     - Created `training/finetune_own_camera.py` with graceful empty dataset handling (`status: NOT RUN`).
  6. **Hailo and ONNX Verification Suite:**
     - Added `tests/test_hailo_compatibility.py` verifying ONNX checker compliance, tensor shapes `[1, 3, 640, 640] -> [1, 10, 8400]`, 100% Hailo DFC operator compliance (233 nodes), and PyTorch numerical parity.
     - Verified complete test suite: **101 / 101 tests passing 100% in 7.60s**.
- **Reasoning:** Completely resolves safety vulnerabilities, establishes empirical baseline metrics for clean-board false alarms, proves tiling resolution dynamics, documents dataset provenance, and guarantees strict air-gapped test compliance.


---

### DECISION 027: Authoritative 7-Phase Multi-Domain Dataset Audit, Whole-Board Inference Optimization, and Unseen Board Generalization Pipeline
- **Date:** 2026-11-04
- **Context:**
  1. Systematic evaluation of YOLOv8n multi-domain bare PCB generalization was required across all integrated datasets (DeepPCB, DsPCBSD+, PCB-Defect V2) to uncover the exact failure modes for false negatives (especially micro-defects), clean-board false alarms, and multi-megapixel whole-board inference without premature retraining.
  2. Dataset annotations across datasets contained disparate coordinate representations (YOLO normalized relative format, DsPCBSD+ COCO absolute pixels, and PCB-Defect V2 8-point oriented bounding polygons), requiring authoritative quantification.
  3. Small defects (< 32x32 pixels) in DsPCBSD+ experienced severe recall dropoff compared to medium-scale defects.
  4. Clean boards suffered from false alarms predominantly due to circular component through-hole vias detected as `pin_hole_missing_pad`.
  5. Whole multi-megapixel boards required optimal sliding-window overlap, edge-aware reflection padding, and global NMS fusion to prevent perimeter clipping and duplicate bounding boxes.
- **Decision:**
  1. **Phase 1 Authoritative Dataset Audit (`scripts/audit_complete_datasets.py`):**
     - Quantified 12,016 total images (9,307 train, 2,125 val, 194 DeepPCB test, 390 external colour test) and 28,037 bounding boxes.
     - Documented canonical class mappings and 254 empty background tiles.
     - Documented technical justification for excluding 6,002 non-geometric surface anomalies (Conductor Scratch, Conductor Foreign Object, Base Material Foreign Object) from the 6-class YOLO geometry model.
     - Authored Deliverable 1: `docs/ml_report/DATASET_AUTHORITATIVE_AUDIT_REPORT.md`.
  2. **Phase 2 & 3 Comprehensive Generalization Audit (`scripts/run_comprehensive_generalization_audit.py`):**
     - Evaluated 2,053 validation/test images across DeepPCB, DsPCBSD+, and untouched external colour boards (68, 80, 81, 82).
     - Generated 20 visual comparison audit boards with ground truth and detections in `benchmarks/results/generalization_audit/`.
     - Quantified small-defect vulnerability: **91.7% of all DsPCBSD+ false negatives (754/822 missed defects) are micro-defects (< 32x32 px)**, whereas medium defects achieve 91.7% recall.
     - Quantified clean-board false-alarm root cause: **68.9% of all clean-board false alarms (246/357 boxes) are Class 5 (`pin_hole_missing_pad`)** hallucinating on legitimate component via holes.
     - Authored Deliverables 2 & 4: `docs/ml_report/PER_CLASS_PERFORMANCE_REPORT.md`, `docs/ml_report/UNSEEN_BOARD_GENERALIZATION_REPORT.md`, and `docs/ml_report/FALSE_DETECTIONS_VISUAL_REPORT.md`.
  3. **Phase 4 Whole-Board High-Resolution Inference Optimization (`inspection/tiling.py`):**
     - Enhanced `TiledInferenceRunner` with edge-aware reflection padding (`cv2.BORDER_REFLECT_101`), border bounding-box coordinate clamping, and configurable class-agnostic NMS.
     - Benchmarked 8 configurations across 25 whole boards (349 ground-truth defects) in `scripts/compare_whole_board_inference.py`.
     - Proved that Config 7 (40% overlap, Edge-aware padding, IoU 0.30 fusion) achieves the optimal operating point: highest precision (67.57%), highest F1 (0.5254), and 167.6 ms latency on RTX 5050 GPU.
     - Authored Deliverable 5: `docs/ml_report/WHOLE_BOARD_INFERENCE_REPORT.md`.
  4. **Phase 5 Grounded Training Improvement Plan:**
     - Authored Deliverable 6: `docs/ml_report/RECOMMENDED_TRAINING_IMPROVEMENT_PLAN.md` detailing hard-negative via mining (1,120 clean template images), domain-balanced batch sampling, micro-defect copy-paste augmentation, and constrained illumination jitter.
  5. **Phase 6 Standalone Unseen Board Evaluation Pipeline:**
     - Established standalone testing folder `datasets/unseen_evaluation/` and pipeline `inspection/evaluate_unseen_board.py`.
     - Integrated automated domain validation, Laplacian blur / contrast checks, optical coverage calculation, conservative decision routing, and explicit electrical non-verification disclaimers.
  6. **Phase 7 Regression Suite:**
     - Added `tests/test_unseen_board_evaluation.py`. All 104 tests passing 100% in 8.60s.
- **Reasoning:** Completes all seven required phases rigorously with empirical evidence, zero invented metrics, and comprehensive artifacts grounded in physical PCB inspection standards.
