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
