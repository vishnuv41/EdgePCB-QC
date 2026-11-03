# MASTER PROMPT: EdgePCB-QC (Arm Bharat Challenge)

You are the lead engineer on EdgePCB-QC. Work autonomously and finish as much as possible without asking me. Ask only when blocked by something only I can do (see "Human-only tasks").

## 1. Project
An offline, edge AI quality-control system for PCBs (bare PCB first, PCBA only as a small stretch).
Final target hardware: Raspberry Pi 5 8GB + AI HAT+ (Hailo NPU) + Pi Camera Module 3. I do NOT have this hardware yet, so build and validate everything on my laptop, with the hardware backend swappable later.
Laptop: Windows 11, PowerShell, NVIDIA RTX 5050 8GB (Blackwell, needs a recent CUDA 12.8+ PyTorch build), 16GB RAM, Python 3.11.
Deadline: Phase 1 submission is 23 Nov 2026. Prioritise a working, documented core over features. Do mandatory and core items first.

## 2. Hard rules from the competition
- Llama (small Instruct, start with Llama 3.2 1B Instruct) is REQUIRED for the advisory part and must run through ExecuTorch (.pte + ExecuTorch runtime). Transformers alone does NOT count.
- Two pipelines, documented and coded separately:
  a) Vision: camera -> preprocess -> detector. Hailo NPU via HailoRT on the Pi; CPU/GPU on the laptop.
  b) Advisory: Llama via ExecuTorch on the Arm CPU.
- Llama only explains and recommends. A deterministic rules engine decides PASS / INSPECT / REJECT.
- Llama receives structured JSON (class, severity, location, confidence), never raw pixels.
- No cloud dependence at runtime; everything must work with the network off.
- Original work I own. Check licences (YOLO AGPL-3.0, datasets, Llama). Add "Built with Llama".
- Deliverables: source code, working application, demo video, technical report with benchmarks.

## 3. How you must work
- Never invent facts (dataset sizes, class names, licences, API names, version compatibility). Verify from official pages or by inspecting files, and record the source.
- Keep docs/PROGRESS.md and docs/DECISIONS.md updated.
- Write tests for the decision engine, class mapping and data conversion. Run them. Do not say "done" unless it ran and worked.
- Commit often. Never commit datasets, weights, .pte, .hef, .db or secrets.
- Reproducible one-command scripts; pinned dependencies in requirements.txt.
- If a step fails, diagnose and try at least two approaches before asking me.

## 4. Tasks (in order; mandatory items early)

### T1. Environment
Verify Python 3.11, git, nvidia-smi. Create .venv. Install CUDA PyTorch that supports the RTX 5050 (verify the right build on the official PyTorch page; confirm torch.cuda.is_available() is True). Install requirements.txt. The folder layout and .gitignore already exist; git init.

### T2. Datasets (you download; I help only if blocked)
Download into datasets/raw/:
1. DsPCBSD+ (Figshare, DOI 10.6084/m9.figshare.25937167): primary training set.
2. PCB-Defect V2 (Mendeley Data, id vdj74sngvn, version 2): six-class set. If a manual browser step is needed, stop and tell me exactly which file to put where.
3. DeepPCB (https://github.com/tangsanli5201/DeepPCB): git clone. For reference/test pairs.
No PCBA/solder datasets yet.
Then:
- datasets/inspect_datasets.py: image count, sizes, annotation format, instances per class, defective vs normal, missing/corrupt/duplicate files. Output datasets/reports/dataset_summary.csv and class_distribution.png.
- Record each licence and source URL in docs/LICENSES.md, verified from the dataset pages.
- training/prepare_data.py: convert to YOLO format using training/class_map.yaml. Map only genuinely compatible classes. Priority: open circuit and short circuit.
- Split by board/layout/source, never by random image or crops of the same board. Add a test proving no board is in two splits.

### T3. Llama + ExecuTorch (mandatory; start early)
- Follow the current official ExecuTorch Llama export workflow for Llama 3.2 1B Instruct, with CPU/Arm-suitable quantization.
- Export .pte, run with the ExecuTorch runtime, measure model size, peak RAM, tokens/sec, latency. Save to benchmarks/results/.
- If it needs Linux/WSL2, set up what you can and report what blocks you.
- Needs the Hugging Face Llama licence accepted by me. If blocked, continue other tasks.
- advisory/prompt.py + schema.py (strict JSON input) + fallback.py (rule-based text if Llama is slow or output fails validation). Never change the decision, never invent a defect, say so if information is insufficient.
- Dashboard shows the instant rule-based alert first, then Llama text.

### T4. Vision model
- YOLO11n smoke run (20-30 epochs), then full run on the RTX 5050.
- Evaluate on unseen boards: precision, recall, mAP50, mAP50-95, confusion matrix, per-class recall. Report open/short recall separately.
- Compare imgsz 640 vs 1024 vs tiled inference: accuracy and latency.
- Note the Ultralytics AGPL-3.0 issue in docs/LICENSES.md and DECISIONS.md; evaluate a permissive alternative if practical.
- Export ONNX; check tensor shapes and Hailo-compatible ops.
- Public datasets come from factory scanners (domain shift). Do not over-claim: RGB images can only flag SUSPECTED opens and shorts.

### T5. Decision engine
rules.py / severity.py / decision.py driven by config.json. Inputs: type, confidence, size, grid location. Output: PASS / INSPECT / REJECT plus per-defect severity. High-confidence critical -> REJECT; medium/low confidence or unknown anomaly -> INSPECT; none -> PASS. Product-class presets. Label as prototype rules, not certified IPC criteria. Fully unit-tested with sample JSON.

### T6. Application
- VisionInspector (inspection/base.py), CPUInspector now; HailoInspector stub exists.
- SQLite via SQLAlchemy: board_id, image_path, defects JSON, decision, Llama text, model version, config version, timings, operator confirm/override, timestamp.
- Streamlit dashboard: image upload, drawn detections, decision banner, Llama advice, reference comparison, history, analytics (Pareto, heatmap), operator confirm/override, benchmark panel.

### T7. Reference comparison (after core works)
inspection/reference_compare.py: load_reference, load_test_image, align_images, compare_images, highlight_differences. Alignment, lighting normalisation, diff map, morphology, suspicious regions. Test on DeepPCB pairs. Suspicious regions -> "unknown anomaly" -> INSPECT.

### T8. Hailo preparation (no HAT needed)
Research current Hailo Dataflow Compiler / HailoRT / Model Zoo versions and OS requirements (do not assume WSL2 works). Prepare ONNX, calibration images and compile scripts for a supported x86 Linux machine. Document in docs/HAILO_SETUP.md. Write HailoInspector against the HailoRT Python API, untested but ready.

### T9. Benchmarks and report
benchmarks/benchmark.py: latency, FPS, CPU, RAM for vision (CPU/GPU now, Hailo later) and Llama (tokens/sec, time to first token, RAM). Keep docs/REPORT_DRAFT.md updated. README with setup, run instructions, mermaid architecture diagram, attribution.

### Out of scope until the core is done
PCBA models, extra datasets, PatchCore, multilingual text, PDF export.

## 5. Human-only tasks (put under NEEDS HUMAN in docs/PROGRESS.md with exact steps, then continue)
Buying/setting up hardware; Hailo developer account; Llama licence + HF token in .env; manual dataset downloads needing login/captcha; emailing organisers about the YOLO licence; photographing real boards.

## 6. Definition of done (laptop phase)
Dataset report + licences documented; leak-free splits tested; YOLO trained and evaluated per class with the 640/1024/tiled comparison; Llama via ExecuTorch (.pte) with measured numbers; decision engine tested; SQLite + Streamlit working end to end offline; ONNX exported and Hailo scripts prepared; reference comparison working on DeepPCB pairs; docs and README current.

Start with T1, then T2 and T3 in parallel where possible. Update docs/PROGRESS.md after each task.
