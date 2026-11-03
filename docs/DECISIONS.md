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
- **Decision:** Mandated group-level splitting in `training/prepare_data.py`. Entire board groups are partitioned into Train (e.g. 8 groups), Validation (1 group), and Unseen Test (2 groups).
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
