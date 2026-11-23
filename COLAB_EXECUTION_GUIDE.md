# EdgePCB-QC — Google Colab Turnkey Execution Guide (Python 3.12 Isolated Environment)
**Document ID:** `COLAB_EXECUTION_GUIDE.md`  
**Target Notebook:** `notebooks/EdgePCB_QC_Hailo8_HEF_Conversion_PY312.ipynb`  
**Toolchain:** Hailo Dataflow Compiler (DFC) 3.34.0  
**Target Hardware:** Raspberry Pi 5 with AI HAT+ (Hailo-8 / Hailo-8L)  
**Date:** November 20, 2026  

---

## 1. Overview & Root Cause Analysis

Google Colab's default runtime environment recently transitioned to **Python 3.13.16**.  
However, Hailo Dataflow Compiler (DFC) 3.34.0 relies on strictly pinned dependencies:
- `numpy==1.26.4`
- `scipy==1.12.0`
- `tensorflow==2.18.0`
- `jax==0.5.3` & `jaxlib==0.5.3`
- `onnx==1.16.0` & `onnxruntime==1.18.0`

Because TensorFlow 2.18 and NumPy 1.26 do not provide wheels for Python 3.13, attempting to install DFC into Colab's default Python 3.13 kernel results in installation failure.

### The Solution:
`EdgePCB_QC_Hailo8_HEF_Conversion_PY312.ipynb` provisions a dedicated, hermetic **Python 3.12 environment** at `/content/dfc_py312_env` using **Astral `uv`**.
- The base Colab kernel remains intact.
- DFC 3.34.0 and all pinned packages are installed strictly inside the Python 3.12 venv.
- All subsequent compiler commands (`hailo parser`, `hailo optimize`, `hailo compiler`) and audit scripts run via `/content/dfc_py312_env/bin/hailo` and `/content/dfc_py312_env/bin/python`.
- No binary packages are imported into the Python 3.13 kernel.

---

## 2. Google Drive Prerequisites

Ensure your Google Drive contains the folder `EdgePCB-QC_Hailo` with the two uploaded assets:
```
My Drive / EdgePCB-QC_Hailo
├── hailo_colab_package.zip                               (23.4 MB)
└── hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl  (500.1 MB)
```

---

## 3. Launching Google Colab & Selecting Runtime

1. Open your browser and navigate to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and upload `EdgePCB_QC_Hailo8_HEF_Conversion_PY312.ipynb`.
3. **Configure Runtime:**
   - In the Colab top menu, click **Runtime** $\rightarrow$ **Change runtime type**.
   - **Hardware accelerator:** Select **T4 GPU** (recommended for accelerated quantization profiling) or **CPU** (fully supported by DFC).
   - Click **Save**.

---

## 4. Step-by-Step Cell Execution Plan (Cells 1 to 25)

Execute each cell sequentially from top to bottom:

### Step 1: Base Environment & Kernel Audit (Cell 3)
- Click Run on **Cell 3**.
- Verifies `x86_64` Linux host and logs the base Python 3.13 kernel.

### Step 2: Mount Google Drive & Auto-Extract Package (Cell 5)
- Click Run on **Cell 5**.
- Grant permission to mount Google Drive.
- The cell auto-detects `hailo_colab_package.zip` in `EdgePCB-QC_Hailo` and extracts all frozen assets (`pcb_multidomain_yolov8n_640.onnx`, `calib_dataset.npy`, etc.) directly to high-speed NVMe storage `/content/edgepcb_hailo_workspace`.

### Step 3: Provision Dedicated Python 3.12 Environment (Cell 7)
- Click Run on **Cell 7**.
- Installs `uv` and provisions `/content/dfc_py312_env` with standalone Python 3.12.
- **Expected Execution Time:** $\approx 3\text{ seconds}$.
- **Expected Output:** `[PASS] Dedicated Python Interpreter verified: Python 3.12.x`.

### Step 4: Install Hailo DFC 3.34.0 & Pinned Dependencies (Cell 9)
- Click Run on **Cell 9**.
- Installs native Linux packages (`graphviz`, `libgraphviz-dev`).
- Uses `uv pip` to install the pinned dependencies:
  `numpy==1.26.4`, `scipy==1.12.0`, `tensorflow==2.18.0`, `jax==0.5.3`, `jaxlib==0.5.3`, `onnx==1.16.0`, `onnxruntime==1.18.0`.
- Installs `hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl` into `/content/dfc_py312_env`.
- Captures the complete installation log to `/content/dfc_install.log` and mirrors it to Google Drive.
- **Expected Execution Time:** $\approx 45\text{–}60\text{ seconds}$.

### Step 5: Verify Python 3.12 Environment & DFC SDK (Cell 11)
- Click Run on **Cell 11**.
- **Gate Check (Requirement 10):** Runs an internal verification script inside Python 3.12:
  - Asserts Python version is 3.12.
  - Verifies imports of NumPy 1.26.4, SciPy 1.12.0, TensorFlow 2.18.0, JAX 0.5.3, ONNX 1.16.0, ONNX Runtime 1.18.0.
  - Imports `hailo_sdk_client` and instantiates `ClientRunner`.
  - Verifies `/content/dfc_py312_env/bin/hailo --version`.
- **Expected Output:** `[PASS] ALL DFC 3.34.0 PRE-FLIGHT CHECKS PASSED CLEANLY!`.
- If any check fails, execution immediately raises `RuntimeError`, halting before compilation.

### Step 6: Inspect Frozen ONNX Model Asset (Cell 13)
- Click Run on **Cell 13**.
- Validates SHA-256 against frozen baseline `0f219cae19807720f6528d635fc7168e1c067a91370e59d165a79c7536089662`.
- Runs `onnx.checker` and dummy inference using the Python 3.12 interpreter.

### Step 7: Load Calibration Dataset (Cell 15)
- Click Run on **Cell 15**.
- Verifies `calib_dataset.npy` (shape `[256, 640, 640, 3]`, `uint8`, range $[0, 255]$).

### Step 8: Generate Model Script (Cell 17)
- Click Run on **Cell 17**.
- Writes `pcb_yolov8n.alls` with pixel normalization parameters `[0.0, 0.0, 0.0]` / `[255.0, 255.0, 255.0]`.

### Step 9: Stage 1 — Parse ONNX to HAR (Cell 19)
- Click Run on **Cell 19**.
- Executes `/content/dfc_py312_env/bin/hailo parser onnx`.
- Attempts Strategy A (`output0`); automatically engages Strategy B (6 raw detection heads) if needed.
- Persists `pcb_multidomain_yolov8n_640.har` to Google Drive.

### Step 10: Stage 2 — INT8 Post-Training Quantization (Cell 21)
- Click Run on **Cell 21**.
- Executes `/content/dfc_py312_env/bin/hailo optimize` with `calib_dataset.npy` and `pcb_yolov8n.alls`.
- **Expected Execution Time:** $\approx 3\text{–}6\text{ minutes}$.
- Persists `pcb_multidomain_yolov8n_quantized.har` to Google Drive.

### Step 11: Stage 3 — Compile to Target HEFs (Cell 23)
- Click Run on **Cell 23**.
- Compiles binaries for both **`hailo8`** (26 TOPS) and **`hailo8l`** (13 TOPS).
- **Expected Execution Time:** $\approx 2\text{–}4\text{ minutes}$.
- Outputs: `pcb_multidomain_yolov8n_hailo8.hef` and `pcb_multidomain_yolov8n_hailo8l.hef`.

### Step 12: Export HEFs to Google Drive (Cell 25)
- Click Run on **Cell 25**.
- Copies compiled `.hef` files to Google Drive folder `EdgePCB-QC_Hailo`.

---

## 5. Artifact Checklist in Google Drive

When the notebook completes, your `EdgePCB-QC_Hailo` folder will contain:
1. `dfc_install.log` — Full audit log of the Python 3.12 DFC installation.
2. `pcb_multidomain_yolov8n_640.har` (~12.2 MB) — Parsed network graph.
3. `pcb_multidomain_yolov8n_quantized.har` (~12.2 MB) — Calibrated INT8 network graph.
4. `pcb_multidomain_yolov8n_hailo8.hef` (~3.5–4.5 MB) — **Production HEF for Raspberry Pi AI HAT+ 26 TOPS**.
5. `pcb_multidomain_yolov8n_hailo8l.hef` (~3.5–4.5 MB) — **Production HEF for Raspberry Pi AI HAT+ 13 TOPS**.

---

## 6. Raspberry Pi 5 Edge Deployment

1. Download `pcb_multidomain_yolov8n_hailo8.hef` from Google Drive to your local machine.
2. Transfer to your Raspberry Pi 5:
   ```bash
   scp pcb_multidomain_yolov8n_hailo8.hef pi@raspberrypi.local:~/EdgePCB-QC/models/hailo/pcb_multidomain_yolov8n_640.hef
   ```
3. Verify on physical hardware:
   ```bash
   cd ~/EdgePCB-QC
   source .venv/bin/activate
   python hardware/deployment/verify_hailo.py
   python scripts/benchmark_hailo_parity.py --hef models/hailo/pcb_multidomain_yolov8n_640.hef
   ```
4. Restart the system service:
   ```bash
   sudo systemctl restart edgepcb.service
   ```
