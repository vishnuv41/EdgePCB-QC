# Offline Air-Gap Operation & Zero-Dependency Audit
**EdgePCB-QC | Bharat AI-SoC Challenge 2026–27 | PS5**

EdgePCB-QC is engineered from the ground up for strict **100% offline, air-gapped industrial deployment**. This document certifies the offline compliance of all subsystems.

---

## 1. Compliance Invariants

| Category | Offline Policy | Verification Method |
|---|---|---|
| **Vision Inference** | 100% On-Device (Hailo NPU / ARM CPU) | No cloud vision APIs, zero outbound socket calls. |
| **Reasoning Advisory** | 100% On-Device (ExecuTorch Llama 3.2 1B / Deterministic Fallback) | Local `.pte` weights loaded in system RAM; zero OpenAI/Anthropic/HuggingFace API calls. |
| **Database** | 100% Local (SQLite `database/inspections.db`) | File-based transactional storage with atomic commits. |
| **User Interface** | 100% Self-Contained Streamlit (Port 8501) | Bundled assets, zero CDN stylesheet or font requests (`test_offline_mode.py`). |
| **Telemetry & Analytics** | 100% Local Device Telemetry | No external Google Analytics, Sentry, or tracking beacons. |

---

## 2. Air-Gap Verification Tests

The offline integrity of the repository is continuously verified by automated tests in `tests/test_offline_mode.py`:

```bash
pytest tests/test_offline_mode.py -v
```

Verified checks:
1. `test_local_model_artifacts_exist`: Verifies all model checkpoints exist in local directories.
2. `test_local_decision_engine_config_exists`: Verifies rule config is locally stored.
3. `test_sqlite_database_local`: Confirms SQLite database is mounted locally.
4. `test_no_external_cdn_in_dashboard`: Scans `dashboard/app.py` for remote script URLs (`cdn.jsdelivr.net`, `cdnjs.cloudflare.com`, `fonts.googleapis.com`, `unpkg.com`), confirming zero occurrences.

---

## 3. Offline Package Management on Raspberry Pi

To install Python dependencies on an air-gapped Raspberry Pi without internet access:

### Step 1: Pre-download Wheels on an Internet-Connected Machine
```bash
pip download -r hardware/requirements_pi.txt -d edgepcb_wheels/
```

### Step 2: Transfer Wheels to the Raspberry Pi
```bash
scp -r edgepcb_wheels/ pi@<raspberry-pi-ip>:~/
```

### Step 3: Install Offline
```bash
source .venv/bin/activate
pip install --no-index --find-links=~/edgepcb_wheels -r hardware/requirements_pi.txt
```
