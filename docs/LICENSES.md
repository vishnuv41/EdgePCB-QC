# LICENSES AND ATTRIBUTIONS
*Verified from primary official sources on 2026-11-03.*

| Item | Component Type | Licence | Official Source URL | Redistribution / Usage Notes |
|---|---|---|---|---|
| **DsPCBSD+** | Dataset (Primary Training) | CC BY 4.0 | https://doi.org/10.6084/m9.figshare.24970329.v1 | Full commercial & academic use permitted with attribution (Lv et al., Nature Scientific Data 2024, DOI: 10.1038/s41597-024-03656-8). |
| **DeepPCB** | Dataset (Reference Pairs) | MIT License | https://github.com/tangsanli5201/DeepPCB | Permissive MIT open source licence. Includes paired template and defective PCB images. |
| **PCB-Defect V2** | Dataset (Secondary / Test) | CC BY 4.0 | https://doi.org/10.17632/vdj74sngvn.2 | Academic & commercial reuse permitted with attribution (Rashid et al., Elsevier Data in Brief 2025). |
| **Ultralytics YOLO** | Object Detection Framework | AGPL-3.0 | https://github.com/ultralytics/ultralytics | Strong copyleft licence (AGPL-3.0). Evaluated alongside permissive alternatives. Models exported to ONNX for edge deployment. |
| **Llama 3.2 1B Instruct** | LLM Advisory Model | Llama 3.2 Community License | https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct | Meta community licence for weights. Requires user acceptance of license agreement on Hugging Face. "Built with Llama" branding required. |
| **ExecuTorch** | Edge On-Device Runtime | BSD-3-Clause | https://github.com/pytorch/executorch | Permissive BSD-3-Clause licence by Meta / PyTorch Foundation for on-device inference runtime. |
| **HailoRT & DFC** | NPU Runtime & Compiler | Hailo Software License | https://hailo.ai/developer-zone/ | Free for deployment on Hailo-8/8L hardware (e.g. Raspberry Pi AI HAT+). Compilation runs on x86-64 development host. |
| **OpenCV** | Computer Vision Library | Apache 2.0 | https://opencv.org/ | Permissive Apache 2.0 licence. |
| **Streamlit** | Dashboard Framework | Apache 2.0 | https://streamlit.io/ | Permissive Apache 2.0 licence. |

---

## Llama 3.2 Attribution and Compliance Statement

### 1. License Grant and Attribution
This project incorporates and builds upon **Llama 3.2 1B Instruct**, developed and released by Meta Platforms, Inc. under the **Llama 3.2 Community License Agreement** (Release Date: September 25, 2024).

In compliance with Section 1.b of the Llama 3.2 Community License:
- **Attribution Display:** *"Built with Llama 3.2"* is prominently displayed on the EdgePCB-QC user interface, dashboard, and technical documentation.
- **Model Distribution:** The foundational Llama 3.2 model weights, exported ExecuTorch `.pte` artifacts, and associated tokenizers are not distributed through public source repositories and remain excluded from Git tracking via `.gitignore`.
- **Monthly Active User Threshold:** The deployment scale of EdgePCB-QC is strictly within educational, research, and edge industrial benchmarking limits, well below the 700 million monthly active user commercial threshold.

### 2. Acceptable Use Policy
All usage of Llama 3.2 1B Instruct within EdgePCB-QC complies with the official Meta Llama Acceptable Use Policy (`USE_POLICY.md`), specifically serving as an explainability and operator remediation advisory system for industrial defect inspection in electronics manufacturing.

