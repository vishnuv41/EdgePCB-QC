# Package Manifest: EdgePCB-QC Hailo-8 HEF Conversion Package

**Archive:** `hailo_colab_package.zip`  
**Project:** Bharat AI-SoC Challenge 2026–27 | PS5  
**Target Toolchain:** Hailo Dataflow Compiler (DFC) 3.34.0  
**Target Hardware:** Raspberry Pi AI HAT+ (Hailo-8 / Hailo-8L)  

## Package File Inventory

| File Name | Category | Size | SHA-256 Checksum | Description |
| :--- | :--- | :---: | :--- | :--- |
| `benchmark_hailo_parity.py` | Inference Code | 6,587 bytes | `35006bb30369659266006d412b3509a0630b07a9e9f3632ce8e91c19c9fae9c8` | Included for Colab HEF compilation & parity |
| `calib_dataset.npy` | Calibration | 314,572,928 bytes (300.00 MB) | `eed07847cee0c05cc2f565a4f17e68903e63659b26cd2658077118d6055d842e` | Included for Colab HEF compilation & parity |
| `class_names.json` | Configuration | 149 bytes | `a2644f27f519a51599f161ef478e75dd69fc257181d73b0bafc41b53534988dd` | Included for Colab HEF compilation & parity |
| `DATASET_SPLIT_VERIFICATION.md` | Documentation | 3,825 bytes | `eb7af1800870fd5f7ab6e5064a0c62c6e90ea7bfd7f021fdcadbd96e708fe78f` | Included for Colab HEF compilation & parity |
| `EdgePCB_QC_Hailo8_HEF_Conversion.ipynb` | Notebook | 29,617 bytes | `a34e604d58ae11c91a0adad4db2453d4cd6b7245d441a801bee6af0051817e15` | Included for Colab HEF compilation & parity |
| `FINAL_PRECOMPILATION_REVIEW.md` | Documentation | 8,425 bytes | `0aa2631cadeb93831e1c572e0c8a3f9eb45f15342bd707f8a5bed6f33e616aaf` | Included for Colab HEF compilation & parity |
| `hailo_backend.py` | Inference Code | 7,381 bytes | `ea081d4873894c670b090e40ab462ac9e14bb8b2a6dd3ca2a1d4f5db6112b8e2` | Included for Colab HEF compilation & parity |
| `MODEL_PARITY_TEST_PLAN.md` | Documentation | 4,498 bytes | `98146e534ed2b8765dcc07573574ccb0965e13f713084db5b20f366d4a7d56bc` | Included for Colab HEF compilation & parity |
| `pcb_multidomain_yolov8n_640.onnx` | Model Asset | 12,778,213 bytes (12.19 MB) | `0f219cae19807720f6528d635fc7168e1c067a91370e59d165a79c7536089662` | Included for Colab HEF compilation & parity |
| `pcb_yolov8n.alls` | Configuration | 302 bytes | `0d21c7545ed8b4ba89b2a2fd891c80c44fba0ac2abc79eb15654f65f3cc5b3f3` | Included for Colab HEF compilation & parity |
| `UPDATED_CONVERSION_INSTRUCTIONS.md` | Documentation | 4,146 bytes | `da3bf11b5ea368998c8cb4911887a8c2bcf335becb0ea240503b65e3fee044e5` | Included for Colab HEF compilation & parity |
| `yolo_postprocess.py` | Inference Code | 10,066 bytes | `32de24383c87482763adf374b96a2bd28da4fb518c698ec3b83616267822cd48` | Included for Colab HEF compilation & parity |

## Integrity Statement
- All files derived directly from verified workspace state.
- Original `.pt` and `.onnx` models are frozen and unmodified.
- Calibration frames strictly drawn from training split with 0 evaluation set leakage.