# EdgePCB-QC

Offline edge AI quality control for PCBs (Arm Bharat Challenge).
Target hardware: Raspberry Pi 5 + AI HAT+ (Hailo) + Camera Module 3. Developed on a laptop first.

Two separate pipelines:
- **Vision** (inspection/): camera -> preprocess -> detector (Hailo NPU on the Pi, CPU/GPU on the laptop)
- **Advisory** (advisory/): Llama 3.2 1B Instruct via ExecuTorch on the Arm CPU. It explains only; the rules engine decides.

Agent instructions: see `AGENTS.md` and `docs/MASTER_PROMPT.md`.

Built with Llama.
