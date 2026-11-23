"""
ExecuTorch Llama 3.2 1B Hardware Manager for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Executes on-device Llama-family model on Raspberry Pi 5 CPU (Quad-core Cortex-A76).
Provides:
- Model (.pte) and tokenizer loading
- Edge memory safeguards (RAM preflight check to prevent OOM)
- Structured inspection prompt formatting (never raw camera pixels)
- Non-blocking advisory generation with measured TTFT & latency
- Automatic deterministic IPC expert fallback (<0.1 ms)
- Strict safety gate: LLM never overrides PASS / INSPECT / REJECT verdicts
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from advisory.executorch_runtime import ExecuTorchLlamaRunner
from advisory.schema import AdvisoryExplanation

logger = logging.getLogger("edgepcb.llm.executorch")


class ExecuTorchManager:
    """Manages the Llama 3.2 1B ExecuTorch inference pipeline on the edge CPU."""

    def __init__(
        self,
        pte_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None,
        min_free_ram_mb: float = 1200.0
    ):
        self.min_free_ram_mb = min_free_ram_mb
        self.runner = ExecuTorchLlamaRunner(
            pte_path=pte_path,
            tokenizer_path=tokenizer_path
        )

    @property
    def is_loaded(self) -> bool:
        return self.runner.is_loaded

    @property
    def pte_path(self) -> Path:
        return self.runner.pte_path

    def check_memory_headroom(self) -> bool:
        """Verifies that the edge device has sufficient free RAM to run LLM inference safely."""
        try:
            mem = psutil.virtual_memory()
            free_mb = mem.available / (1024 * 1024)
            return free_mb >= self.min_free_ram_mb
        except Exception:
            return True

    def generate_advisory(
        self,
        payload: Dict[str, Any],
        mode: str = "auto",
        max_new_tokens: int = 24
    ) -> Dict[str, Any]:
        """
        Generates structured operator guidance from the deterministic inspection findings.
        Enforces:
        - Safety Gate: Returns verdict unchanged.
        - Memory safeguard: Falls back to deterministic rule engine if free RAM < threshold.
        """
        # Memory preflight
        if not self.check_memory_headroom() and mode != "fast":
            logger.warning("Low memory detected on edge device; routing to deterministic fallback.")
            return self.runner.generate_advisory(payload, mode="fast")

        return self.runner.generate_advisory(
            payload=payload,
            mode=mode,
            max_new_tokens=max_new_tokens
        )

    def get_status_info(self) -> Dict[str, Any]:
        """Returns diagnostic metadata for dashboard hardware display."""
        pte_exists = self.runner.pte_path.exists()
        pte_size_mb = self.runner.pte_path.stat().st_size / (1024 * 1024) if pte_exists else 0.0

        return {
            "model_name": "Meta Llama 3.2 1B Instruct",
            "runtime": "ExecuTorch (.pte on ARM Cortex-A76 / x86_64 CPU)",
            "pte_path": str(self.runner.pte_path),
            "pte_file_exists": pte_exists,
            "pte_size_mb": round(pte_size_mb, 1),
            "is_loaded": self.is_loaded,
            "fallback_active": not self.is_loaded,
            "min_free_ram_mb": self.min_free_ram_mb,
            "has_memory_headroom": self.check_memory_headroom()
        }
