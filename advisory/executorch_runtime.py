"""
ExecuTorch Llama 3.2 1B Runtime & Advisory Manager
Bharat AI-SoC Challenge 2026-27 | PS5

Executes on-device Llama-family model using ExecuTorch (.pte) on CPU.
Includes:
- Time To First Token (TTFT) measurement
- Latency & tokens/second measurement
- Peak RAM tracking
- Automatic graceful fallback to deterministic advisory
"""

import os
import json
import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional

from .schema import AdvisoryExplanation, InspectionPayload
from .prompt import format_llama_prompt, format_concise_prompt
from .fallback import generate_fallback_advisory

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LLAMA_MODELS_DIR = WORKSPACE_ROOT / "models" / "llama"
DEFAULT_TOKENIZER_PATH = LLAMA_MODELS_DIR / "tokenizer.model"

CANDIDATE_PTE_PATHS = [
    LLAMA_MODELS_DIR / "checkpoints" / "Llama-3.2-1B-Instruct" / "llama3_2_1b_fp16.pte",
    Path(r"E:\EdgePCB-QC_cache\models\llama3_2_1b\llama3_2_1b_fp16.pte"),
    LLAMA_MODELS_DIR / "llama3_2_1b_fp16.pte",
    LLAMA_MODELS_DIR / "checkpoints" / "Llama-3.2-1B-Instruct" / "llama3_2_1b.pte",
    Path(r"E:\EdgePCB-QC_cache\models\llama3_2_1b\llama3_2_1b.pte"),
    LLAMA_MODELS_DIR / "llama3_2_1b.pte"
]


def _find_default_pte() -> Path:
    for p in CANDIDATE_PTE_PATHS:
        if p.exists():
            return p
    return CANDIDATE_PTE_PATHS[0]


class ExecuTorchLlamaRunner:
    def __init__(
        self,
        pte_path: Optional[Path] = None,
        tokenizer_path: Optional[Path] = None
    ):
        self.pte_path = pte_path or _find_default_pte()
        self.tokenizer_path = tokenizer_path or DEFAULT_TOKENIZER_PATH
        self.module = None
        self.tokenizer = None
        self.is_loaded = False
        self._load_runtime()

    def _load_runtime(self):
        """Attempts to load the ExecuTorch .pte runtime module."""
        if not self.pte_path.exists():
            self.is_loaded = False
            return

        try:
            import torch
            torch_lib = Path(torch.__file__).parent / "lib"
            if torch_lib.exists() and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(str(torch_lib))

            from executorch.extension.pybindings._C import _load_for_executorch
            self.module = _load_for_executorch(str(self.pte_path.resolve()))
            
            if self.tokenizer_path.exists():
                import pytorch_tokenizers
                self.tokenizer = pytorch_tokenizers.get_tokenizer(str(self.tokenizer_path))
            
            self.is_loaded = True
            print(f"[+] ExecuTorch runtime initialized: {self.pte_path.name}")
        except Exception as e:
            print(f"[i] ExecuTorch load deferred or not ready: {e}")
            self.is_loaded = False

    def _normalize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures standard schema fields exist in the inspection payload."""
        norm = dict(payload)
        if "defects" not in norm and "detected_defects" in norm:
            norm["defects"] = norm["detected_defects"]
        if "rule_id" not in norm:
            decision = norm.get("decision", "PASS")
            norm["rule_id"] = "RULE-CRIT-001" if decision == "REJECT" else "RULE-PASS-001"
        if "rule_description" not in norm:
            norm["rule_description"] = "Inspection anomaly assessment"
        if "timestamp" not in norm:
            norm["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return norm

    def generate_advisory(
        self,
        payload: Dict[str, Any],
        mode: str = "auto",
        max_new_tokens: int = 16,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generates an advisory explanation for the operator.
        Measures execution metrics and guarantees valid schema return.
        Supports:
        - mode='fast': Instant deterministic IPC expert fallback (<1ms).
        - mode='auto' / 'full' / 'executorch': Genuine ExecuTorch on-device .pte inference & token generation.
        """
        norm_payload = self._normalize_payload(payload)
        process = psutil.Process(os.getpid())
        ram_before = process.memory_info().rss / (1024 * 1024)
        t_start = time.perf_counter()

        # If fast mode explicitly requested or ExecuTorch runtime is not loaded, use deterministic fallback
        if mode == "fast" or not self.is_loaded:
            t0 = time.perf_counter()
            advisory_obj = generate_fallback_advisory(norm_payload)
            t_end = time.perf_counter()
            ram_after = process.memory_info().rss / (1024 * 1024)

            fallback_reason = "interactive_mode_fast" if mode == "fast" else "runtime_not_loaded"
            latency_ms = round((t_end - t0) * 1000, 2)

            return {
                "source": "deterministic_fallback",
                "backend": "deterministic_fallback",
                "model_id": "ipc_expert_rules_engine",
                "model_version": "None (Deterministic Expert Fallback)",
                "status": "active",
                "fallback_reason": fallback_reason,
                "advisory": advisory_obj.model_dump(),
                "metrics": {
                    "backend": "deterministic_fallback",
                    "model_id": "ipc_expert_rules_engine",
                    "model_version": "None (Deterministic Expert Fallback)",
                    "status": "active",
                    "fallback_reason": fallback_reason,
                    "latency_ms": latency_ms,
                    "ttft_ms": latency_ms,
                    "tokens_per_second": 0.0,
                    "peak_ram_mb": round(max(ram_before, ram_after), 1)
                }
            }

        # Genuine ExecuTorch execution path
        try:
            import torch
            # Thread tuning for CPU execution optimization
            num_threads = kwargs.get("num_threads", min(8, os.cpu_count() or 4))
            if hasattr(torch, "set_num_threads"):
                torch.set_num_threads(num_threads)

            # Select prompt tier ('compact' ~32, 'standard' ~64, 'comprehensive' ~96)
            prompt_tier = kwargs.get("prompt_tier", "standard")
            use_concise = kwargs.get("concise_prompt", True)
            if use_concise:
                prompt = format_concise_prompt(norm_payload, tier=prompt_tier)
            else:
                prompt = format_llama_prompt(norm_payload)

            # Tokenize with allowed_special='all' so Llama 3 special tokens are parsed correctly
            token_ids = self.tokenizer.encode(prompt, bos=False, eos=False, allowed_special='all') if self.tokenizer else []
            seq_len = min(len(token_ids), 255)
            eval_tokens = token_ids[:seq_len]
            input_tokens = torch.tensor([eval_tokens], dtype=torch.long)
            input_pos = torch.tensor([0], dtype=torch.long)

            # 1. Prefill pass
            t_infer_start = time.perf_counter()
            outputs = self.module.forward((input_tokens, input_pos))
            t_prefill_end = time.perf_counter()
            ttft_ms = round((t_prefill_end - t_infer_start) * 1000, 2)

            # 2. Autoregressive token generation with sentence completion
            generated_tokens = []
            curr_pos = len(eval_tokens)
            logits = outputs[0]
            max_gen = max(1, min(max_new_tokens, 64))
            grace_budget = 8  # Allow up to 8 extra tokens to cleanly close the sentence
            stop_token_ids = {128001, 128008, 128009}  # <|end_of_text|>, <|eom_id|>, <|eot_id|>

            total_token_budget = max_gen + grace_budget
            for step in range(total_token_budget):
                next_tok = torch.argmax(logits[0]).item()
                if next_tok in stop_token_ids:
                    break

                generated_tokens.append(next_tok)

                # If we passed max_gen, check if sentence boundary was reached
                if step >= max_gen:
                    if self.tokenizer:
                        latest_char = self.tokenizer.decode([next_tok])
                        if any(p in latest_char for p in (".", "!", "?", "\n")):
                            break

                in_tok = torch.tensor([[next_tok]], dtype=torch.long)
                pos = torch.tensor([curr_pos], dtype=torch.long)
                logits = self.module.forward((in_tok, pos))[0]
                curr_pos += 1

            t_gen_end = time.perf_counter()
            raw_gen_text = self.tokenizer.decode(generated_tokens) if (self.tokenizer and generated_tokens) else ""
            clean_gen_text = (
                raw_gen_text.strip()
                .replace("<|eot_id|>", "")
                .replace("<|start_header_id|>", "")
                .replace("<|end_header_id|>", "")
                .replace("<|begin_of_text|>", "")
                .replace("<|end_of_text|>", "")
                .strip()
            )

            # Verify sentence completeness and truncate dangling incomplete fragments
            last_punc = max(clean_gen_text.rfind("."), clean_gen_text.rfind("!"), clean_gen_text.rfind("?"))
            if last_punc != -1 and last_punc >= 15:
                complete_gen_text = clean_gen_text[:last_punc + 1].strip()
                is_complete = True
            elif len(clean_gen_text) >= 20 and clean_gen_text.endswith((".", "!", "?")):
                complete_gen_text = clean_gen_text
                is_complete = True
            else:
                complete_gen_text = ""
                is_complete = False

            ram_after = process.memory_info().rss / (1024 * 1024)
            total_time_s = t_gen_end - t_infer_start
            total_tokens = len(eval_tokens) + len(generated_tokens)
            tokens_per_sec = round(total_tokens / (total_time_s or 1.0), 2)
            total_latency_ms = round(total_time_s * 1000, 2)

            if is_complete and complete_gen_text:
                # Valid Llama generation: create advisory with standalone neural explanation
                advisory_obj = generate_fallback_advisory(norm_payload)
                advisory_obj.generated_text = complete_gen_text
                advisory_obj.is_complete = True
                # NOTE: Never append Llama text into operator_action; keep them cleanly separated

                return {
                    "source": "executorch_llama_3.2_1b",
                    "backend": "executorch_llama_3.2_1b",
                    "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "model_version": "Llama-3.2-1B-Instruct",
                    "status": "executed",
                    "fallback_reason": None,
                    "advisory": advisory_obj.model_dump(),
                    "metrics": {
                        "backend": "executorch_llama_3.2_1b",
                        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
                        "model_version": "Llama-3.2-1B-Instruct",
                        "status": "executed",
                        "model_artifact": self.pte_path.name,
                        "fallback_reason": None,
                        "prompt_tokens": len(eval_tokens),
                        "tokens_evaluated": len(eval_tokens),
                        "tokens_generated": len(generated_tokens),
                        "is_complete": True,
                        "latency_ms": total_latency_ms,
                        "ttft_ms": ttft_ms,
                        "tokens_per_second": tokens_per_sec,
                        "peak_ram_mb": round(max(ram_before, ram_after), 1)
                    }
                }
            else:
                # Incomplete or empty generation: activate clean deterministic fallback
                advisory_obj = generate_fallback_advisory(norm_payload)
                advisory_obj.generated_text = None
                advisory_obj.is_complete = False
                fallback_reason = "generation_incomplete_or_fragmented"

                return {
                    "source": "fallback_on_incomplete",
                    "backend": "deterministic_fallback",
                    "model_id": "ipc_expert_rules_engine",
                    "model_version": "None (Deterministic Expert Fallback)",
                    "status": "fallback_on_incomplete",
                    "fallback_reason": fallback_reason,
                    "advisory": advisory_obj.model_dump(),
                    "metrics": {
                        "backend": "deterministic_fallback",
                        "model_id": "ipc_expert_rules_engine",
                        "model_version": "None (Deterministic Expert Fallback)",
                        "status": "fallback_on_incomplete",
                        "model_artifact": self.pte_path.name,
                        "fallback_reason": fallback_reason,
                        "raw_incomplete_output": clean_gen_text,
                        "prompt_tokens": len(eval_tokens),
                        "tokens_evaluated": len(eval_tokens),
                        "tokens_generated": len(generated_tokens),
                        "is_complete": False,
                        "latency_ms": total_latency_ms,
                        "ttft_ms": ttft_ms,
                        "tokens_per_second": tokens_per_sec,
                        "peak_ram_mb": round(max(ram_before, ram_after), 1)
                    }
                }
        except Exception as e:
            # Graceful recovery on any runtime fault
            advisory_obj = generate_fallback_advisory(norm_payload)
            t_end = time.perf_counter()
            ram_after = process.memory_info().rss / (1024 * 1024)
            latency_ms = round((t_end - t_start) * 1000, 2)

            return {
                "source": "fallback_on_error",
                "backend": "deterministic_fallback",
                "model_id": "ipc_expert_rules_engine",
                "model_version": "None (Deterministic Expert Fallback)",
                "status": "fallback_on_error",
                "fallback_reason": f"Execution error: {str(e)}",
                "advisory": advisory_obj.model_dump(),
                "metrics": {
                    "backend": "deterministic_fallback",
                    "model_id": "ipc_expert_rules_engine",
                    "model_version": "None (Deterministic Expert Fallback)",
                    "status": "fallback_on_error",
                    "error_detail": str(e),
                    "fallback_reason": f"Execution error: {str(e)}",
                    "latency_ms": latency_ms,
                    "ttft_ms": latency_ms,
                    "tokens_per_second": 0.0,
                    "peak_ram_mb": round(max(ram_before, ram_after), 1)
                }
            }
