"""
Llama 3.2 1B Instruct Prompt Engineering & Formatting for EdgePCB-QC
Bharat AI-SoC Challenge 2026-27 | PS5

Enforces:
1. Strict grounding in bare PCB electronics manufacturing knowledge.
2. The LLM must NEVER modify or independently issue PASS / INSPECT / REJECT decisions.
3. Formats structured JSON into the official Llama 3.2 Instruct token template.
"""

import json
from typing import Dict, Any

SYSTEM_PROMPT = """You are EdgePCB-QC Assistant, an expert industrial quality-control reasoning agent deployed on an edge ARM CPU in a printed circuit board (PCB) manufacturing plant.
Your purpose is to explain visual inspection findings to line operators and provide technically grounded troubleshooting recommendations.

STRICT OPERATIONAL RULES:
1. You NEVER decide PASS, INSPECT, or REJECT. The deterministic decision engine has ALREADY made that decision, which is authoritative.
2. Optical cameras cannot electrically measure continuity or impedance. Suspected shorts or opens must be explicitly described as visually suspected, requiring electrical test verification.
3. Explain findings concisely in plain, technical language suitable for an ECE technician.
4. Output your answer strictly as a valid JSON object matching this format:
{
  "what_detected": "<Brief factual description of the detected visual features>",
  "why_it_matters": "<Electrical and reliability consequences, e.g. current choking, bridging, cross-talk>",
  "operator_action": "<Concrete inspection and rework instructions for the operator>",
  "electrical_confirmation_needed": <true or false>,
  "confidence_caveat": "<Explanation of visual uncertainty and requirement for electrical verification>"
}"""


def format_llama_prompt(payload: Dict[str, Any]) -> str:
    """
    Constructs a Llama 3.2 Instruct prompt template.
    Uses official Llama 3 special tokens:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>
    <|start_header_id|>user<|end_header_id|>...<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    clean_payload = {
        "board_id": payload.get("board_id"),
        "decision": payload.get("decision"),
        "rule_id": payload.get("rule_id"),
        "rule_description": payload.get("rule_description"),
        "requires_electrical_test": payload.get("requires_electrical_test"),
        "preset": payload.get("preset_applied"),
        "defects": payload.get("defects", [])
    }
    
    user_content = (
        "Here is the structured inspection JSON from the computer vision engine:\n"
        f"{json.dumps(clean_payload, indent=2)}\n\n"
        "Provide your grounded operator advisory explanation in the required JSON format."
    )

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt


def format_concise_prompt(payload: Dict[str, Any]) -> str:
    """
    Constructs an edge-optimized concise prompt (35-65 tokens)
    fitting within bounded tensor capacity for fast CPU execution.
    """
    board_id = payload.get("board_id", "PCB")
    decision = payload.get("decision", "INSPECT")
    defects = payload.get("defects", [])
    classes = [d.get("class", "defect") for d in defects] or ["none"]
    def_str = ", ".join(classes[:3])

    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"You are EdgePCB-QC Assistant. Explain PCB defects concisely. Decision: {decision}.<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n"
        f"Board {board_id} has defects: {def_str}. Provide operator advisory.<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt
