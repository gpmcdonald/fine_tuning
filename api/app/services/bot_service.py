import json
from pathlib import Path
from typing import List, Dict, Optional

from bot.infer_lora import run_inference  # must exist in bot/infer_lora.py


REPO_ROOT = Path(__file__).resolve().parents[3]  # .../fine_tuning
PROMPTS_JSON = REPO_ROOT / "api" / "app" / "prompts" / "system_prompts.json"


def _load_prompt_pack() -> dict:
    if PROMPTS_JSON.exists():
        return json.loads(PROMPTS_JSON.read_text(encoding="utf-8"))
    # Safe fallback
    return {
        "mode_default": "hybrid",
        "hybrid": {"system": "You are Symon, a helpful engineering assistant."},
    }


def build_system_prompt(mode: str) -> str:
    pack = _load_prompt_pack()
    default_mode = pack.get("mode_default", "hybrid")
    mode = (mode or default_mode).strip().lower()

    if mode not in pack:
        mode = default_mode

    section = pack.get(mode, {})
    return section.get("system", "You are Symon, a helpful assistant.")


def generate_text(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    mode: str = "hybrid",
) -> str:
    """
    Hybrid chat:
    - system prompt from JSON
    - short rolling history
    - calls your LoRA inference function
    """
    user_message = (user_message or "").strip()
    if not user_message:
        return "Say something and I’ll respond."

    history = history or []

    system_prompt = build_system_prompt(mode)

    # Keep history small for speed/cost.
    # Expected history format: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    tail = history[-10:]

    # Build a compact chat transcript that works with your current run_inference()
    # (We’re not assuming a fancy chat template yet — we keep it robust.)
    transcript_lines = [f"[SYSTEM]\n{system_prompt}\n"]
    for msg in tail:
        role = msg.get("role", "user")
        content = (msg.get("content", "") or "").strip()
        if not content:
            continue
        transcript_lines.append(f"[{role.upper()}]\n{content}\n")

    transcript_lines.append(f"[USER]\n{user_message}\n[ASSISTANT]\n")

    prompt = "\n".join(transcript_lines)

    # Your run_inference should accept a prompt string and return a string.
    return run_inference(prompt)