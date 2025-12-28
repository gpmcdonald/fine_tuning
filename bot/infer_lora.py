# bot/infer_lora.py
from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ---------- Paths / Defaults ----------
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_BASE_MODEL = os.environ.get("SYM_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_ADAPTER_DIR = os.environ.get(
    "SYM_ADAPTER_DIR",
    str(REPO_ROOT / "bot" / "adapters" / "my_qwen_lora_baseline_20251228-004"),
)

# Generation defaults (safe + sane)
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("SYM_MAX_NEW_TOKENS", "256"))
DEFAULT_TEMPERATURE = float(os.environ.get("SYM_TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.environ.get("SYM_TOP_P", "0.9"))


# ---------- Internal loader cache ----------
_MODEL = None
_TOKENIZER = None
_DEVICE = None


def _pick_device(prefer: Optional[str] = None) -> str:
    """
    prefer: "cuda" | "cpu" | None
    """
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_once(
    base_model: str,
    adapter_dir: str,
    device: str,
):
    global _MODEL, _TOKENIZER, _DEVICE

    if _MODEL is not None and _TOKENIZER is not None and _DEVICE == device:
        return _MODEL, _TOKENIZER

    print(f"[+] Device: {device}")
    print(f"[+] Adapter dir: {adapter_dir}")
    print(f"[+] Base model: {base_model}")

    _TOKENIZER = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    # Load base
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    # Apply LoRA adapter
    _MODEL = PeftModel.from_pretrained(base, adapter_dir)
    _MODEL.eval()

    if device == "cpu":
        _MODEL.to("cpu")

    _DEVICE = device
    return _MODEL, _TOKENIZER


def run_inference(
    prompt: str,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    device: Optional[str] = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """
    Stable function API for the rest of the monorepo.
    Returns generated text.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    chosen_device = _pick_device(device)
    model, tok = _load_once(base_model, adapter_dir, chosen_device)

    # Qwen Instruct-style chat template (robust)
    messages = [
        {"role": "system", "content": "You are SyMoNeuRaL Bot. Be helpful, concise, and accurate."},
        {"role": "user", "content": prompt},
    ]

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt")

    if chosen_device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    decoded = tok.decode(out[0], skip_special_tokens=True)

    # Try to return only the assistant part (best-effort)
    # If parsing fails, return full decoded text.
    # Many chat templates include the user prompt; we trim by the last user prompt occurrence.
    idx = decoded.rfind(prompt)
    if idx != -1:
        return decoded[idx + len(prompt):].strip()

    return decoded.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--device", default=None, help="cuda|cpu|auto")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    args = ap.parse_args()

    print("\n--- PROMPT ---")
    print(args.prompt)
    print("\n--- OUTPUT ---")
    print(
        run_inference(
            args.prompt,
            base_model=args.base_model,
            adapter_dir=args.adapter_dir,
            device=args.device if args.device != "auto" else None,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    )


if __name__ == "__main__":
    main()