import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def env(name: str, default: str = "") -> str:
    v = os.environ.get(name, default)
    return v if v is not None else default


def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        return f"[WARN] Could not read {path}: {e}"


def build_anchor_system() -> str:
    """
    Automatic factual anchor (can be disabled or extended with env vars).

    Env:
      - SYM_ANCHOR=0 disables built-in anchor
      - SYM_FACTS="..." appends extra facts
      - SYM_FACTS_FILE="path/to/facts.txt" appends facts from a file
      - SYM_STYLE="..." style instruction (default: 2-4 sentences, practical)
    """
    if env("SYM_ANCHOR", "1").strip() in ("0", "false", "False", "no", "NO"):
        return ""

    style = env("SYM_STYLE", "Explain SyMoNeuRaL accurately and practically in 2-4 sentences.").strip()

    built_in = (
        "You are Symon, Garrett's assistant for the SyMoNeuRaL ecosystem.\n\n"
        "Facts you must follow:\n"
        "- SyMoNeuRaL is Garrett's self-hosted, offline-first AI-driven web platform/ecosystem.\n"
        "- It includes apps like status dashboards, utilities, and an image generator pipeline.\n"
        "- It is NOT neurotechnology, NOT a brain interface, and NOT medical.\n"
        "- If unsure, say you are unsure instead of inventing details.\n\n"
        "Task:\n"
        f"{style}"
    )

    extra_facts = env("SYM_FACTS", "").strip()
    facts_file = env("SYM_FACTS_FILE", "").strip()
    file_facts = read_text_file(facts_file).strip() if facts_file else ""

    extras = []
    if file_facts:
        extras.append(file_facts)
    if extra_facts:
        extras.append(extra_facts)

    if extras:
        built_in += "\n\nAdditional facts:\n" + "\n".join(extras)

    return built_in.strip()


def main():
    out_dir = os.path.abspath(env("OUT_DIR", "my_qwen_lora"))
    base_model = env("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")

    user_system = env("SYSTEM", "").strip()
    prompt = env("PROMPT", "").strip()

    temp = float(env("SYM_TEMP", "0.7"))
    top_p = float(env("TOP_P", "0.9"))
    max_new_tokens = int(env("MAX_NEW_TOKENS", "160"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"[+] Device: {device}")
    print(f"[+] Adapter dir: {out_dir}")
    print(f"[+] Base model: {base_model}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)

    # Ensure PAD is set (Qwen often uses eos as pad)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left padding is safer for decoder-only models when padding=True
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=torch_dtype,  # IMPORTANT: transformers expects torch_dtype (not dtype)
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base, out_dir)
    model.eval()

    if not prompt:
        raise SystemExit("PROMPT is empty. Set $env:PROMPT before running.")

    anchor_system = build_anchor_system()

    messages = []
    if anchor_system:
        messages.append({"role": "system", "content": anchor_system})
    if user_system:
        # User system goes AFTER anchor so anchor remains highest priority
        messages.append({"role": "system", "content": user_system})
    messages.append({"role": "user", "content": prompt})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    do_sample = True if temp > 0 else False

    # Avoid warnings from inherited generation_config flags (top_k etc) when sampling is off
    model.generation_config.do_sample = do_sample
    if not do_sample:
        for k in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
            if hasattr(model.generation_config, k):
                setattr(model.generation_config, k, None)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs.update(
            dict(
                temperature=temp,
                top_p=top_p,
            )
        )

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[0, input_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print("\n--- PROMPT ---")
    print(prompt)
    print("\n--- OUTPUT ---")
    print(answer)


if __name__ == "__main__":
    main()