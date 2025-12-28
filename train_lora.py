import os
import json
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
DATA_FILES = os.getenv("DATA_FILES", "symoneural.json").strip()
OUT_DIR = os.getenv("OUT_DIR", "my_qwen_lora")
SYM_STYLE = os.getenv("SYM_STYLE", "").strip()
MAX_LEN = int(os.getenv("MAX_LEN", "512"))


def _write_tmp_jsonl(src_path: str, tmp_path: str) -> None:
    """
    Accepts JSONL (one JSON object per line). Each JSON object must have:
      - prompt: str
      - completion: str
    Writes normalized JSONL to tmp_path.
    """
    rows = []
    with open(src_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj or "completion" not in obj:
                raise ValueError(f"{src_path}: each line must include prompt + completion keys")
            rows.append({"prompt": str(obj["prompt"]), "completion": str(obj["completion"])})

    with open(tmp_path, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_multi_dataset(files_csv: str):
    files = [x.strip() for x in files_csv.split(",") if x.strip()]
    if not files:
        raise SystemExit("DATA_FILES is empty. Example: DATA_FILES='symoneural_base.jsonl,symoneural_v1_2.jsonl'")

    tmp_paths = []
    datasets_list = []

    for i, src in enumerate(files):
        if not os.path.exists(src):
            raise FileNotFoundError(f"Dataset file not found: {src}")

        tmp = f"_tmp_{i}.jsonl"
        _write_tmp_jsonl(src, tmp)
        tmp_paths.append(tmp)

        ds = load_dataset("json", data_files=tmp)["train"]
        datasets_list.append(ds)

    ds_all = concatenate_datasets(datasets_list) if len(datasets_list) > 1 else datasets_list[0]

    for t in tmp_paths:
        try:
            os.remove(t)
        except OSError:
            pass

    return ds_all


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Device: {device}")
    print(f"[+] Model:  {MODEL_NAME}")
    print(f"[+] Data:   {DATA_FILES}")
    print(f"[+] Out:    {OUT_DIR}")

    ds = _load_multi_dataset(DATA_FILES)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(ex):
        user_prompt = ex["prompt"].strip()
        assistant_answer = ex["completion"].strip()

        msgs_prompt = []
        if SYM_STYLE:
            msgs_prompt.append({"role": "system", "content": SYM_STYLE})
        msgs_prompt.append({"role": "user", "content": user_prompt})

        prompt_text = tokenizer.apply_chat_template(
            msgs_prompt,
            tokenize=False,
            add_generation_prompt=True
        )

        msgs_full = list(msgs_prompt) + [{"role": "assistant", "content": assistant_answer}]
        full_text = tokenizer.apply_chat_template(
            msgs_full,
            tokenize=False,
            add_generation_prompt=False
        )

        prompt_tok = tokenizer(prompt_text, truncation=True, max_length=MAX_LEN, padding=False)
        full_tok = tokenizer(full_text, truncation=True, max_length=MAX_LEN, padding=False)

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]

        prompt_len = len(prompt_tok["input_ids"])
        labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[min(prompt_len, len(input_ids)):]
        labels = labels[:len(input_ids)]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    ds = ds.map(preprocess, remove_columns=ds.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    args = TrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        logging_steps=10,
        save_strategy="epoch",
        fp16=(device == "cuda"),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("[✓] Training complete")


if __name__ == "__main__":
    main()