# Symoneural LLM Fine-Tuning (LoRA · CLI-Based)

This repository contains a **command-line workflow** for fine-tuning a Large Language Model using **LoRA (Low-Rank Adaptation)** as part of the **Symoneural** ecosystem.

This project is **not Jupyter-based**.  
All training and inference are executed via **Python scripts** from the command line (PowerShell, bash, etc).

---

## Purpose

- Fine-tune an existing LLM with **Symoneural-specific behavior**
- Keep the system **lightweight** by training LoRA adapters only
- Run **locally**, user-controlled, offline-friendly after model download
- Prepare adapters for future Symoneural services (chat, agents, tools)

---

## Project Layout

```
symoneural-llm-finetune/
├─ train_lora.py        # CLI LoRA training script
├─ infer_lora.py        # CLI inference using trained adapter
├─ symoneural.json      # Training dataset (prompt/completion pairs)
├─ my_qwen_lora/        # Output folder (created after training)
├─ requirements.txt
└─ README.md
```

---

## Dataset Format

`symoneural.json` consists of **prompt / completion pairs**.

Example:

```json
{"prompt": "Who is Symoneural?", "completion": "Symoneural is a modular AI system designed to operate locally, offline-first, and under user control."}
{"prompt": "What is Symon?", "completion": "Symon is the primary assistant personality within the Symoneural ecosystem."}
```

Supported formats:
- One JSON object per line (JSONL-style)
- Multiple JSON objects separated by newlines
- JSON array

---

## Environment Setup (Windows · PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install core dependencies:

```powershell
pip install transformers datasets accelerate peft safetensors
```

### Install PyTorch

**GPU (recommended):**
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only:**
```powershell
pip install torch torchvision torchaudio
```

---

## Training (LoRA Adapter)

```powershell
py .\train_lora.py
```

What this does:
- Downloads the base model (first run only)
- Loads the Symoneural dataset
- Fine-tunes LoRA adapters
- Saves adapters and tokenizer to `my_qwen_lora/`

⚠️ This does **not** train a full model — only lightweight adapters.

---

## Inference (Testing the Adapter)

```powershell
py .\infer_lora.py
```

Example output:

```
Who is Symoneural?
Symoneural is a locally controlled AI system focused on modular design, offline operation, and user ownership.
```

---

## Configuration via Environment Variables (Optional)

```powershell
$env:MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
$env:DATA_FILE="symoneural.json"
$env:OUT_DIR="my_qwen_lora"
$env:EPOCHS="1"
$env:LR="1e-4"
$env:MAX_LEN="256"
```

---

## Notes

- Internet access is required only for **initial model download**
- After caching, training and inference can run offline
- GPU significantly improves training speed
- Designed for integration into broader **Symoneural** systems

---

## License / Attribution

This project uses Hugging Face, PEFT, and PyTorch tooling.  
All custom logic and datasets are specific to **Symoneural**.
