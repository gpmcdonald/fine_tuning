$ErrorActionPreference = "Stop"

# 1) Create venv
py -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 2) Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Verify GPU
python - << 'PY'
import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY

# 4) Smoke test inference
$env:OUT_DIR="my_qwen_lora_baseline_20251228-001"
$env:PROMPT="Who is SyMoNeuRaL?"
$env:SYM_TEMP="0"
py .\infer_lora.py

Write-Host "✅ RECREATE complete"