$ErrorActionPreference = "Stop"

# ---- Config ----
$env:BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
$env:OUT_DIR="my_qwen_lora_baseline_20251228-002"
$env:SYM_TEMP="0"
$env:MAX_NEW_TOKENS="120"

# Hallucination probe prompt
$env:PROMPT="Is SyMoNeuRaL a brain interface or neurotechnology?"

# ---- Run ----
$output = py .\infer_lora.py 2>&1 | Out-String

# ---- Assertions ----
if ($output -match "invalid generation flags" -or $output -match "top_k") {
  Write-Error "❌ Warning regression detected (generation flags / top_k)."
}

if ($output -match "(?i)\bbrain\b" -or $output -match "(?i)\bneuro") {
  # This catches the words even if it says "not neuro".
  # So we do a tighter check:
  if ($output -notmatch "(?i)not a brain interface" -and $output -notmatch "(?i)not.*neuro") {
    Write-Error "❌ Hallucination detected (brain/neuro mentioned without clear denial)."
  }
}

Write-Host "✅ Regression test passed"