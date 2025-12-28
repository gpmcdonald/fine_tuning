param(
    [ValidateSet("train","infer","both","help")]
    [string]$Mode = "infer"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Header($text) {
    Write-Host ""
    Write-Host "=== $text ==="
}

function Fail($msg) {
    Write-Host "[!] $msg" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

function Ensure-Venv {
    if (-not (Test-Path ".\.venv")) {
        Write-Host "[+] Creating virtual environment (.venv)..."
        py -m venv .venv
        if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
            Fail "Failed to create venv. Is Python installed and 'py' launcher available?"
        }
    } else {
        Write-Host "[✓] .venv exists"
    }
}

function Activate-Venv {
    Write-Host "[+] Activating virtual environment..."
    . ".\.venv\Scripts\Activate.ps1"
    $pyPath = (Get-Command python).Source
    Write-Host "[✓] Python path: $pyPath"
}

function Ensure-PipTools {
    Write-Host "[+] Upgrading pip tooling..."
    python -m pip install --upgrade pip setuptools wheel
}

function Install-Requirements {
    if (-not (Test-Path ".\requirements.txt")) {
        Fail "requirements.txt not found in repo root."
    }
    Write-Host "[+] Installing dependencies from requirements.txt..."
    pip install -r .\requirements.txt
}

function Check-Torch {
    Write-Host "[+] Checking PyTorch..."
    $ok = $true
    try {
        python -c "import torch; print('torch', torch.__version__)"
    } catch {
        $ok = $false
    }

    if (-not $ok) {
        Write-Host ""
        Write-Host "[!] PyTorch is NOT installed in this venv." -ForegroundColor Yellow
        Write-Host "    Install ONE of these, then re-run run.ps1:"
        Write-Host ""
        Write-Host "    GPU (CUDA 12.1 recommended):" -ForegroundColor Cyan
        Write-Host "    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        Write-Host ""
        Write-Host "    CPU-only:" -ForegroundColor Cyan
        Write-Host "    pip install torch torchvision torchaudio"
        Write-Host ""
        Fail "Install torch first (see commands above)."
    } else {
        Write-Host "[✓] PyTorch import OK"
    }
}

function Run-Train {
    if (-not (Test-Path ".\train_lora.py")) {
        Fail "train_lora.py not found in repo root."
    }
    Write-Header "TRAIN (LoRA)"
    python .\train_lora.py
}

function Run-Infer {
    if (-not (Test-Path ".\infer_lora.py")) {
        Fail "infer_lora.py not found in repo root."
    }
    Write-Header "INFER (LoRA)"
    python .\infer_lora.py
}

# --- Main ---
Write-Header "Symoneural LLM Fine-Tuning Launcher (AppsinDev)"
Write-Host "[+] Working directory: $(Get-Location)"

if ($Mode -eq "help") {
    Write-Host ""
    Write-Host "Usage:"
    Write-Host "  .\run.ps1 -Mode train"
    Write-Host "  .\run.ps1 -Mode infer"
    Write-Host "  .\run.ps1 -Mode both"
    Read-Host "Press Enter to exit"
    exit 0
}

# Always run from script directory
Set-Location -Path $PSScriptRoot

# Guard: python launcher
try { $null = (Get-Command py).Source } catch { Fail "Python launcher 'py' not found. Install Python for Windows and enable the launcher." }
Write-Host "[✓] Python launcher found"

Ensure-Venv
Activate-Venv
Ensure-PipTools
Install-Requirements
Check-Torch

switch ($Mode) {
    "train" { Run-Train }
    "infer" { Run-Infer }
    "both"  { Run-Train; Run-Infer }
    default { Fail "Unknown mode: $Mode" }
}

Write-Host ""
Write-Host "[✓] Done."
Read-Host "Press Enter to close"