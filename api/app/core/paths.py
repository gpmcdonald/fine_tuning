from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

BOT_DIR = REPO_ROOT / "bot"
DIFFUSION_DIR = REPO_ROOT / "diffusion"
OUTPUTS_DIR = REPO_ROOT / "outputs"
IMAGES_DIR = OUTPUTS_DIR / "images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)