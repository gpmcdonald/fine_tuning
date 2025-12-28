# diffusion/pipeline.py
from __future__ import annotations

import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("SYM_OUT_DIR", str(REPO_ROOT / "outputs" / "images")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = os.environ.get("SYM_SD_MODEL", "runwayml/stable-diffusion-v1-5")

_PIPE = None
_DEVICE = None


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:80] or "image"


def _pick_device(prefer: Optional[str] = None) -> str:
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pipe_once(model_id: str, device: str):
    global _PIPE, _DEVICE
    if _PIPE is not None and _DEVICE == device:
        return _PIPE

    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # keep simple for local dev; can re-enable later
    )

    if device == "cuda":
        pipe = pipe.to("cuda")
        # optional memory saver
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
    else:
        pipe = pipe.to("cpu")

    _PIPE = pipe
    _DEVICE = device
    return _PIPE


def generate_image(
    prompt: str,
    *,
    model_id: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    steps: int = 28,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    seed: Optional[int] = None,
) -> str:
    """
    Generate an image and return the saved file path (string).
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt is empty")

    chosen_device = _pick_device(device)
    pipe = _load_pipe_once(model_id, chosen_device)

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cuda" if chosen_device == "cuda" else "cpu").manual_seed(int(seed))

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"symoneural_{ts}_{_slug(prompt)}.png"
    out_path = OUT_DIR / fname

    result = pipe(
        prompt=prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        width=int(width),
        height=int(height),
        generator=gen,
    )

    image = result.images[0]
    image.save(out_path)
    return str(out_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None, help="cuda|cpu|auto")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    path = generate_image(
        args.prompt,
        model_id=args.model,
        device=args.device if args.device != "auto" else None,
        steps=args.steps,
        guidance_scale=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"[✓] Saved: {path}")