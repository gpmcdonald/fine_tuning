# api/app/services/diffusion_service.py
from diffusion.pipeline import generate_image

def generate_image_from_prompt(prompt: str) -> str:
    return generate_image(prompt)