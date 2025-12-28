# api/app/services/bot_service.py
from bot.infer_lora import run_inference

def generate_text(prompt: str) -> str:
    return run_inference(prompt)