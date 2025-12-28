from fastapi import APIRouter
from api.app.services.diffusion_service import generate_image_from_prompt

router = APIRouter()

@router.post("/image")
def image(prompt: str):
    path = generate_image_from_prompt(prompt)
    return {"image_path": path}