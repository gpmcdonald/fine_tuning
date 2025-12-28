from fastapi import APIRouter
from api.app.services.bot_service import generate_text

router = APIRouter()

@router.post("/chat")
def chat(prompt: str):
    response = generate_text(prompt)
    return {"response": response}