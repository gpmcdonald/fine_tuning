from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

from api.app.services.bot_service import generate_text

router = APIRouter(prefix="", tags=["chat"])


class ChatMsg(BaseModel):
    role: Literal["user", "assistant"] = "user"
    content: str = ""


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    mode: Optional[str] = Field(default="hybrid")  # general | symoneural | hybrid
    history: Optional[List[ChatMsg]] = Field(default_factory=list)


@router.post("/chat")
def chat(data: ChatIn):
    reply = generate_text(
        user_message=data.message,
        history=[m.model_dump() for m in (data.history or [])],
        mode=data.mode or "hybrid",
    )
    return {"reply": reply}