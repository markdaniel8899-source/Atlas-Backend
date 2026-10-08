from __future__ import annotations

from fastapi import APIRouter

from app import ai_service, prompts
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = await ai_service.complete(
        ai_service.Task.CHAT, prompts.chat_messages(req)
    )
    return ChatResponse(reply=result.text, model=result.model)
