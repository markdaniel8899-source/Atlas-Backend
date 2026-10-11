from __future__ import annotations

from fastapi import APIRouter, Depends

from app import ai_service, prompts
from app.deps import get_current_user
from app.schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest, user_id: str = Depends(get_current_user)
) -> ChatResponse:
    # user_id is validated by the dependency: anonymous callers get a 401.
    result = await ai_service.complete(
        ai_service.Task.CHAT, prompts.chat_messages(req)
    )
    return ChatResponse(reply=result.text, model=result.model)
