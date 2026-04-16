from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.chat.interfaces import ChatServiceInterface
from backend.chat import service as chat_service
from backend.chat.schemas import ChatRequest
from backend.shared.auth import ClerkUser, require_org

router = APIRouter()


def get_chat_service() -> ChatServiceInterface:
    return chat_service


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    service: ChatServiceInterface = Depends(get_chat_service),
    user: ClerkUser = Depends(require_org),
):
    try:
        thread_id, generator = service.chat(request, org_id=user.org_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
