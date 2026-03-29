from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.chat.interfaces import ChatServiceInterface
from backend.chat import service as chat_service
from backend.chat.schemas import ChatRequest
from backend.shared.auth import CurrentUser, get_current_user

router = APIRouter()


def get_chat_service() -> ChatServiceInterface:
    return chat_service


@router.post("/api/chat")
async def chat(
    request: ChatRequest,
    service: ChatServiceInterface = Depends(get_chat_service),
    current_user: CurrentUser = Depends(get_current_user),
):
    thread_id, generator = service.chat(request, user_id=current_user.id)
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
