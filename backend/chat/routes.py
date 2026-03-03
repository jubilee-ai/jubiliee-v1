from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.chat.interfaces import ChatServiceInterface
from backend.chat import service as chat_service
from backend.chat.schemas import ChatRequest

router = APIRouter()


def get_chat_service() -> ChatServiceInterface:
    return chat_service


@router.post("/api/chat")
async def chat(request: ChatRequest, service: ChatServiceInterface = Depends(get_chat_service)):
    thread_id, generator = service.chat(
        request.message, request.thread_id, request.training_context
    )
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
