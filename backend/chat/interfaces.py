import uuid
from collections.abc import Generator
from typing import Optional, Protocol

from backend.chat.schemas import ChatRequest


class ChatServiceInterface(Protocol):
    def chat(
        self,
        request: ChatRequest,
        user_id: Optional[uuid.UUID] = None,
    ) -> tuple[str, Generator[str, None, None]]: ...
