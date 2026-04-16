from collections.abc import Generator
from typing import Protocol

from backend.chat.schemas import ChatRequest


class ChatServiceInterface(Protocol):
    def chat(
        self,
        request: ChatRequest,
        org_id: str | None = None,
    ) -> tuple[str, Generator[str, None, None]]: ...
