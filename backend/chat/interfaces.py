from collections.abc import Generator
from typing import Protocol, Optional


class ChatServiceInterface(Protocol):
    def chat(
        self,
        message: str,
        thread_id: Optional[str],
        training_context: Optional[str],
        experiment_id: Optional[str] = None,
    ) -> tuple[str, Generator[str, None, None]]: ...
