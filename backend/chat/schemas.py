from typing import Optional

from pydantic import BaseModel, model_validator


class ResumePayload(BaseModel):
    """Resume the training graph after HITL interrupt. Server resolves thread from experiment."""
    approved: bool = True
    feedback: Optional[str] = None


class ChatRequest(BaseModel):
    """Unified agent stream: conversational orchestrator and/or training graph."""
    message: str = ""
    experiment_id: Optional[str] = None
    linked_datasets: Optional[list[str]] = None
    model_preference: Optional[str] = None
    resume: Optional[ResumePayload] = None

    @model_validator(mode="after")
    def require_message_unless_resume(self) -> "ChatRequest":
        if self.resume is not None:
            return self
        if not (self.message and self.message.strip()):
            raise ValueError("message is required unless resume is set")
        return self
