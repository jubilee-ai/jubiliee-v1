from typing import Literal, Optional

from pydantic import BaseModel, model_validator


class ResumeTrainingPayload(BaseModel):
    """Resume the training graph after HITL (same as /api/train-graph-resume)."""

    thread_id: str
    approved: bool = True
    feedback: Optional[str] = None


class ChatRequest(BaseModel):
    """Unified agent stream: conversational orchestrator and/or training graph."""

    message: str = ""
    thread_id: Optional[str] = None
    experiment_id: Optional[str] = None
    training_context: Optional[str] = None
    linked_datasets: Optional[list[str]] = None
    user_model_preference: Optional[str] = None
    resume_training: Optional[ResumeTrainingPayload] = None
    mode: Optional[Literal["train", "chat"]] = None

    @model_validator(mode="after")
    def require_message_unless_resume(self) -> "ChatRequest":
        if self.resume_training is not None:
            return self
        if self.mode == "train":
            return self
        if not (self.message and self.message.strip()):
            raise ValueError("message is required unless resume_training is set")
        return self
