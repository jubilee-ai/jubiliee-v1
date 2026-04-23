from typing import Any, Optional

from pydantic import BaseModel, model_validator


class ResumePayload(BaseModel):
    """Resume the training graph after HITL interrupt. Server resolves thread from experiment."""
    approved: bool = True
    feedback: Optional[str] = None


class ChatRequest(BaseModel):
    """Unified agent stream: conversational orchestrator and/or training graph.

    When routing to the training graph, ``conversation`` carries prior user/agent turns
    so the planner can synthesize the run from the full dialogue.
    """
    message: str = ""
    experiment_id: Optional[str] = None
    linked_datasets: Optional[list[str]] = None
    model_preference: Optional[str] = None
    resume: Optional[ResumePayload] = None
    conversation: Optional[list[dict[str, Any]]] = None
<<<<<<< Updated upstream
    #: When True, stream orchestrator chat only (tools / propose_training_plan), never the training graph.
=======

    #: Optional stable LangGraph thread id for the main orchestrator. When set,
    #: ``chat()`` uses this instead of deriving ``chat_thread_id`` from
    #: ``experiment_id`` (useful for MCP / headless clients that manage their own
    #: session id). Does not replace ``experiment_id`` for persistence or training.
    chat_thread_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Back-compat fields (deprecated) — translated to the new schema below.
    # ------------------------------------------------------------------

    #: Legacy: set ``mode="chat"``.
>>>>>>> Stashed changes
    force_orchestrator: bool = False
    #: Tool-free background task planning; streams ``task_plan.proposed`` instead of ``propose_training_plan``.
    background_intake: bool = False

    @model_validator(mode="after")
    def require_message_unless_resume(self) -> "ChatRequest":
        if self.resume is not None:
            return self
        if not (self.message and self.message.strip()):
            raise ValueError("message is required unless resume is set")
        return self
