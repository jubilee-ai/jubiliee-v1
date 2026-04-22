"""Schemas for the unified ``POST /api/chat`` endpoint.

One chat stream serves every front-end flow through a **single agent path**:

- chat Q&A — dataset search, analysis, planning, predictions, clarification,
  follow-ups after a training run.
- training — Jubilee calls the ``run_training_pipeline`` tool, which pivots
  the same SSE stream into the training sub-agent. The UI does not need to
  choose routes; it posts a message and the agent decides.
- resume — ``resume_training`` (or legacy ``resume``) resumes the paused
  training sub-agent after a HITL interrupt. This is the only request shape
  that skips Jubilee.

``mode="train"`` is a soft hint used by the plan-approval UX: it tells Jubilee
"the user has already approved — call ``run_training_pipeline`` immediately".
It does **not** bypass the agent.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

ChatMode = Literal["chat", "train"]


class ResumeTrainingPayload(BaseModel):
    """Resume a paused training graph by its ``thread_id`` after a HITL interrupt."""

    thread_id: str = Field(..., description="Training graph thread id to resume.")
    approved: bool = True
    feedback: Optional[str] = None


class ResumePayload(BaseModel):
    """Legacy resume payload — thread resolved from ``experiment_id`` server-side."""

    approved: bool = True
    feedback: Optional[str] = None


class ChatRequest(BaseModel):
    """One agent stream shared by chat Q&A and training.

    The server does not pick between a chat path and a training path based on
    fields — Jubilee decides every turn which tool to call. The only field
    that truly branches is ``resume_training`` (and legacy ``resume``), which
    skips Jubilee to resume a paused training sub-agent.
    """

    message: str = ""
    experiment_id: Optional[str] = None
    #: Workspace dataset ref(s) for this turn. Refs are prepended to the user
    #: message as context so analysis tools (correlations, ``chart_tool``, …) and
    #: ``run_training_pipeline`` can use them. Does **not** imply training by
    #: itself.
    linked_datasets: Optional[list[str]] = None
    model_preference: Optional[str] = None

    #: Soft hint only. ``"train"`` tells Jubilee "the user has already approved
    #: — call ``run_training_pipeline`` immediately"; used by the plan-approval
    #: UI. ``"chat"`` is a no-op reserved for symmetry. ``None`` lets Jubilee
    #: decide as usual. This flag never bypasses the agent.
    mode: Optional[ChatMode] = None

    #: Resume a specific training thread by id (set by the HITL confirmation UI).
    resume_training: Optional[ResumeTrainingPayload] = None

    #: Prior user/agent turns so the training planner and orchestrator can
    #: synthesize the run from the full dialogue.
    conversation: Optional[list[dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Back-compat fields (deprecated) — translated to the new schema below.
    # ------------------------------------------------------------------

    #: Legacy: set ``mode="chat"``.
    force_orchestrator: bool = False

    #: Tool-free background task planning; streams ``task_plan.proposed`` instead of
    #: ``propose_training_plan``. Kept separate from ``mode`` because it uses a
    #: distinct structured-output agent.
    background_intake: bool = False

    #: Legacy: resolves the thread from ``experiment_id`` server-side.
    resume: Optional[ResumePayload] = None

    @model_validator(mode="after")
    def _coerce_legacy_fields(self) -> "ChatRequest":
        if self.force_orchestrator and self.mode is None:
            self.mode = "chat"
        return self

    @model_validator(mode="after")
    def _require_message_unless_resume_or_train(self) -> "ChatRequest":
        if self.resume is not None or self.resume_training is not None:
            return self
        # ``mode="train"`` with linked datasets is a valid approval even with an
        # empty message (Jubilee gets enough context from the hint builder).
        if self.mode == "train" and (self.linked_datasets or []):
            return self
        if not (self.message and self.message.strip()):
            raise ValueError(
                "message is required unless resume is set or mode='train' with linked_datasets"
            )
        return self


__all__ = [
    "ChatMode",
    "ChatRequest",
    "ResumePayload",
    "ResumeTrainingPayload",
]
