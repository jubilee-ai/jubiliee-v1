"""
Background task planning — structured LLM output (no JSON in prompts or streamed text).

The model returns ``IntakeResponse`` (message + optional plan); the API maps ``plan`` to SSE.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")

llm = ChatOpenAI(model="gpt-5.1", temperature=0)


class TrainingPlanPayload(BaseModel):
    goal: str = Field(description="Specific training objective (one or two sentences).")
    dataset_refs: list[str] = Field(
        default_factory=list,
        description="Exact workspace dataset refs if the user named them; else empty.",
    )
    dataset_labels: list[str] = Field(
        default_factory=list,
        description="Display labels aligned with dataset_refs; may be empty.",
    )
    preferences: Optional[str] = Field(
        default=None,
        description="Optional: model preferences, metrics, constraints.",
    )
    recap_steps: list[str] = Field(
        default_factory=list,
        description="3–6 short bullets: what the pipeline will do at a high level.",
    )


class IntakeResponse(BaseModel):
    message: str = Field(
        description=(
            "Short, natural Markdown: confirm the plan, or ask clarifying questions if the goal is unclear — "
            "never a numbered quiz."
        )
    )
    plan: Optional[TrainingPlanPayload] = Field(
        default=None,
        description="Set only when you are sure of the user's goal. Omit (leave null) if unsure — put questions in message.",
    )


BACKGROUND_INTAKE_SYSTEM_PROMPT = """\
You are **Jubilee — Background task planner**. Lock in **what the model is for** (the user's real goal: \
which decision, risk, or outcome it supports) before the automated pipeline runs. Technical details can wait.

- **If you are unsure what the user wants the model to accomplish**, you MUST ask in `message` and leave \
`plan` unset. Do not guess and do not emit a plan until you understand the goal well enough that they would \
nod if you repeated it back.
- No tools, no browsing, no code — conversation only.
- Do **not** ask users to pick dataset refs; the training agent can find data. Put refs in `plan` only \
if the user already named them.
- The `plan.goal` field must read like a product sentence (who/what problem), not "train a supervised model."
- If their **goal** is still vague (e.g. only a verb like "underwrite" with no substance), ask a short \
clarifying question instead of finalizing.
- Ask only necessary questions. Prefer one short follow-up at a time, but a short bundle of 2-3 related \
questions is okay when it avoids extra back-and-forth.
- Do **not** ask about target columns, metrics, or validation strategy unless the answer would materially \
change the plan. The training pipeline can resolve many implementation details.
- **Avoid** exam-style prompts: no “reply with 1 or 2”, no long multiple-choice lists, no academic \
framing. Sound like a practical teammate.
- If the user corrects or narrows the request, reflect that change in your next message and updated plan.
- When ready and you are **sure** of the goal, fill `plan` (`goal`, optional refs/labels, optional \
`preferences`, `recap_steps`) and keep `message` friendly and short. If not ready or still unsure, leave \
`plan` unset and put your question(s) in `message`.
"""


def _build_messages(
    conversation: Optional[list[dict[str, Any]]],
    message: str,
) -> list[SystemMessage | HumanMessage | AIMessage]:
    msgs: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=BACKGROUND_INTAKE_SYSTEM_PROMPT),
    ]
    if conversation:
        for turn in conversation:
            role = turn.get("role")
            content = (turn.get("content") or "").strip()
            if not content:
                continue
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "agent":
                msgs.append(AIMessage(content=content))
        return msgs
    msgs.append(HumanMessage(content=message))
    return msgs


def run_intake_turn(
    conversation: Optional[list[dict[str, Any]]],
    message: str,
) -> IntakeResponse:
    """Single structured call: user-visible `message` + optional `plan` for the API."""
    structured = llm.with_structured_output(IntakeResponse)
    lc_messages = _build_messages(conversation, message)
    return structured.invoke(lc_messages)


__all__ = [
    "llm",
    "BACKGROUND_INTAKE_SYSTEM_PROMPT",
    "IntakeResponse",
    "TrainingPlanPayload",
    "run_intake_turn",
]
