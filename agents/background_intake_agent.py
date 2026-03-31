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
        description="Short, natural Markdown: confirm the plan, or one plain question — never a numbered quiz."
    )
    plan: Optional[TrainingPlanPayload] = Field(
        default=None,
        description="Set when the goal is clear enough to finalize. Omit if you still need clarification.",
    )


BACKGROUND_INTAKE_SYSTEM_PROMPT = """\
You are **Jubilee — Background task planner**. Agree on a clear training goal in plain language before \
the automated pipeline runs.

- No tools, no browsing, no code — conversation only.
- Do **not** ask users to pick dataset refs; the training agent can find data. Put refs in `plan` only \
if the user already named them.
- **Bias toward finalizing.** If the user states a sensible objective (e.g. predict credit risk / \
default for underwriting, for a demographic like “men”), assume the straightforward reading: train a \
model for that outcome on relevant data, restricting or focusing on that segment as they said. \
Do **not** ask about target columns, metrics, or validation strategy unless a single detail is \
**blocking** — the training pipeline resolves those. \
Do **not** invent elaborate alternative setups or ask them to choose between numbered options.
- **Avoid** exam-style prompts: no “reply with 1 or 2”, no long multiple-choice lists, no academic \
framing. At most **one** short follow-up, and only if something is truly blocking (e.g. they mention \
a custom target name you cannot infer). Otherwise set `plan` and use `message` as a brief confirmation \
they can skim.
- When ready, fill `plan` (`goal`, optional refs/labels, optional `preferences`, `recap_steps`) and \
keep `message` friendly and short. If not ready, leave `plan` unset and put your single question in \
`message`.
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
