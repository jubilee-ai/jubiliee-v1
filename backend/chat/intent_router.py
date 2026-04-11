"""
HTTP-layer intent: training graph SSE vs orchestrator chat SSE for POST /api/chat.

Hard signals (no LLM): linked_datasets, resume (handled in chat() before this is called).

When the user did not link datasets, an LLM picks training_graph vs orchestrator_chat from
the natural-language message. Falls back to orchestrator if no API key or the model call fails.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from backend.chat.schemas import ChatRequest
from backend.shared.settings import get_settings

logger = logging.getLogger(__name__)

_TRAINING_CUES = (
    "train",
    "training",
    "build a model",
    "build model",
    "classifier",
    "regressor",
    "predictive model",
    "predict ",
)

_CLARIFICATION_CUES = (
    "help me",
    "can you help",
    "not sure",
    "unsure",
    "figure out",
    "what should",
    "which model",
    "where do i start",
)

_EXPLICIT_RUN_CUES = (
    "run the pipeline",
    "start training",
    "run training",
    "run the experiment",
    "go ahead",
    "train on",
    "using dataset",
    "use dataset",
)

_DATASET_DETAIL_CUES = (
    "dataset",
    "table",
    ".csv",
    "target",
    "label",
    "column",
)


class _IntentResult(BaseModel):
    route: Literal["training_graph", "orchestrator_chat"] = Field(
        ...,
        description="training_graph = run the LangGraph training pipeline; orchestrator_chat = Q&A / tools only",
    )


def _prefer_orchestrator_for_ambiguous_training_request(message: str) -> bool:
    """
    Keep vague training requests in chat so the assistant can clarify first.

    This is intentionally simple: if the user sounds like they want modeling help
    but has not clearly said "run now on this dataset", prefer the orchestrator.
    """
    lower = (message or "").strip().lower()
    if not lower:
        return False
    if not any(cue in lower for cue in _TRAINING_CUES):
        return False
    if any(cue in lower for cue in _CLARIFICATION_CUES):
        return True
    if any(cue in lower for cue in _EXPLICIT_RUN_CUES):
        return False
    if any(cue in lower for cue in _DATASET_DETAIL_CUES):
        return False
    return True


def should_route_to_training_graph(request: ChatRequest) -> bool:
    """
    Return True to stream generate_graph_sse_events; False for generate_chat_sse.

    resume is handled before this function is called.
    """
    if getattr(request, "force_orchestrator", False):
        return False
    if getattr(request, "background_intake", False):
        return False
    if request.linked_datasets and len(request.linked_datasets) > 0:
        return True

    message = (request.message or "").strip()
    if not message:
        return False
    if _prefer_orchestrator_for_ambiguous_training_request(message):
        return False

    settings = get_settings()
    if not settings.openai_api_key:
        return False

    try:
        return _classify_via_llm(message, request)
    except Exception:
        logger.exception("chat intent router LLM failed; defaulting to orchestrator")
        return False


def _classify_via_llm(message: str, request: ChatRequest) -> bool:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage

    model_id = get_settings().CHAT_INTENT_ROUTER_MODEL
    llm = init_chat_model(model_id, temperature=0)
    structured = llm.with_structured_output(_IntentResult)

    ctx_bits = []
    if request.model_preference:
        ctx_bits.append(f"User preferred model type id: {request.model_preference}")
    ctx = "\n".join(ctx_bits) if ctx_bits else "(none)"

    sys = SystemMessage(
        content=(
            "You route a single user turn for a machine-learning workbench.\n"
            "Choose training_graph if the user wants to RUN the automated training pipeline "
            "(train/fit a model, run the experiment, start pipeline, continue training, "
            "build a classifier/regressor on data, full ML workflow).\n"
            "Choose orchestrator_chat for definitions, explanations, small talk, analysis questions "
            "without starting the pipeline, or when they only ask about available datasets/models.\n"
            "When unsure, prefer orchestrator_chat."
        )
    )
    human = HumanMessage(
        content=f"Context:\n{ctx}\n\nUser message:\n{message}",
    )
    out: _IntentResult = structured.invoke([sys, human])
    return out.route == "training_graph"
