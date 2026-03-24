"""
HTTP-layer intent: training graph SSE vs orchestrator chat SSE for POST /api/chat.

Hard signals (no LLM): linked_datasets, mode=train / mode=chat, resume_training (handled in chat()).

When the user did not force a mode and did not link datasets, an LLM picks training_graph vs
orchestrator_chat from the natural-language message. Falls back to orchestrator if no API key
or the model call fails.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from backend.chat.schemas import ChatRequest
from backend.shared.settings import get_settings

logger = logging.getLogger(__name__)


class _IntentResult(BaseModel):
    route: Literal["training_graph", "orchestrator_chat"] = Field(
        ...,
        description="training_graph = run the LangGraph training pipeline; orchestrator_chat = Q&A / tools only",
    )


def should_route_to_training_graph(request: ChatRequest) -> bool:
    """
    Return True to stream generate_graph_sse_events; False for generate_chat_sse.

    resume_training is handled before this function is called.
    """
    if request.linked_datasets and len(request.linked_datasets) > 0:
        return True
    if request.mode == "train":
        return True
    if request.mode == "chat":
        return False

    message = (request.message or "").strip()
    if not message:
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
    if request.user_model_preference:
        ctx_bits.append(f"User preferred model type id: {request.user_model_preference}")
    if request.training_context:
        tc = request.training_context.strip()
        if len(tc) > 800:
            tc = tc[:800] + "…"
        ctx_bits.append(f"Training context excerpt:\n{tc}")
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
