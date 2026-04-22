"""
Training agent entrypoints.

Execution is handled by the unified LLM tool-calling agent in
``agents.training.agent_simple`` (single executor, no planner/dispatcher loop).
This module keeps a stable import surface for ``invoke_training_agent`` and
``ALL_STEP_NAMES`` for SSE and tests.
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver

from ..agent_simple import UNIFIED_PIPELINE_STEP_NAMES, create_simple_training_agent
from .conversation_context import (
    format_training_user_message,
    normalize_conversation_turns,
)

# Names used by SSE / experiment step tracking (unified executor path).
ALL_STEP_NAMES = sorted(UNIFIED_PIPELINE_STEP_NAMES)


def invoke_training_agent(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
    thread_id: Optional[str] = None,
    conversation: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    Run the unified training agent (non-streaming, HITL off for automation).

    Returns the shared mutable training state dict built by tool wrappers, plus
    ``_thread_id`` when applicable.
    """
    g = (goal or "").strip() or "Training run"
    turns = normalize_conversation_turns(conversation, triggering_message=g)
    user_message = format_training_user_message(g, turns)

    if not thread_id:
        thread_id = f"training-{uuid.uuid4().hex[:8]}"

    config = {"configurable": {"thread_id": thread_id}}
    agent, state = create_simple_training_agent(
        goal=g,
        linked_datasets=linked_datasets,
        user_model_preference=user_model_preference,
        model="openai:gpt-5.4",
        hitl=False,
        checkpointer=MemorySaver(),
        use_external_sources=False,
    )
    agent.invoke(
        {"messages": [{"role": "user", "content": user_message}]},
        config=config,
    )
    return {**state, "_thread_id": thread_id}


def resume_training_agent(
    decision: Any,
    thread_id: str,
) -> dict[str, Any]:
    """
    Programmatic resume is not supported for the unified agent: checkpointed
    state is tied to a specific compiled agent instance. Use chat SSE resume
    with the stored agent (``generate_graph_resume_sse_events`` / simple resume).
    """
    raise NotImplementedError(
        "resume_training_agent() is not supported for the unified executor; "
        "use SSE resume with a persisted training thread."
    )


def create_training_agent(checkpointer=None):
    """
    Deprecated. Kept so tests can ``patch`` this symbol; production code uses
    :func:`create_simple_training_agent`.
    """
    raise RuntimeError(
        "create_training_agent() was removed; patch or import "
        "agents.training.agent_simple.create_simple_training_agent instead."
    )
