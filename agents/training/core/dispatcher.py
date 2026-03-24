"""
Dispatcher node for the ML Training Agent.

Pure routing function (no LLM call). Reads the current plan and plan_index,
sets ``current_step`` so the conditional edge can route to the correct step
node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.training.utils.graph_stream_hooks import emit_graph_stream

if TYPE_CHECKING:
    from .state import TrainingAgentState


def dispatcher_node(state: "TrainingAgentState") -> "TrainingAgentState":
    """Read the plan and advance ``current_step`` to the next step name.

    If the plan is exhausted, sets ``current_step`` to ``"done"`` so the
    conditional edge routes to ``END``.
    """
    plan = state.get("plan") or []
    plan_index = state.get("plan_index", 0)

    if plan_index >= len(plan):
        return {**state, "current_step": "done"}

    next_step = plan[plan_index]
    step_name = next_step["step"] if isinstance(next_step, dict) else next_step.step

    print(f"[dispatcher] Step {plan_index + 1}/{len(plan)}: {step_name}")
    emit_graph_stream({
        "phase": "dispatch",
        "message": f"Next step {plan_index + 1}/{len(plan)}: {step_name.replace('_', ' ')}",
    })

    return {**state, "current_step": step_name}
