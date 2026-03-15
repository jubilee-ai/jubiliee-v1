"""
Conditional edge functions for the agentic ML Training Agent graph.

The planner + evaluator architecture only needs two routing functions:
  - ``route_to_step``: dispatcher → step node
  - ``should_continue``: evaluator → dispatcher | planner | END
"""

from typing import Literal

from .state import TrainingAgentState


def route_to_step(state: TrainingAgentState) -> str:
    """Route from the dispatcher to the next step node.

    Returns ``state["current_step"]`` which was set by the dispatcher.
    When the plan is exhausted the dispatcher sets this to ``"done"``.
    """
    return state.get("current_step", "done")


def should_continue(
    state: TrainingAgentState,
) -> Literal["continue", "replan", "done"]:
    """Route from the evaluator to the next phase.

    - ``"continue"`` → back to dispatcher (next step in plan)
    - ``"replan"``   → back to planner for a new plan
    - ``"done"``     → END
    """
    decision = state.get("evaluator_decision", "done")
    if decision in ("continue", "replan", "done"):
        return decision
    return "done"
