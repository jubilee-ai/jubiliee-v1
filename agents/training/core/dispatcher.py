"""
Dispatcher node for the ML Training Agent.

Pure routing function (no LLM call). Reads the current plan and plan_index,
sets ``current_step`` so the conditional edge can route to the correct step
node.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents.training.utils.graph_stream_hooks import emit_graph_stream
from .state import STEP_PREREQUISITES, canonical_step_name

if TYPE_CHECKING:
    from .state import TrainingAgentState


def _plan_entry_step(entry: object) -> str:
    step_name = entry["step"] if isinstance(entry, dict) else entry.step  # type: ignore[index]
    return canonical_step_name(step_name)


def _ensure_data_collection_before_cleaning(
    state: "TrainingAgentState", plan: list, plan_index: int, step_name: str
) -> tuple[list, str, bool]:
    """Planner sometimes orders ``cleaning`` before ``data_collection``. Without a collected
    dataset, cleaning is a no-op and corrupts downstream summaries. Swap in a later
    ``data_collection`` step or insert one at the current index.
    """
    if step_name != "cleaning" or state.get("collected_dataset_ref"):
        return plan, step_name, False
    new_plan = list(plan)
    n = len(new_plan)
    for j in range(plan_index + 1, n):
        if _plan_entry_step(new_plan[j]) == "data_collection":
            new_plan[plan_index], new_plan[j] = new_plan[j], new_plan[plan_index]
            return new_plan, _plan_entry_step(new_plan[plan_index]), True
    insert: dict = {
        "step": "data_collection",
        "rationale": "Dataset must be loaded before cleaning (auto-inserted).",
    }
    new_plan.insert(plan_index, insert)
    return new_plan, "data_collection", True


def _ensure_prerequisites(
    state: "TrainingAgentState", plan: list, plan_index: int, step_name: str
) -> tuple[list, str, bool]:
    """Check ``STEP_PREREQUISITES`` and auto-insert a missing provider step
    when the current step depends on a state key that hasn't been produced yet.
    """
    for prereq in STEP_PREREQUISITES:
        if step_name not in prereq["dependents"]:
            continue
        if state.get(prereq["state_key"]):
            continue
        skip_model = prereq.get("skip_when_model")
        if skip_model and state.get("selected_model") == skip_model:
            continue
        new_plan = list(plan)
        new_plan.insert(plan_index, {
            "step": prereq["provider"],
            "rationale": f"Auto-inserted: {prereq['state_key']} required.",
        })
        return new_plan, prereq["provider"], True
    return plan, step_name, False


def dispatcher_node(state: "TrainingAgentState") -> "TrainingAgentState":
    """Read the plan and advance ``current_step`` to the next step name.

    If the plan is exhausted, sets ``current_step`` to ``"done"`` so the
    conditional edge routes to ``END``.
    """
    plan = list(state.get("plan") or [])
    plan_index = state.get("plan_index", 0)

    if plan_index >= len(plan):
        return {**state, "current_step": "done"}

    next_step = plan[plan_index]
    step_name = _plan_entry_step(next_step)
    plan, step_name, plan_mutated = _ensure_data_collection_before_cleaning(state, plan, plan_index, step_name)
    plan, step_name, mutated2 = _ensure_prerequisites(state, plan, plan_index, step_name)
    plan_mutated = plan_mutated or mutated2

    print(f"[dispatcher] Step {plan_index + 1}/{len(plan)}: {step_name}")
    emit_graph_stream({
        "phase": "dispatch",
        "message": f"Next step {plan_index + 1}/{len(plan)}: {step_name.replace('_', ' ')}",
    })

    out = {**state, "current_step": step_name}
    if plan_mutated:
        out["plan"] = plan
    return out
