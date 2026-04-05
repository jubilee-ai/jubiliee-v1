"""
Human-in-the-loop (HITL) utilities for the ML Training Agent.
"""

from typing import Any, Callable, Optional

from langgraph.types import interrupt

from .state import STATE_SNAPSHOT_KEYS, TrainingAgentState

# Pending work results keyed by "{thread_id}::{node_name}" so a LangGraph node replay
# after interrupt() can skip duplicate work_fn side effects (see run_with_hitl).
_HITL_WORK_CACHE: dict[str, dict[str, Any]] = {}


def _hitl_memory_key(node_name: str) -> Optional[str]:
    """Stable key for the current graph thread, or None if not in a runnable context."""
    try:
        from langgraph.config import get_config

        cfg = get_config() or {}
        tid = (cfg.get("configurable") or {}).get("thread_id")
        if tid:
            return f"{tid}::{node_name}"
    except Exception:
        pass
    return None


def _invalidate_split_downstream(state: TrainingAgentState) -> TrainingAgentState:
    """Clear label/split and feature outputs so later steps cannot use stale refs."""
    out = dict(state)
    for k in (
        "label_definition",
        "split_indices",
        "train_dataset_ref",
        "val_dataset_ref",
        "test_dataset_ref",
        "transformed_train_ref",
        "transformed_val_ref",
        "transformed_test_ref",
        "transformed_dataset_ref",
        "feature_spec",
        "feature_validation_passed",
        "experiment_result",
        "feature_rankings",
        "experiment_grid_summary",
    ):
        out[k] = None
    out["feature_pipeline_mode"] = None
    return out  # type: ignore[return-value]


# =============================================================================
# SERIALIZATION HELPERS
# =============================================================================


def make_serializable(obj: Any) -> Any:
    """Convert numpy/pandas types to Python native types for serialization."""
    import numpy as np

    if obj is None:
        return None
    elif isinstance(obj, (np.bool_, np.integer)):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Try to convert to string as last resort
        try:
            return str(obj)
        except Exception:
            return None


# =============================================================================
# DECISION PARSING (extracted from duplication)
# =============================================================================


def parse_decision(decision: Any) -> tuple[bool, Optional[str]]:
    """
    Parse a HITL decision into (approved, feedback).

    Accepts:
        - True or False: Direct boolean
        - "yes"/"y"/"ok"/"approve"/"approved"/"continue": Approve
        - Any other string: Reject with that string as feedback
        - {"approved": bool, "feedback": str}: Dict format

    Returns:
        Tuple of (approved: bool, feedback: Optional[str])
    """
    if isinstance(decision, bool):
        return decision, None
    elif isinstance(decision, str):
        # String response - "yes"/"y" means approved, anything else is feedback
        if decision.lower() in ["yes", "y", "ok", "approve", "approved", "continue"]:
            return True, None
        else:
            return False, decision
    elif isinstance(decision, dict):
        approved = decision.get("approved", True)
        feedback = decision.get("feedback")
        return approved, feedback
    else:
        # Default to approved if unclear
        return True, None


# =============================================================================
# STATE SNAPSHOT EXTRACTION
# =============================================================================


def extract_state_snapshot(
    state: dict[str, Any], keys: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Extract a serializable snapshot from state with only key fields.

    Args:
        state: Full state dict
        keys: Optional list of keys to extract. Defaults to STATE_SNAPSHOT_KEYS.

    Returns:
        Dict with only the specified keys that exist in state
    """
    if keys is None:
        keys = STATE_SNAPSHOT_KEYS

    return {k: state.get(k) for k in keys if k in state}


# =============================================================================
# HITL WRAPPER
# =============================================================================


def run_with_hitl(
    node_name: str,
    state: TrainingAgentState,
    work_fn: Callable[[TrainingAgentState, Optional[str]], TrainingAgentState],
    get_summary_fn: Optional[Callable[[TrainingAgentState], str]] = None,
) -> TrainingAgentState:
    """
    Run node work with human-in-the-loop approval.

    After the work completes, pauses for human review. User can:
    - Approve: continue to next node
    - Reject with feedback: re-run the node with the feedback

    Args:
        node_name: Name of the node (for display)
        state: Current agent state
        work_fn: Function that takes (state, feedback) and returns updated state
        get_summary_fn: Optional function to create human-readable summary from result

    Returns:
        Updated state after human approval
    """
    feedback = None

    while True:
        mem_key = _hitl_memory_key(node_name)
        used_cache = bool(
            mem_key and mem_key in _HITL_WORK_CACHE and feedback is None
        )
        if used_cache:
            result = dict(_HITL_WORK_CACHE[mem_key])
        else:
            try:
                result = work_fn(state, feedback)
            except Exception as e:
                # Merge into state so the graph can route to the evaluator with error set
                result = {**state, "error": str(e), "hitl_error_node": node_name}
                if node_name == "label_split_definition":
                    result = _invalidate_split_downstream(result)

            # Sanitize the entire result to ensure all values are serializable
            # This is critical for LangGraph's checkpointer (msgpack)
            result = {k: make_serializable(v) for k, v in result.items()}

            if not result.get("hitl_auto_approve") and not state.get(
                "hitl_auto_approve"
            ):
                if mem_key:
                    _HITL_WORK_CACHE[mem_key] = dict(result)

        if result.get("hitl_auto_approve") or state.get("hitl_auto_approve"):
            if mem_key:
                _HITL_WORK_CACHE.pop(mem_key, None)
            return result

        if get_summary_fn:
            try:
                summary = get_summary_fn(result)
            except Exception as e:
                summary = (
                    f"Node '{node_name}': could not build summary ({type(e).__name__}: {e})"
                )
        else:
            summary = f"Node '{node_name}' completed successfully."

        # Build state snapshot with only key fields
        state_snapshot = extract_state_snapshot(result)

        # Interrupt for human review
        decision = interrupt(
            {
                "node": node_name,
                "summary": summary,
                "message": "Approve to continue, or provide feedback to redo.",
                "state_snapshot": state_snapshot,
            }
        )

        # Parse the decision using shared helper
        approved, new_feedback = parse_decision(decision)

        if approved:
            if mem_key:
                _HITL_WORK_CACHE.pop(mem_key, None)
            return result
        else:
            # User rejected - loop will redo with feedback
            if mem_key:
                _HITL_WORK_CACHE.pop(mem_key, None)
            feedback = new_feedback if new_feedback else "Please redo this step."
            print(f"[HITL] Node '{node_name}' rejected. Feedback: {feedback}")
