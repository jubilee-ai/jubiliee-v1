"""
Shared prerequisite checks for training tools (logic formerly enforced by the
graph dispatcher). Tool wrappers should call these to keep execution safe when
the LLM chooses steps out of strict order.
"""

from __future__ import annotations

from typing import Any

from agents.training.core.state import STEP_PREREQUISITES


def has_collected_dataset(state: dict[str, Any]) -> bool:
    return bool(state.get("collected_dataset_ref"))


def cleaning_blocked_without_data(state: dict[str, Any]) -> str | None:
    """Return a user-facing reason if cleaning should not run yet, else None.

    The returned string starts with ``SKIP:`` so the SSE stream and test harness
    can uniformly treat guard messages as skip-style tool outputs (see
    ``_should_skip_tool_message`` in ``backend/training/service.py``).
    """
    if has_collected_dataset(state):
        return None
    return (
        "SKIP: Cannot run cleaning — no dataset loaded yet. "
        "Call data_collection first to load or register data."
    )


def missing_prerequisite_for_step(
    step_name: str, state: dict[str, Any]
) -> str | None:
    """If ``step_name`` requires state that is not set, return SKIP-prefixed guidance."""
    selected = (state.get("selected_model") or "").strip().lower()
    for prereq in STEP_PREREQUISITES:
        if step_name not in prereq["dependents"]:
            continue
        if state.get(prereq["state_key"]):
            continue
        skip_model = prereq.get("skip_when_model")
        if skip_model and selected == skip_model:
            continue
        provider = prereq["provider"]
        return (
            f"SKIP: Cannot run {step_name} yet — need {prereq['state_key']} "
            f"from {provider} first."
        )
    return None
