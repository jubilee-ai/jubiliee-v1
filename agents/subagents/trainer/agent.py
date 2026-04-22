"""Trainer subagent entrypoint — same implementation as the training pipeline step."""

from __future__ import annotations

from typing import Any

from agents.training.steps.training import run_training_agent


def run_training_subagent(**kwargs: Any) -> dict[str, Any]:
    """Context-isolated training run (delegates to ``run_training_agent``)."""
    return run_training_agent(**kwargs)
