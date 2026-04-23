"""Trainer subagent entrypoint — same implementation as the training pipeline step."""

from __future__ import annotations

from typing import Any

from backend.shared.settings import get_settings

from agents.training.steps.h2o_training import run_h2o_training
from agents.training.steps.training import run_training_agent


def run_training_subagent(**kwargs: Any) -> dict[str, Any]:
    """Context-isolated training run (H2O AutoML or full LLM agent per settings)."""
    if get_settings().TRAINING_USE_H2O_ONLY:
        return run_h2o_training(**kwargs)
    return run_training_agent(**kwargs)
