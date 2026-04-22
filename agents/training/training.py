"""Shim for tests and callers that import ``agents.training.training``."""

from agents.training.steps.training import (
    TrainingIteration,
    TrainingResult,
    run_training_agent,
)

__all__ = ["run_training_agent", "TrainingResult", "TrainingIteration"]
