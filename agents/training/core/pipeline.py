"""Canonical names and recap text for the unified training pipeline.

One source of truth used by:
- ``agents/training/agent_simple.py`` (the Deep agent tool wrappers)
- ``backend/training/service.py`` (SSE routing / step filtering)
- ``agent.py`` (``propose_training_plan`` default recap)
"""

from __future__ import annotations

from typing import Final

#: Step names emitted to the UI / SSE from the unified executor path.
UNIFIED_PIPELINE_STEP_NAMES: Final[frozenset[str]] = frozenset(
    {
        "data_collection",
        "select_model",
        "cleaning",
        "label_split_definition",
        "feature_specification_and_engineering",
        "evaluate_models",
        "training_approval",
        "training",
        "generate_report",
    }
)


#: Short bullets describing what the training pipeline will do — shown on the
#: UI plan card when the user or planner does not override ``recap_steps``.
DEFAULT_TRAINING_RECAP: Final[tuple[str, ...]] = (
    "Load and validate the dataset",
    "Choose model family, clean and standardize columns",
    "Define target, splits, and features",
    "Train, evaluate, and generate the audit report",
)


__all__ = ["UNIFIED_PIPELINE_STEP_NAMES", "DEFAULT_TRAINING_RECAP"]
