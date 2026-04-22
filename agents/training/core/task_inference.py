"""Task-type inference — one source of truth for classification / regression / unsupervised.

Two entry points:

- :func:`infer_task_type` — heuristic inference from goal text + model/estimator names.
- :func:`infer_supervised_task_type_from_target_column` — data-driven inference from
  the actual target column using ``sklearn.utils.multiclass.type_of_target``.

Prefer the data-driven variant when a training frame and target column are available;
fall back to the heuristic variant for early-stage planning before data is loaded.
"""

from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from sklearn.utils.multiclass import type_of_target

TaskType = Literal["classification", "regression", "unsupervised"]


_UNSUPERVISED_CUES = (
    "unsupervised",
    "cluster",
    "clustering",
    "segmentation",
    "anomaly",
    "outlier",
    "dimensionality reduction",
    "pca",
)

_GOAL_REGRESSION_CUES = (
    "regress",
    "predict value",
    "forecast",
    "amount",
    "price",
    "cost",
    "salary",
    "revenue",
    "income",
    "continuous",
)

_MODEL_REGRESSION_CUES = ("regress", "continuous", "numeric")


def infer_task_type(
    goal: str = "",
    *,
    selected_model: Optional[str] = None,
    estimator_hint: Optional[str] = None,
) -> TaskType:
    """Heuristic task-type inference from goal + model metadata (no data inspection).

    Resolution order:

    1. ``selected_model == "unsupervised"`` → unsupervised.
    2. Unsupervised cue words in any field → unsupervised.
    3. Regression cues in the model name (``regress``/``continuous``/``numeric``)
       when the model is not specifically ``logistic``.
    4. Regression cues in the goal text.
    5. ``glm`` / ``regression`` in the model name (non-logistic).
    6. ``regress`` in the estimator hint.
    7. Default → classification.
    """
    goal_lower = (goal or "").lower()
    estimator_lower = (estimator_hint or "").lower()
    model_lower = (selected_model or "").lower()

    if selected_model == "unsupervised":
        return "unsupervised"

    combined = f"{goal_lower} {estimator_lower} {model_lower}"
    if any(cue in combined for cue in _UNSUPERVISED_CUES):
        return "unsupervised"

    if "logistic" not in model_lower and any(w in model_lower for w in _MODEL_REGRESSION_CUES):
        return "regression"

    if any(w in goal_lower for w in _GOAL_REGRESSION_CUES):
        return "regression"

    if any(w in model_lower for w in ("glm", "regression")) and "logistic" not in model_lower:
        return "regression"

    if "regress" in estimator_lower:
        return "regression"

    return "classification"


def infer_supervised_task_type_from_target_column(
    df: pd.DataFrame, target_column: str
) -> Optional[Literal["classification", "regression"]]:
    """Use sklearn's ``type_of_target`` on the training label column.

    Resolves mismatches when model-selection heuristics say "classification"
    but the target is clearly continuous (e.g. insurance ``charges``).
    """
    if not target_column or target_column not in df.columns:
        return None
    y = df[target_column].dropna()
    if len(y) == 0:
        return None
    try:
        tt = type_of_target(y)
    except Exception:
        return None
    if tt in ("continuous", "continuous-multioutput"):
        return "regression"
    if tt in ("binary", "multiclass", "multiclass-multioutput", "multilabel-indicator"):
        return "classification"
    return None


__all__ = [
    "TaskType",
    "infer_task_type",
    "infer_supervised_task_type_from_target_column",
]
