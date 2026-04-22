"""Feature engineer: wraps ``run_feature_engineering``."""

from __future__ import annotations

from typing import Any, Optional

from agents.training.steps.feature_engineering import run_feature_engineering


def run_feature_engineer_subagent(ctx: dict[str, Any], llm_model: Optional[str] = None) -> dict[str, Any]:
    return run_feature_engineering(
        dataset_ref=str(ctx["dataset_ref"]),
        goal=str(ctx.get("goal", "")),
        target_column=str(ctx["target_column"]),
        grain=str(ctx.get("grain", "row")),
        task_type=ctx.get("task_type", "classification"),
        forbidden_columns=ctx.get("forbidden_columns"),
        as_of_cutoff=ctx.get("as_of_cutoff"),
        prediction_horizon=ctx.get("prediction_horizon"),
        model=llm_model or "openai:gpt-5-mini",
        max_iterations=int(ctx.get("max_iterations", 10)),
    )
