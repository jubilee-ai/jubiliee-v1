"""Data analyst: wraps the existing data collection agent."""

from __future__ import annotations

from typing import Any, Optional

from agents.subagents.data_analyst.schemas import DataAnalysisResult
from agents.training.steps.data_collection import run_data_collection


def run_data_analyst_subagent(
    goal: str,
    selected_model: Optional[str] = None,
    linked_datasets: Optional[list[str]] = None,
    llm_model: Optional[str] = None,
) -> dict[str, Any]:
    out = run_data_collection(
        goal=goal,
        selected_model=selected_model,
        linked_datasets=linked_datasets,
        model=llm_model or "openai:gpt-5-mini",
    )
    return DataAnalysisResult(
        primary_dataset_ref=out.get("dataset_ref"),
        description=out.get("description"),
        schema_notes="See pipeline for train/val/test after label_and_split.",
        issues=[],
    ).model_dump()
