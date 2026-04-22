"""
Orchestrator (Jubilee) tools that invoke context-isolated subagents.

Each tool returns JSON text for the main chat model to consume.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class InvokeDataAnalystInput(BaseModel):
    goal: str = Field(description="What data is needed and for what modeling purpose")
    selected_model: Optional[str] = Field(default=None, description="Optional model family hint")
    linked_datasets_json: str = Field(
        default="[]",
        description='JSON list of dataset refs to prefer, e.g. ["sales_clean"]',
    )


@tool(args_schema=InvokeDataAnalystInput)
def invoke_data_analyst(
    goal: str,
    selected_model: Optional[str] = None,
    linked_datasets_json: str = "[]",
) -> str:
    """Run the data analyst subagent (dataset discovery / collection)."""
    from agents.subagents.data_analyst.agent import run_data_analyst_subagent

    try:
        linked = json.loads(linked_datasets_json or "[]")
    except json.JSONDecodeError:
        linked = []
    if not isinstance(linked, list):
        linked = []
    out = run_data_analyst_subagent(goal=goal, selected_model=selected_model, linked_datasets=linked)
    return json.dumps({"subagent": "data_analyst", "result": out}, ensure_ascii=False)


class InvokeFeatureEngineerInput(BaseModel):
    context_json: str = Field(
        description=(
            "JSON object with keys: dataset_ref, goal, target_column, grain (optional), "
            "task_type (classification|regression), forbidden_columns (optional list), "
            "as_of_cutoff, prediction_horizon, max_iterations"
        )
    )


@tool(args_schema=InvokeFeatureEngineerInput)
def invoke_feature_engineer(context_json: str) -> str:
    """Run the feature engineer subagent (feature spec + validation)."""
    from agents.subagents.feature_engineer.agent import run_feature_engineer_subagent

    ctx = json.loads(context_json or "{}")
    if not isinstance(ctx, dict):
        return json.dumps({"error": "context_json must be a JSON object"})
    out = run_feature_engineer_subagent(ctx)
    return json.dumps({"subagent": "feature_engineer", "result": out}, default=str)[:80_000]


class InvokeTrainerInput(BaseModel):
    context_json: str = Field(
        description="JSON object with the same kwargs as run_training_agent (train_ref, val_ref, ...)"
    )


@tool(args_schema=InvokeTrainerInput)
def invoke_trainer(context_json: str) -> str:
    """Run the trainer subagent (full training loop). Can take minutes and many LLM calls."""
    from agents.subagents.trainer.agent import run_training_subagent

    ctx = json.loads(context_json or "{}")
    if not isinstance(ctx, dict):
        return json.dumps({"error": "context_json must be a JSON object"})
    out = run_training_subagent(**ctx)
    return json.dumps({"subagent": "trainer", "result": out}, default=str)[:120_000]


class InvokeEvaluatorInput(BaseModel):
    model_name: str
    test_ref: str
    target_column: str
    task_type: str = Field(default="classification")


@tool(args_schema=InvokeEvaluatorInput)
def invoke_evaluator(
    model_name: str,
    test_ref: str,
    target_column: str,
    task_type: str = "classification",
) -> str:
    """Run evaluator subagent (holdout metrics + structured report)."""
    from agents.subagents.evaluator.agent import run_evaluator_subagent

    out = run_evaluator_subagent(
        {"model_name": model_name, "test_ref": test_ref, "target_column": target_column, "task_type": task_type}
    )
    return json.dumps({"subagent": "evaluator", "result": out}, ensure_ascii=False)


class InvokeExplainerInput(BaseModel):
    evaluation_json: str = Field(description="JSON from invoke_evaluator (or equivalent)")


@tool(args_schema=InvokeExplainerInput)
def invoke_explainer(evaluation_json: str) -> str:
    """Run explainer subagent on evaluator output."""
    from agents.subagents.explainer.agent import run_explainer_subagent

    try:
        ev = json.loads(evaluation_json or "{}")
    except json.JSONDecodeError:
        ev = {"raw": evaluation_json}
    out = run_explainer_subagent({"evaluation": ev})
    return json.dumps({"subagent": "explainer", "result": out}, ensure_ascii=False)


class InvokeDeployerInput(BaseModel):
    context_json: str = Field(
        default="{}",
        description="JSON with keys like registered_model_name, version, alias, endpoint (optional)",
    )


@tool(args_schema=InvokeDeployerInput)
def invoke_deployer(context_json: str = "{}") -> str:
    """Run deployer subagent (aliases, serve command, smoke tests)."""
    from agents.subagents.deployer.agent import run_deployer_subagent

    ctx = json.loads(context_json or "{}")
    if not isinstance(ctx, dict):
        ctx = {}
    out = run_deployer_subagent(ctx)
    return json.dumps({"subagent": "deployer", "result": out}, ensure_ascii=False)


class InvokeMonitorInput(BaseModel):
    context_json: str = Field(
        default="{}",
        description="JSON with reference_dataset_ref, current_dataset_ref, optional prediction_column/target_column",
    )


@tool(args_schema=InvokeMonitorInput)
def invoke_monitor(context_json: str = "{}") -> str:
    """Run monitor subagent (drift / performance / retrain request)."""
    from agents.subagents.monitor.agent import run_monitor_subagent

    ctx = json.loads(context_json or "{}")
    if not isinstance(ctx, dict):
        ctx = {}
    out = run_monitor_subagent(ctx)
    return json.dumps({"subagent": "monitor", "result": out}, ensure_ascii=False)


ORCHESTRATOR_SUBAGENT_TOOLS = [
    invoke_data_analyst,
    invoke_feature_engineer,
    invoke_trainer,
    invoke_evaluator,
    invoke_explainer,
    invoke_deployer,
    invoke_monitor,
]
