"""Evaluator subagent: holdout metrics + structured narrative."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from agents.subagents.evaluator.schemas import EvaluationReport

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def run_evaluator_subagent(ctx: dict[str, Any], llm_model: Optional[str] = None) -> dict[str, Any]:
    """Compute test metrics and return a structured EvaluationReport (isolated LLM context)."""
    from agents.training.steps.training import _evaluate_model_on_test

    metrics = _evaluate_model_on_test(
        str(ctx.get("model_name", "")),
        str(ctx.get("test_ref", "")),
        str(ctx.get("target_column", "")),
        str(ctx.get("task_type", "classification")),
    )
    model = llm_model or "openai:gpt-5-mini"
    skill_path = Path(__file__).parent / "SKILL.md"
    sys_prompt = (
        (skill_path.read_text(encoding="utf-8") if skill_path.exists() else "")
        + "\n\nYou receive numeric metrics. Fill EvaluationReport fields honestly; "
        "if calibration/fairness were not computed, leave them null/empty."
    )
    agent = create_agent(
        model=init_chat_model(model),
        tools=[],
        system_prompt=sys_prompt,
        response_format=ToolStrategy(EvaluationReport),
    )
    body = (
        "HOLDOUT METRICS (JSON):\n"
        f"{json.dumps(metrics, default=str)}\n\n"
        "Return an EvaluationReport. recommendation should be 2-4 sentences for a PM."
    )
    r = agent.invoke({"messages": [{"role": "user", "content": body}]})
    rep = r.get("structured_response")
    if rep is not None and hasattr(rep, "model_dump"):
        d = rep.model_dump()
        d.setdefault("core_metrics", metrics or {})
        return d
    return EvaluationReport(core_metrics=metrics or {}, recommendation="See core_metrics.").model_dump()
