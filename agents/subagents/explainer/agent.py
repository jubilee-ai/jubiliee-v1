"""Explainer subagent: plain-language narrative from evaluator JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from agents.subagents.explainer.schemas import ExplanationReport

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def run_explainer_subagent(ctx: dict[str, Any], llm_model: Optional[str] = None) -> dict[str, Any]:
    """Turn evaluator output into user-facing markdown (isolated context)."""
    model = llm_model or "openai:gpt-5-mini"
    skill_path = Path(__file__).parent / "SKILL.md"
    sys_prompt = skill_path.read_text(encoding="utf-8") if skill_path.exists() else "You explain model results clearly."
    agent = create_agent(
        model=init_chat_model(model),
        tools=[],
        system_prompt=sys_prompt,
        response_format=ToolStrategy(ExplanationReport),
    )
    body = (
        "EVALUATION JSON FROM THE EVALUATOR SUBAGENT:\n"
        f"{json.dumps(ctx.get('evaluation') or ctx, default=str)[:24_000]}\n"
        "Produce an ExplanationReport."
    )
    r = agent.invoke({"messages": [{"role": "user", "content": body}]})
    rep = r.get("structured_response")
    if rep is not None and hasattr(rep, "model_dump"):
        return rep.model_dump()
    return ExplanationReport(summary_markdown=str(ctx)[:4000]).model_dump()
