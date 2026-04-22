"""Monitor: drift tools + optional retrain request."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from agents.subagents.monitor.schemas import MonitoringReport
from agents.training.mcp.drift_server import (
    evidently_data_drift_tool,
    evidently_model_performance_tool,
    open_retrain_request_tool,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def run_monitor_subagent(ctx: dict[str, Any], llm_model: Optional[str] = None) -> dict[str, Any]:
    """Run drift/performance tools; LLM fills MonitoringReport."""
    model = llm_model or "openai:gpt-5-mini"
    skill = Path(__file__).parent / "SKILL.md"
    sys_prompt = skill.read_text(encoding="utf-8") if skill.exists() else "You monitor ML models."
    tools = [evidently_data_drift_tool, evidently_model_performance_tool, open_retrain_request_tool]
    agent = create_agent(
        model=init_chat_model(model),
        tools=tools,
        system_prompt=sys_prompt + "\n\nContext:\n" + json.dumps(ctx, default=str)[:6000],
        response_format=ToolStrategy(MonitoringReport),
    )
    r = agent.invoke({"messages": [{"role": "user", "content": "Run checks and return MonitoringReport."}]})
    rep = r.get("structured_response")
    if rep is not None and hasattr(rep, "model_dump"):
        return rep.model_dump()
    return MonitoringReport(recommendation="no structured output").model_dump()
