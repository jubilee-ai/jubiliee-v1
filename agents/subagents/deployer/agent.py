"""Deployer: registry aliases + serve command + smoke test (agent composes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model

from agents.subagents.deployer.schemas import DeploymentDecision
from agents.training.mcp.mlflow_server import (
    mlflow_models_serve_command,
    mlflow_set_model_alias,
    smoke_test_json_endpoint_tool,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def run_deployer_subagent(ctx: dict[str, Any], llm_model: Optional[str] = None) -> dict[str, Any]:
    """Use tools to set alias / build serve command; LLM returns DeploymentDecision."""
    model = llm_model or "openai:gpt-5-mini"
    skill = Path(__file__).parent / "SKILL.md"
    sys_prompt = skill.read_text(encoding="utf-8") if skill.exists() else "You are a release engineer."
    tools = [mlflow_set_model_alias, mlflow_models_serve_command, smoke_test_json_endpoint_tool]
    agent = create_agent(
        model=init_chat_model(model),
        tools=tools,
        system_prompt=sys_prompt + "\n\nContext JSON:\n" + json.dumps(ctx, default=str)[:8000],
        response_format=ToolStrategy(DeploymentDecision),
    )
    r = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Decide promotion steps. Use tools as needed. "
                        "If touching production, set requires_human_approval=True."
                    ),
                }
            ]
        }
    )
    rep = r.get("structured_response")
    if rep is not None and hasattr(rep, "model_dump"):
        return rep.model_dump()
    return DeploymentDecision(reason="no structured output").model_dump()
