"""
Main Orchestrator Agent — Jubilee AI Chatbot

Uses langchain's create_agent to run a conversational loop that can spin up
two specialised sub-agents on demand:

  1. Analysis sub-agent  (agents/analysis_agent_v2)
     → statistical analysis, pretrained-model inference, data exploration

  2. Training sub-agent  (agents/training/agent_simple)
     → full ML pipeline: model selection → data collection → cleaning →
       label/split → feature engineering → training → report
"""

import json
import uuid
from typing import Any, Optional

from dotenv import load_dotenv
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path / env bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-5.1", temperature=0)


# ============================================================================
# Tool 1 — Data Analysis (delegates to analysis_agent_v2)
# ============================================================================

class AnalyzeDataInput(BaseModel):
    question: str = Field(
        description="The analytical question to answer (e.g. 'What is the distribution of income in the loan dataset?')"
    )


@tool(args_schema=AnalyzeDataInput)
def analyze_data(question: str) -> str:
    """Run the data-analysis sub-agent to answer a question that requires
    data exploration, statistical analysis, or pretrained-model inference.

    USE FOR:
    - Statistical profiling, correlations, trends
    - Pretrained model inference (sentiment, credit-risk scoring, forecasting …)
    - Dataset exploration and lookup

    DO NOT USE FOR:
    - Training custom models  → use train_model
    """
    from agents.analysis_agent_v2 import agent as analysis_agent

    result = analysis_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "Analysis completed but no response was generated."


# ============================================================================
# Tool 2 — Model Training (delegates to the training LangGraph pipeline)
# ============================================================================

class TrainModelInput(BaseModel):
    goal: str = Field(
        description="Training objective (e.g. 'Train a loan-default prediction model on the Loan_default dataset')"
    )
    linked_datasets: Optional[list[str]] = Field(
        default=None,
        description="Dataset references the agent should use (e.g. ['csv_Loan_default'])",
    )
    model_preference: Optional[str] = Field(
        default=None,
        description="Preferred model family: supervised | unsupervised | neural_networks",
    )


_last_training_state: dict[str, Any] = {}


def _run_training_to_completion(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_preference: Optional[str],
) -> dict[str, Any]:
    """Run the simple training agent end-to-end without HITL.

    Stores the final shared-state dict in ``_last_training_state`` so the SSE
    chat handler can generate step-level events for the frontend.
    """
    from agents.training.agent_simple import create_simple_training_agent

    thread_id = f"orch-{uuid.uuid4().hex[:8]}"
    agent, shared_state = create_simple_training_agent(
        goal=goal,
        linked_datasets=linked_datasets,
        user_model_preference=model_preference,
        hitl=False,
    )

    config = {"configurable": {"thread_id": thread_id}}
    agent.invoke({"messages": [{"role": "user", "content": goal}]}, config=config)

    final_state = dict(shared_state)

    _last_training_state.clear()
    _last_training_state.update(final_state)

    return final_state


@tool(args_schema=TrainModelInput)
def train_model(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    model_preference: Optional[str] = None,
) -> str:
    """Train a new ML model using the training sub-agent.

    Runs the full pipeline: model selection → data collection → cleaning →
    label/split definition → feature engineering → training → evaluation → report.

    This tool may take several minutes. Use it only after confirming with the
    user that they want to proceed with training.
    """
    try:
        result = _run_training_to_completion(goal, linked_datasets, model_preference)

        selected_model = result.get("selected_model", "unknown")
        model_explanation = result.get("model_explanation", "")
        metrics = result.get("training_metrics", {})
        report = result.get("report_path", "")
        model_weights = result.get("model_weights_path", "")

        if not metrics.get("success", True) and not model_weights:
            return f"Training failed. Check the report for details: {report}"

        parts = [
            "Training completed successfully!",
            f"  Model family : {selected_model}",
            f"  Explanation  : {model_explanation}" if model_explanation else None,
            f"  Model name   : {metrics.get('model_name', 'N/A')}",
            f"  Model type   : {metrics.get('model_type', 'N/A')}",
        ]

        for key, label in [
            ("val_accuracy", "Val Accuracy"),
            ("val_roc_auc", "Val ROC-AUC"),
            ("test_accuracy", "Test Accuracy"),
            ("test_roc_auc", "Test ROC-AUC"),
            ("val_r2", "Val R²"),
            ("test_r2", "Test R²"),
            ("val_rmse", "Val RMSE"),
            ("test_rmse", "Test RMSE"),
        ]:
            v = metrics.get(key)
            if v is not None:
                parts.append(f"  {label:14s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {label:14s}: {v}")

        if metrics.get("summary"):
            parts.append(f"\n  Summary: {metrics['summary']}")
        if metrics.get("recommendations"):
            parts.append(f"  Recommendations: {metrics['recommendations']}")
        if report:
            parts.append(f"\n  Report: {report}")

        return "\n".join(p for p in parts if p is not None)

    except Exception as exc:
        return f"Training failed: {type(exc).__name__}: {exc}"


# ============================================================================
# System Prompt
# ============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are **Jubilee**, an AI assistant for data analysis and machine learning.

## Available Tools

| Tool | When to use |
|---|---|
| `analyze_data` | Analytical questions: statistics, trends, pretrained-model inference, data exploration |
| `train_model` | User explicitly wants to train / build a custom ML model |

## Decision Flow

1. **General / conversational question** → answer directly, no tool needed.
2. **Analytical question** (e.g. "what trends …", "analyze …", "what is the distribution …")
   → `analyze_data`
3. **User wants to train a model** → Use `train_model` to run the full pipeline.
   Warn the user this can take several minutes. The user may optionally specify a
   dataset, but it is not required — the pipeline can discover suitable data on
   its own.  Model families available: supervised, unsupervised, neural_networks.
   If the user doesn't specify, the pipeline selects the best fit automatically.
4. **Ambiguous request** → ask clarifying questions (target variable? prediction
   type? which dataset?) BEFORE calling any tool.

## Rules
- When a user wants to train a model, confirm with them before starting (it takes time).
- After training completes, summarise the results (metrics, model type, report location).
- Be concise but thorough. Show your reasoning when it helps the user.\
"""


# ============================================================================
# Agent
# ============================================================================

TOOLS = [
    analyze_data,
    train_model,
]

_checkpointer = MemorySaver()

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    checkpointer=_checkpointer,
)

__all__ = ["agent", "TOOLS", "ORCHESTRATOR_SYSTEM_PROMPT"]
