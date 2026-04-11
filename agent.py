"""
Main Orchestrator Agent — Jubilee AI Chatbot

Uses langchain's create_agent to run a conversational loop that can spin up
specialised sub-agents on demand:

  1. Analysis sub-agent  (agents/analysis_agent_v2)
     → statistical analysis, pretrained-model inference, data exploration

  2. Dataset search — queries the backend dataset registry (Postgres).

Training is handled by the intent router + training graph (not the orchestrator).
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.factory import (AgentMiddleware, ModelRequest,
                                      ModelResponse)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path / env bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")


def _configure_langsmith_tracing() -> None:
    """
    Keep tracing opt-in for local orchestrator runs.

    LangSmith uploads can add long delays when the local dev environment cannot
    reach the LangSmith API. Set ENABLE_LANGSMITH_TRACING=true to re-enable.
    """
    enabled = os.getenv("ENABLE_LANGSMITH_TRACING", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


_configure_langsmith_tracing()

_MODEL_TOOLS_DIR = _ROOT / "tools" / "models-tools" / "training"
if str(_MODEL_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_TOOLS_DIR))
from model_storage import (evaluate_model_tool, get_model_info_tool,
                           list_trained_models_tool, predict_with_model_tool)

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-5.4", temperature=0)


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
    - Training custom models (handled by the training pipeline)
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
# Tool 2 — Dataset Search (queries the backend dataset registry)
# ============================================================================

class SearchDatasetsInput(BaseModel):
    query: str = Field(
        description=(
            "Search terms to find relevant datasets, or use '*' / 'all' / empty "
            "string to list every available dataset."
        )
    )


_SEARCH_RESULTS_MAX = 15

_NOISE_WORDS = frozenset({
    "find", "search", "look", "for", "me", "data", "dataset", "datasets",
    "related", "to", "about", "the", "a", "an", "some", "get", "show",
    "list", "browse", "discover", "recommend", "suggest", "local", "i",
    "have", "do", "what", "my", "with", "on", "of", "in", "and", "or",
})


def _extract_search_words(raw_query: str) -> list[str]:
    """Strip common filler words so only meaningful terms remain."""
    words = raw_query.lower().split()
    meaningful = [w for w in words if w not in _NOISE_WORDS]
    return meaningful if meaningful else words


def _get_backend_datasets() -> list[dict]:
    """Fetch the full dataset list from the backend catalog (Postgres)."""
    from backend.catalog import service as catalog_service
    return catalog_service.get_datasets(include_derived=False)


def _match_datasets(datasets: list[dict], query: str) -> list[dict]:
    """Score and filter datasets against search terms.

    Empty/wildcard queries return everything (inventory mode).
    """
    q = (query or "").strip().lower()
    is_list_all = not q or q in ("*", "**", "all", "any", "everything")

    if is_list_all:
        return datasets[:_SEARCH_RESULTS_MAX]

    search_words = _extract_search_words(q)
    scored: list[tuple[int, dict]] = []
    for ds in datasets:
        haystack = " ".join([
            (ds.get("name") or ""),
            (ds.get("description") or ""),
            (ds.get("use_case") or ""),
            " ".join(ds.get("columns") or []) if isinstance(ds.get("columns"), list) else "",
        ]).lower()

        score = sum(1 for w in search_words if w in haystack)
        if score > 0:
            scored.append((score, ds))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ds for _, ds in scored[:_SEARCH_RESULTS_MAX]]


def _format_dataset_results(datasets: list[dict]) -> str:
    if not datasets:
        return ""
    lines = ["### Datasets\n"]
    for i, ds in enumerate(datasets, start=1):
        name = ds.get("name") or "Untitled"
        rows = ds.get("rows") or "?"
        cols = ds.get("columns")
        ncols = len(cols) if isinstance(cols, list) else (cols or "?")
        desc = ds.get("description") or ""
        desc_part = f" — {desc}" if desc else ""
        lines.append(f"{i}. **{name}** ({rows} rows, {ncols} cols){desc_part}")
    return "\n".join(lines)


@tool(args_schema=SearchDatasetsInput)
def search_datasets(query: str) -> str:
    """Search available datasets in the workspace.

    MUST be called whenever the user asks to find, search for, discover, or
    get recommendations for datasets. NEVER answer dataset questions from
    your own knowledge — always use this tool to get real results.

    After the user picks a dataset, it's ready to use immediately for
    training or analysis via its name.
    """
    all_datasets = _get_backend_datasets()
    matched = _match_datasets(all_datasets, query)

    if not matched:
        return f"No datasets found matching '{query}'."

    header = "You can train on any of these immediately using the dataset name.\n\n"
    return header + _format_dataset_results(matched)


# ============================================================================
# Tool — Training plan proposal (conversational intake → structured plan for UI)
# ============================================================================

_DEFAULT_TRAINING_RECAP = [
    "Load and validate the dataset",
    "Choose model family, clean and standardize columns",
    "Define target, splits, and features",
    "Train, evaluate, and generate the audit report",
]


class ProposeTrainingPlanInput(BaseModel):
    goal: str = Field(
        description=(
            "What the model is for in plain language: the user's real decision or outcome "
            "(e.g. rank default risk for underwriting), not generic 'train a model'. One or two sentences."
        )
    )
    dataset_refs: list[str] = Field(
        description=(
            "Exact dataset names from search_datasets results or the user's selection. "
            "Must match workspace names — never invent them."
        ),
    )
    dataset_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable dataset names in the same order as dataset_refs "
            "(for display). If empty, refs are shown as labels."
        ),
    )
    preferences: str = Field(
        default="",
        description="Optional: model family, metric to optimize, class imbalance, etc.",
    )
    recap_steps: list[str] = Field(
        default_factory=list,
        description="Short bullets describing what the pipeline will do; omit to use defaults.",
    )


@tool(args_schema=ProposeTrainingPlanInput)
def propose_training_plan(
    goal: str,
    dataset_refs: list[str],
    dataset_labels: Optional[list[str]] = None,
    preferences: str = "",
    recap_steps: Optional[list[str]] = None,
) -> str:
    """Finalize a training plan once you have workable dataset ref(s) and a goal you could explain to a product owner.

    Do **not** call this tool if you are **unsure** what the user wants the model to accomplish. In that case,
    reply in normal chat with one or two clarifying questions, then call this tool only after they answer.

    Call ONLY when:
    - The user wants to train / build a predictive model, and
    - You have at least one real local dataset ref (use search_datasets if needed), and
    - You understand **why** they want the model: what decision, risk, or outcome it supports. Technical
      details (metrics, exact target column) can stay vague; the **goal** string must not be.

    After calling, do not add visible reply text in that turn (app shows the plan). Do NOT paste this JSON.
    The app shows **Run step-by-step** and **Run in background** buttons from the tool result.
    """
    refs = [str(r).strip() for r in (dataset_refs or []) if str(r).strip()]
    raw_labels = [str(x).strip() for x in (dataset_labels or []) if str(x).strip()]
    labels = [raw_labels[i] if i < len(raw_labels) else refs[i] for i in range(len(refs))]
    steps = [str(s).strip() for s in (recap_steps or []) if str(s).strip()]
    if not steps:
        steps = list(_DEFAULT_TRAINING_RECAP)
    payload = {
        "goal": goal.strip(),
        "dataset_refs": refs,
        "dataset_labels": labels,
        "preferences": (preferences or "").strip() or None,
        "recap_steps": steps,
    }
    return json.dumps(payload, ensure_ascii=False)


# ============================================================================
# System Prompt
# ============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are **Jubilee**, an AI assistant for data analysis and machine learning.

## Available Tools

| Tool | When to use |
|---|---|
| `search_datasets` | User wants to find, browse, or discover datasets in the workspace |
| `analyze_data` | Analytical questions: statistics, trends, pretrained-model inference, data exploration |
| `predict_with_model` | Run predictions on a dataset using a trained model |
| `evaluate_model` | Evaluate a trained model's performance on a labeled dataset |
| `list_trained_models` | List all trained models with their metrics |
| `get_model_info` | Get detailed info about a specific trained model |
| `propose_training_plan` | User wants to **train** and you are **sure** what the goal is (see Decision Flow §4). If the goal is unclear, **ask questions in chat first** — do not call this tool until you understand it. |

## Prediction & Model Tools
- **predict_with_model**: Run predictions on a dataset using a trained model. Use when the user wants to make predictions, score new data, or test a model on a dataset. Requires a model name and a registered dataset ref.
- **evaluate_model**: Evaluate a trained model's performance on a labeled dataset. Use when the user asks about model accuracy, performance metrics, or wants to compare how a model performs. Supports threshold optimization for imbalanced classification.
- **list_trained_models**: List all trained models with their metrics. Use when the user asks what models are available, wants to see trained models, or needs to pick a model for prediction.
- **get_model_info**: Get detailed info about a specific trained model. Use when the user asks about a specific model's features, hyperparameters, or training details.

## Decision Flow

**Training — ask when unsure:** If you are not confident you understand the user's **goal** (what problem
the model solves or what decision it supports), **stop and ask** in your normal reply. Do **not** call
`propose_training_plan` until you could state their goal in one clear sentence you believe they would agree with.

1. **General / conversational question** → answer directly, no tool needed.
2. **User mentions finding, searching, looking for, or wanting datasets** →
   **ALWAYS call `search_datasets`**. NEVER answer dataset questions from your own
   knowledge — you MUST use the tool because it searches the workspace's dataset
   registry for real, usable results. This includes ANY of these patterns:
   - "find me data for …", "search for datasets …", "look for … data"
   - "what datasets are good for …", "recommend a dataset for …"
   - "I need data for …", "get me some … data", "what data do I have?"
   Present the results as a numbered list and ask which dataset the user wants to use.
3. **Analytical question** (e.g. "what trends …", "analyze …", "what is the distribution …")
   → `analyze_data`
4. **User wants to train / build a predictive model** → Follow this **order**: (1) understand the **goal**
   in plain language, (2) lock a **dataset name** from the workspace, (3) call `propose_training_plan`.
   The **goal** is what real-world decision or outcome the model supports (e.g. rank default risk for
   underwriting). Technical details (splits, metrics, model family) are for the pipeline unless the user
   cares—**do not** quiz them on recall vs precision vs AUC unless they ask about tradeoffs or metrics.
   - **If the goal is unclear:** ask one short clarifying question. **Do not** call `propose_training_plan`
     until you could write an honest one-sentence `goal` they would agree with.
   - **If you only have a vague verb** ('underwrite', 'score'): ask what outcome they need until the goal
     is concrete enough.
   - **Datasets:** If you do not have a workspace dataset name, call `search_datasets`. After it returns:
     - **Do not** paste or reformat the entire tool output again, and **do not** add redundant sections
       (e.g. "## Dataset found" repeating the same table). Say briefly what you found in **one** short
       paragraph or a tiny list.
     - If **exactly one** dataset clearly matches and the user **already** stated a concrete goal, either
       call `propose_training_plan` with that name and goal, **or** ask one yes/no ("Use **name** for this
       run?")—do **not** ask them to reply with both `1` **and** the dataset name.
     - If multiple datasets apply, ask which one in **one** line (number **or** name is enough).
   - **System context** from a prior trained run: terse follow-ups like `train` or `again` are ambiguous—ask
     what they want to change or train next; do not assume the old run answers the new message.
   - **Clarifying questions:** one at a time when possible; skip long option menus unless the user asked.
   - If the user message starts with `[Background task` or **Run on my behalf**, same flow: clarify if
     needed, then `propose_training_plan`.
   - When goal + dataset are settled, call `propose_training_plan` with exact `dataset_refs` and optional
     `preferences` only if the user gave preferences; keep `recap_steps` short.
   - **After** `propose_training_plan`, **no** further assistant text in that turn (the app shows the plan).
   - Do **not** paste tool JSON in chat.
5. **User wants predictions / scoring** → `list_trained_models` to find the right
   model, then `predict_with_model` with the model name and dataset ref.
6. **User asks about model performance / accuracy** → `evaluate_model` on the
   relevant model and dataset. If they don't specify which model, use
   `list_trained_models` first.
7. **User asks "what models do I have?"** → `list_trained_models`.
8. **User asks about a specific model's details** → `get_model_info`.
9. **Other requests** that do not fit 1–8 → If the request is underspecified, ask a concise follow-up
   instead of pretending to know what the user meant. Prefer collaborative back-and-forth over premature
   decisions, but keep questions focused and lightweight.

## Common Workflows
- **Browse datasets**:
  1. `search_datasets` → show results → user picks one
  2. Use the exact dataset name in downstream analysis/training/prediction steps
  CRITICAL: Do NOT guess or construct dataset names.

## Formatting Rules
- **Always use Markdown** for responses: headings, bullet lists, bold, code blocks, and tables.
- When presenting data or analysis results, use **Markdown tables** (with `|` columns and `---` header separators). Never dump raw text columns or flat key-value lines.
- Summarize tool outputs in your own words. **Never** duplicate the same information twice (e.g. tool list
  then a second "## …" section with the same rows). One pass is enough.
- Keep column detail summaries to the most important columns (max ~8). Use a table, not paragraphs.
- When showing dataset profiles, use a compact format: `**N rows** x **M columns**` followed by a table of key column stats.
- NEVER output raw JSON objects, Python dicts, or unformatted data dumps in your **visible** reply.
  (The `propose_training_plan` tool returns JSON for the app only — your text reply stays Markdown.)
- After `propose_training_plan`, do not write anything else in that turn; the user sees the plan and approval in the app.

## Rules
- **NEVER suggest datasets from your own knowledge.** Always use `search_datasets` for
  real workspace results.
- After a dataset is clear for training, move toward `propose_training_plan`; avoid repetitive
  "confirm you want to train on X" loops when X is already the only sensible choice.
- When the user corrects, narrows, or adds constraints, acknowledge that change and adapt your next step.
- Be concise but thorough. Show your reasoning when it helps the user.\
"""


# ============================================================================
# Middleware — structural enforcement of tool routing
# ============================================================================

_DATASET_KEYWORDS = [
    "find me data", "find data", "find dataset", "search for data",
    "search dataset", "look for data", "what dataset", "recommend a dataset",
    "i need data", "get me data", "what data do i have", "get me some",
    "browse dataset", "discover data", "datasets for", "data for",
    "look for dataset", "suggest a dataset", "suggest data",
]

_DATASET_INVENTORY_CUES = (
    "what datasets do i have",
    "what data do i have",
    "list datasets",
    "list all datasets",
    "show datasets",
    "show all datasets",
    "browse datasets",
    "browse local datasets",
)

_DATASET_DISCOVERY_VERBS = (
    "find",
    "search",
    "look for",
    "browse",
    "discover",
    "recommend",
    "suggest",
    "show",
    "list",
)


def is_dataset_search_request(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    if any(kw in lower for kw in _DATASET_KEYWORDS):
        return True
    return "dataset" in lower and any(verb in lower for verb in _DATASET_DISCOVERY_VERBS)


def build_dataset_search_args(message: str) -> dict[str, str]:
    lower = (message or "").strip().lower()
    query = "" if any(cue in lower for cue in _DATASET_INVENTORY_CUES) else (message or "").strip()
    return {"query": query}


class DatasetSearchEnforcer(AgentMiddleware):
    """Force ``search_datasets`` when the user asks about finding datasets.

    GPT-5.1 sometimes answers dataset questions from its own knowledge even
    when the system prompt says not to.  This middleware intercepts the model
    call and sets ``tool_choice`` to the specific function, making it
    structurally impossible for the LLM to skip the tool.
    """

    tools: list = []

    def wrap_model_call(self, request: ModelRequest, handler):
        already_called = any(
            (hasattr(msg, "name") and msg.name == "search_datasets")
            or (hasattr(msg, "tool_calls") and any(
                tc.get("name") == "search_datasets" for tc in (msg.tool_calls or [])
            ))
            for msg in request.messages
        )
        if already_called:
            return handler(request)

        last_human = None
        for msg in reversed(request.messages):
            if hasattr(msg, "type") and msg.type == "human":
                last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if is_dataset_search_request(last_human):
            request = request.override(
                tool_choice={"type": "function", "function": {"name": "search_datasets"}}
            )

        return handler(request)


# ============================================================================
# Agent
# ============================================================================

TOOLS = [
    search_datasets,
    analyze_data,
    predict_with_model_tool,
    evaluate_model_tool,
    list_trained_models_tool,
    get_model_info_tool,
    propose_training_plan,
]

_checkpointer = MemorySaver()

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
    middleware=[DatasetSearchEnforcer()],
    checkpointer=_checkpointer,
)

__all__ = [
    "agent",
    "TOOLS",
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "is_dataset_search_request",
    "build_dataset_search_args",
]
