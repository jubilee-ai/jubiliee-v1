"""
Main Orchestrator Agent — Jubilee AI Chatbot

Uses langchain's ``create_agent`` to run a conversational loop that decides on
every turn what the user needs and picks the right tool:

  1. ``search_datasets`` — workspace dataset registry.
  2. Analysis tools (EDA, correlations, grouping, distributions, charts, validation, …)
     + optional pretrained inference tools — see ``TOOLS``.
  3. ``propose_training_plan`` — show the user a plan card with Run/Background buttons.
  4. ``run_training_pipeline`` — spin up the training sub-agent (data_collection →
     select_model → cleaning → label/split → features → evaluate → training_approval →
     training → generate_report) with HITL. Jubilee sees the sub-agent's inputs and
     final output and can answer follow-ups (predict, evaluate, explain, retrain).
  5. ``predict_with_model`` / ``evaluate_model`` / ``list_trained_models`` /
     ``get_model_info`` — inference & registry operations on already-trained models.

The chat service pivots the current SSE stream into the training sub-agent's
stream when ``run_training_pipeline`` is called, so the UI gets live step
updates without any additional routing.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.factory import AgentMiddleware, ModelRequest
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from orchestrator_context import attached_datasets_active

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
from agents.subagents.orchestrator_tools import ORCHESTRATOR_SUBAGENT_TOOLS
from model_storage import (evaluate_model_tool, get_model_info_tool,
                           list_trained_models_tool, predict_with_model_tool)

# Analysis + data access (tools/data-tools on sys.path)
_DT_ROOT = _ROOT / "tools" / "data-tools"
if str(_DT_ROOT) not in sys.path:
    sys.path.insert(0, str(_DT_ROOT))

from analysis import (  # noqa: E402
    categorical_association_test_tool,
    chart_tool,
    concentration_analysis_tool,
    correlation_matrix_tool,
    data_validation_tool,
    distribution_analysis_tool,
    eda_report_tool,
    feature_diagnostics_tool,
    group_comparison_test_tool,
    group_summary_tool,
    regression_summary_tool,
    trend_analysis_tool,
)
from data_loader import dataset_get_tool  # noqa: E402
from sql_query import sql_query_tool  # noqa: E402

# Pretrained scoring / NLP / forecast tools (direct — no nested selector agent)
_PRETRAINED_DIR = _ROOT / "tools" / "models-tools" / "pretrained"
if str(_PRETRAINED_DIR) not in sys.path:
    sys.path.insert(0, str(_PRETRAINED_DIR))

from bert_finetuned_claim_detection import claim_detection_tool  # noqa: E402
from chronos_2 import chronos2_forecast_tool  # noqa: E402
from credit_risk import credit_card_risk_prediction_tool  # noqa: E402
from finbert_tone import finbert_tone_tool  # noqa: E402
from google_timesfm import timesfm_forecast_tool  # noqa: E402
from loan_default_prediction import loan_default_prediction_tool  # noqa: E402
from prosus_finbert import finbert_sentiment_tool  # noqa: E402

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model="gpt-5.4", temperature=0)


# ============================================================================
# Tool — Dataset Search (queries the backend dataset registry)
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
    if attached_datasets_active():
        return (
            "Dataset search is disabled here — the user already attached workspace dataset(s). "
            "Use those refs with analysis tools (`correlation_matrix_tool`, `group_summary_tool`, "
            "`chart_tool`, `eda_report_tool`, …); do **not** call `search_datasets`."
        )
    all_datasets = _get_backend_datasets()
    matched = _match_datasets(all_datasets, query)

    if not matched:
        return f"No datasets found matching '{query}'."

    header = "You can train on any of these immediately using the dataset name.\n\n"
    return header + _format_dataset_results(matched)


# ============================================================================
# Tool — Training plan proposal (conversational intake → structured plan for UI)
# ============================================================================

from agents.training.core.pipeline import DEFAULT_TRAINING_RECAP as _DEFAULT_TRAINING_RECAP_TUPLE

_DEFAULT_TRAINING_RECAP = list(_DEFAULT_TRAINING_RECAP_TUPLE)


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
# Tool — Run Training Pipeline (pivots SSE stream into the training sub-agent)
# ============================================================================


class RunTrainingPipelineInput(BaseModel):
    goal: str = Field(
        description=(
            "What the model is for in plain language — the user's real decision or "
            "outcome (e.g. 'rank default risk for underwriting'). One or two sentences."
        )
    )
    dataset_refs: list[str] = Field(
        description=(
            "Exact workspace dataset name(s) to train on. Must come from "
            "search_datasets results or a workspace ref the user attached."
        )
    )
    model_preference: Optional[str] = Field(
        default=None,
        description=(
            "Optional model family hint: 'supervised', 'unsupervised', or "
            "'neural_networks'. Omit when unsure and let the pipeline choose."
        ),
    )
    preferences: Optional[str] = Field(
        default=None,
        description=(
            "Optional free-form preferences: metric to optimize, class imbalance, "
            "time budget, etc. Omit if the user gave none."
        ),
    )


#: Sentinel returned by ``run_training_pipeline``.
#:
#: The tool itself does no work — :mod:`backend.chat.service` watches for this
#: tool call inside the agent stream and then hands off to the training
#: sub-agent (``generate_graph_sse_events``) on the same SSE response. The
#: tool's return value is recorded in the chat checkpoint so Jubilee can
#: reason about the subsequent training run in follow-up turns.
_TRAINING_HANDOFF_MARKER = "__jubilee_training_handoff__"


@tool(args_schema=RunTrainingPipelineInput)
def run_training_pipeline(
    goal: str,
    dataset_refs: list[str],
    model_preference: Optional[str] = None,
    preferences: Optional[str] = None,
) -> str:
    """Kick off the full training sub-agent (HITL, streamed to the UI).

    Use ONLY when:
    - The user explicitly wants to train / build a predictive model, AND
    - You have at least one real workspace dataset ref (use ``search_datasets``
      first if you don't), AND
    - You can state their goal in one clear sentence (if unclear, ask first —
      do NOT call this tool).

    What happens after you call this:
    - The UI transitions to the training pipeline view with live progress.
    - The sub-agent runs ``data_collection → select_model → cleaning →
      label/split → features → evaluate → training_approval → training →
      generate_report`` with Human-in-the-Loop checkpoints.
    - When training completes, the summary (model, metrics, target, report
      path) is fed back to you on the next user turn so you can answer
      follow-ups (predict, evaluate, compare, retrain, explain) without
      re-asking for basics.

    Do NOT call this tool just for analysis — use the analysis tools directly
    (correlations, grouping, ``chart_tool``, ``eda_report_tool``, …).

    After calling, do NOT write any other reply text in that turn. The app
    shows live pipeline progress from the sub-agent stream.
    """
    refs = [str(r).strip() for r in (dataset_refs or []) if str(r).strip()]
    payload = {
        "marker": _TRAINING_HANDOFF_MARKER,
        "goal": (goal or "").strip(),
        "dataset_refs": refs,
        "model_preference": (model_preference or "").strip() or None,
        "preferences": (preferences or "").strip() or None,
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_training_handoff(tool_result: object) -> Optional[dict]:
    """Return the training handoff payload if the tool result is our sentinel.

    The chat service calls this on every ``run_training_pipeline`` tool result
    captured from the agent stream. A non-``None`` return value means the
    chat generator should pivot into ``generate_graph_sse_events`` with those
    args.
    """
    raw = tool_result if isinstance(tool_result, str) else str(tool_result or "")
    raw = raw.strip()
    if not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("marker") != _TRAINING_HANDOFF_MARKER:
        return None
    refs = data.get("dataset_refs") or []
    if not isinstance(refs, list) or not refs:
        return None
    return {
        "goal": str(data.get("goal") or "").strip(),
        "dataset_refs": [str(r).strip() for r in refs if str(r).strip()],
        "model_preference": data.get("model_preference") or None,
        "preferences": data.get("preferences") or None,
    }


# ============================================================================
# System Prompt
# ============================================================================

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are **Jubilee**, an AI assistant for data analysis and machine learning.

Pick **one** tool per turn unless the user clearly needs chained steps in one reply; prefer **minimal**
tool calls (usually **2–4** tools for a full analytical answer — see analysis workflow below).

## Tools (summary)

**Discovery**
- `search_datasets` — Browse the workspace catalog (disabled when the user already attached datasets — see below).

**Analysis — use refs from `search_datasets` results or `[ATTACHED DATASETS …]` in the message**
- `eda_report_tool` — One-shot profile + target associations + correlations + alerts (good first pass).
- `correlation_matrix_tool` — Numeric correlations / multicollinearity.
- `group_summary_tool` — Slice metrics by categories (means by region, smoker, etc.).
- `distribution_analysis_tool` — Histograms, skew, percentiles.
- `feature_diagnostics_tool` — Feature quality vs a target before training.
- `data_validation_tool` — Rule-based validation when the user cares about DQ rules.
- `concentration_analysis_tool` — Gini / Lorenz / Pareto concentration.
- `trend_analysis_tool` — Time trends (needs a date column).
- `group_comparison_test_tool` — **Inferential:** compare a **numeric** column across **groups** (Welch t / Mann-Whitney / ANOVA / Kruskal). Use for "is the difference significant?", effect sizes, not just group means.
- `categorical_association_test_tool` — **Inferential:** association between two **categorical** columns (chi-square, Cramér's V; Fisher for 2x2 with small expected counts). Use for "is feature X related to the target / segment?".
- `regression_summary_tool` — **Inferential:** OLS (numeric target) or **logit** (binary target) with coefficients, p-values, CIs; use when the user wants multivariate "controlling for other factors" (numeric predictors only — encode categoricals first).
- `chart_tool` — Build **one** primary chart JSON for the UI — **always** include this when answering an analytical question with quantitative evidence (pick chart_type: bar, grouped_bar, histogram, scatter, line, or box).

**Data access**
- `dataset_get_tool` — Peek rows/schema slice.
- `sql_query_tool` — SQL when the workspace exposes SQL tables.

**Pretrained inference** (only when the user's goal matches — credit risk, loan default, sentiment, claims, forecasts)
- `credit_card_risk_prediction_tool`, `loan_default_prediction_tool`, `finbert_tone_tool`, `finbert_sentiment_tool`, `claim_detection_tool`, `chronos2_forecast_tool`, `timesfm_forecast_tool`

**Training & trained models**
- `propose_training_plan`, `run_training_pipeline` (full UI pipeline with HITL)
- **Subagents (optional, composable):** `invoke_data_analyst`, `invoke_feature_engineer`, `invoke_trainer`,
  `invoke_evaluator`, `invoke_explainer`, `invoke_deployer`, `invoke_monitor` — each returns JSON; you choose order.
- `predict_with_model`, `evaluate_model`, `list_trained_models`, `get_model_info`

## Attached datasets (critical)

When the message starts with **`[ATTACHED DATASETS`**:
- Data is **already loaded** — you **must not** call `search_datasets`.
- Use the listed **`dataset_ref`** strings directly in analysis tools.
- Do **not** paste the schema/sample again for the user (they see it in the UI chip).

## Analysis workflow

For questions like “what affects X”, “correlations”, “distribution”, “segments”:

1. If you need a quick overview → `eda_report_tool` with `target_col` when predicting a column.
2. If the user asks about **statistical significance**, **p-values**, **effect size**, or **"is A associated with B"**:
   - numeric outcome vs groups → `group_comparison_test_tool`
   - two categoricals (e.g. target vs feature) → `categorical_association_test_tool`
   - multivariate adjusted effects (numeric features) → `regression_summary_tool`
3. Otherwise pick **up to two** focused descriptive tools (e.g. `correlation_matrix_tool` + `group_summary_tool`).
4. Add **`chart_tool`** when it clarifies the main finding (e.g. bar of mean outcome by group).

Keep total analysis tool calls ≤ **4** when inferential tools are needed; ≤ **3** for descriptive-only questions unless the user asks for exhaustive exploration.

### Response format (analysis answers)

1. **1–2 sentences** — direct answer.
2. **≤5 bullets** — evidence with numbers from tools (no duplicate tables).
3. **Want more?** — 1–2 short follow-up ideas.

**Banned:** "Actions Taken / Data & Evidence / Reasoning / Conclusion" giant templates, repeating the same statistics in multiple sections, dumping raw `<ANALYSIS_JSON>` tags (they are for the app).

## Dataset discovery (no attachment)

When the user asks to **find**, **search**, **browse**, or **list** datasets (and there is **no** `[ATTACHED DATASETS` block):
- **Always** call `search_datasets` — never invent catalog entries.

## Training

Same as before: clarify **goal** before `propose_training_plan` / `run_training_pipeline`. Attaching a dataset is **not** by itself a request to train.

After `propose_training_plan` or `run_training_pipeline`, write **no** assistant text in that turn.

## Prediction & evaluation

Use `list_trained_models` → `predict_with_model`; `evaluate_model` for metrics; `get_model_info` for specifics.

## Formatting

Markdown, compact tables only when they add clarity. Never paste tool JSON intended for the app (`propose_training_plan`, `run_training_pipeline`, structured sidecars).

Be concise by default.
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

_ATTACHED_DATASETS_MARKER = "[ATTACHED DATASETS"


def is_dataset_search_request(message: str) -> bool:
    lower = (message or "").strip().lower()
    if not lower:
        return False
    if any(kw in lower for kw in _DATASET_KEYWORDS):
        return True
    return "dataset" in lower and any(verb in lower for verb in _DATASET_DISCOVERY_VERBS)


class DatasetSearchEnforcer(AgentMiddleware):
    """Force ``search_datasets`` when the user asks about finding datasets.

    When ``[ATTACHED DATASETS …]`` is present, ``search_datasets`` is removed from
    the tool list so the model cannot call search on an already-loaded dataset.
    """

    tools: list = []

    def wrap_model_call(self, request: ModelRequest, handler):
        last_human = None
        for msg in reversed(request.messages):
            if hasattr(msg, "type") and msg.type == "human":
                last_human = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        tools = getattr(request, "tools", None)
        if tools and last_human and _ATTACHED_DATASETS_MARKER in last_human:
            filtered = [
                t for t in tools
                if getattr(t, "name", None) != "search_datasets"
            ]
            if filtered:
                request = request.override(tools=filtered)
            return handler(request)

        already_called = any(
            (hasattr(msg, "name") and msg.name == "search_datasets")
            or (hasattr(msg, "tool_calls") and any(
                tc.get("name") == "search_datasets" for tc in (msg.tool_calls or [])
            ))
            for msg in request.messages
        )
        if already_called:
            return handler(request)

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
    eda_report_tool,
    correlation_matrix_tool,
    group_summary_tool,
    distribution_analysis_tool,
    feature_diagnostics_tool,
    data_validation_tool,
    concentration_analysis_tool,
    trend_analysis_tool,
    group_comparison_test_tool,
    categorical_association_test_tool,
    regression_summary_tool,
    chart_tool,
    dataset_get_tool,
    sql_query_tool,
    credit_card_risk_prediction_tool,
    loan_default_prediction_tool,
    finbert_tone_tool,
    finbert_sentiment_tool,
    claim_detection_tool,
    chronos2_forecast_tool,
    timesfm_forecast_tool,
    predict_with_model_tool,
    evaluate_model_tool,
    list_trained_models_tool,
    get_model_info_tool,
    propose_training_plan,
    run_training_pipeline,
    *ORCHESTRATOR_SUBAGENT_TOOLS,
]

# Tools whose results emit `<ANALYSIS_JSON>` sidecars for the chat UI (suppress duplicate tool_end text).
SIDECHANNEL_TOOL_NAMES = frozenset(
    name
    for name in (
        "eda_report_tool",
        "correlation_matrix_tool",
        "group_summary_tool",
        "distribution_analysis_tool",
        "feature_diagnostics_tool",
        "data_validation_tool",
        "concentration_analysis_tool",
        "trend_analysis_tool",
        "group_comparison_test_tool",
        "categorical_association_test_tool",
        "regression_summary_tool",
        "chart_tool",
    )
)

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
    "SIDECHANNEL_TOOL_NAMES",
    "is_dataset_search_request",
    "run_training_pipeline",
    "parse_training_handoff",
]
