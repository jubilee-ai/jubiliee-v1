"""
Main Orchestrator Agent — Jubilee AI Chatbot

Uses langchain's create_agent to run a conversational loop that can spin up
specialised sub-agents on demand:

  1. Analysis sub-agent  (agents/analysis_agent_v2)
     → statistical analysis, pretrained-model inference, data exploration

  2. Dataset search / curation — local workspace only (Kaggle / HuggingFace disabled
     via EXTERNAL_DATASET_CATALOG_ENABLED; see dataset curator when re-enabled).

Training is handled by the intent router + training graph (not the orchestrator).
"""

import asyncio
import json
import sys
from typing import Optional

from dotenv import load_dotenv
from pathlib import Path
from langchain.agents import create_agent
from langchain.agents.factory import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path / env bootstrap
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
load_dotenv(_ROOT / ".env")

# Kaggle / HuggingFace search uses agents/dataset_curator (MCP). Set True to re-enable.
EXTERNAL_DATASET_CATALOG_ENABLED = False

_MODEL_TOOLS_DIR = _ROOT / "tools" / "models-tools" / "training"
if str(_MODEL_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_TOOLS_DIR))
from model_storage import (
    predict_with_model_tool,
    evaluate_model_tool,
    list_trained_models_tool,
    get_model_info_tool,
)

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
# Tool 2 — Unified Dataset Search (local + Kaggle + HuggingFace)
# ============================================================================

class SearchDatasetsInput(BaseModel):
    query: str = Field(
        description=(
            "Search terms, or use '*' / 'all' / empty string to list every local dataset "
            "(fast; skips external search unless you set source to kaggle or huggingface)."
        )
    )
    source: Optional[str] = Field(
        default=None,
        description="Limit to a source: 'local', 'kaggle', 'huggingface', or None for all",
    )


def _run_async(coro):
    """Run an async coroutine from sync context, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()

    return asyncio.run(coro)


_LOCAL_INVENTORY_MAX = 50


def _is_local_inventory_query(query: str) -> bool:
    """True when the user wants every local dataset (not a semantic search).

    Word-overlap search treats '*' as a token that never matches refs, so we
    must branch to list-all. Skipping external search for these avoids slow
    Kaggle/HuggingFace agent runs for 'what do I have?' style questions.
    """
    q = (query or "").strip().lower()
    if not q:
        return True
    if q in (
        "*",
        "**",
        "all",
        "any",
        "any*",
        "all datasets",
        "everything",
        "list all",
        "list all datasets",
        "show all",
        "show all datasets",
        "what datasets?",
        "what datasets do i have",
        "what data do i have",
    ):
        return True
    # Only wildcards / whitespace (e.g. "*", "**", " * ")
    if q.replace("*", "").strip() == "":
        return True
    return False


def _search_local_datasets(query: str) -> list[dict]:
    """Search local registered datasets, SQL tables, and catalog for matches."""
    _DATA_TOOLS_DIR = _ROOT / "tools" / "data-tools"
    if str(_DATA_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_TOOLS_DIR))
    from utils import get_all_available_datasets

    all_ds = get_all_available_datasets()

    if _is_local_inventory_query(query):
        results: list[dict] = []
        for ds in all_ds.get("registered", []):
            ref = ds["ref"]
            results.append({
                "source": "local",
                "ref": ref,
                "name": ref,
                "rows": ds.get("rows", "?"),
                "columns": ds.get("columns", "?"),
                "match_score": 0,
            })
        for ds in all_ds.get("sql_tables", []):
            ref = ds["ref"]
            results.append({
                "source": "local (SQL)",
                "ref": ref,
                "name": ref,
                "rows": ds.get("rows", "?"),
                "columns": ds.get("columns", "?"),
                "match_score": 0,
            })
        for ds in all_ds.get("catalog", []):
            ref = ds["ref"]
            name = ds.get("name", ref)
            cols = ds.get("columns", [])
            ncol = len(cols) if isinstance(cols, list) else cols
            results.append({
                "source": "local (catalog)",
                "ref": ref,
                "name": name,
                "rows": ds.get("rows", "?"),
                "columns": ncol,
                "match_score": 0,
            })
        results.sort(key=lambda x: (x["source"], str(x["ref"])))
        return results[:_LOCAL_INVENTORY_MAX]

    query_lower = query.lower()
    query_words = set(query_lower.split())
    results = []

    for ds in all_ds.get("registered", []):
        ref = ds["ref"]
        ref_lower = ref.lower()
        ref_words = set(ref_lower.replace("_", " ").split())
        overlap = query_words & ref_words
        if overlap or any(w in ref_lower for w in query_words):
            results.append({
                "source": "local",
                "ref": ref,
                "name": ref,
                "rows": ds.get("rows", "?"),
                "columns": ds.get("columns", "?"),
                "match_score": len(overlap),
            })

    for ds in all_ds.get("sql_tables", []):
        ref = ds["ref"]
        ref_lower = ref.lower()
        col_text = " ".join(ds.get("column_names", [])).lower()
        ref_words = set(ref_lower.replace("_", " ").split())
        overlap = query_words & ref_words
        col_overlap = any(w in col_text for w in query_words)
        if overlap or col_overlap:
            results.append({
                "source": "local (SQL)",
                "ref": ref,
                "name": ref,
                "rows": ds.get("rows", "?"),
                "columns": ds.get("columns", "?"),
                "match_score": len(overlap) + (1 if col_overlap else 0),
            })

    for ds in all_ds.get("catalog", []):
        ref = ds["ref"]
        name = ds.get("name", ref)
        search_text = f"{ref} {name}".lower()
        if any(w in search_text for w in query_words):
            cols = ds.get("columns", [])
            results.append({
                "source": "local (catalog)",
                "ref": ref,
                "name": name,
                "rows": ds.get("rows", "?"),
                "columns": len(cols) if isinstance(cols, list) else cols,
                "match_score": sum(1 for w in query_words if w in search_text),
            })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


def _format_local_results(local_results: list[dict], start_num: int = 1) -> str:
    if not local_results:
        return ""
    lines = ["### Local Datasets\n"]
    for i, ds in enumerate(local_results[:10], start=start_num):
        lines.append(
            f"{i}. **{ds['name']}** — `{ds['ref']}` "
            f"({ds['rows']} rows, {ds['columns']} cols) "
            f"[source: {ds['source']}]"
        )
    return "\n".join(lines)


@tool(args_schema=SearchDatasetsInput)
def search_datasets(query: str, source: Optional[str] = None) -> str:
    """Search for datasets across local storage, Kaggle, and HuggingFace.

    MUST be called whenever the user asks to find, search for, discover, or
    get recommendations for datasets. NEVER answer dataset questions from
    your own knowledge — always use this tool to get real results.

    Searches local datasets (registered, SQL tables, catalog). When
    EXTERNAL_DATASET_CATALOG_ENABLED is True, also searches Kaggle/HuggingFace
    via the dataset curator.

    After the user picks a local dataset, it's ready to use immediately.
    For external sources (when enabled), use curate_dataset to download and register.
    """
    sections = []
    next_num = 1

    # --- Local search (always runs unless source is explicitly external) ---
    if source not in ("kaggle", "huggingface"):
        local_results = _search_local_datasets(query)
        if local_results:
            sections.append(_format_local_results(local_results, start_num=next_num))
            next_num += min(len(local_results), 10)

    # --- External search (Kaggle / HuggingFace) ---
    if source == "local":
        if not sections:
            return f"No local datasets found matching '{query}'."
        return "\n\n".join(sections)

    # List-all / inventory queries: local only — do not spawn the curator agent (slow).
    if source is None and _is_local_inventory_query(query):
        if not sections:
            return (
                "No datasets found in this workspace yet (nothing registered, "
                "no SQL tables, and no catalog assets)."
            )
        header = (
            "For **local** results, you can train on them immediately using the ref name.\n\n"
        )
        return header + "\n\n".join(sections)

    if not EXTERNAL_DATASET_CATALOG_ENABLED:
        if sections:
            header = (
                "For **local** results, you can train on them immediately using the ref name.\n\n"
            )
            return (
                header
                + "\n\n".join(sections)
                + "\n\n*(Kaggle / HuggingFace catalog search is disabled.)*"
            )
        if source in ("kaggle", "huggingface"):
            return (
                "Kaggle and HuggingFace dataset search is disabled. "
                "Use local datasets only (registered refs, SQL tables, catalog), or add data in the workspace."
            )
        return (
            f"No local datasets matched '{query}'. "
            "Kaggle / HuggingFace search is disabled — try different keywords or register data locally."
        )

    # External search is intentionally disabled in main chat.
    if not sections:
        return f"No local datasets found matching '{query}'."
    header = (
        "For **local** results, you can train on them immediately using the ref name.\n\n"
    )
    return header + "\n\n".join(sections) + "\n\n*(Kaggle / HuggingFace search is disabled.)*"


# ============================================================================
# Tool 3 — Dataset Curation (download, profile, register)
# ============================================================================

class CurateDatasetInput(BaseModel):
    goal: str = Field(
        description="What the dataset will be used for (e.g. 'customer segmentation clustering')"
    )
    source: str = Field(
        description="'local', 'kaggle', or 'huggingface'"
    )
    identifier: str = Field(
        description="For local: the dataset ref name. For Kaggle: 'owner/slug'. For HuggingFace: 'org/repo'."
    )


@tool(args_schema=CurateDatasetInput)
def curate_dataset(goal: str, source: str, identifier: str) -> str:
    """Select and prepare a dataset for training.

    For local datasets: validates the ref exists and profiles it.
    For external datasets (when enabled): downloads, profiles, and registers them.

    Use this after the user picks a dataset from search_datasets results.
    Returns the registered dataset reference name that can be used for training.
    """
    _DATA_TOOLS_DIR = _ROOT / "tools" / "data-tools"
    if str(_DATA_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_TOOLS_DIR))
    from utils import get_registered_dataset

    if not EXTERNAL_DATASET_CATALOG_ENABLED and source.lower() in ("kaggle", "huggingface"):
        return (
            "Kaggle and HuggingFace dataset import is disabled. "
            "Use a local dataset ref (profile with source local / local (sql) / local (catalog))."
        )

    if source.lower() in ("local", "local (sql)", "local (catalog)"):
        df = get_registered_dataset(identifier)
        if df is None:
            return f"Local dataset '{identifier}' not found in registry."

        from agents.dataset_curator.tools import profile_dataset
        profile_result = profile_dataset.invoke({"dataset_ref": identifier})

        return (
            f"Dataset **{identifier}** is already available locally ({len(df):,} rows, {len(df.columns)} cols).\n\n"
            f"Ready to use for training — reference it as `{identifier}`.\n\n"
            f"### Profile\n{profile_result}"
        )

    if source.lower() == "kaggle":
        from agents.dataset_curator.tools import download_kaggle_dataset, profile_dataset

        parts = identifier.split("/", 1)
        if len(parts) != 2:
            return f"Invalid Kaggle identifier '{identifier}'. Expected format: owner/slug"

        dl_result = download_kaggle_dataset.invoke(
            {"owner_slug": parts[0], "dataset_slug": parts[1]}
        )

        if "Error" in dl_result:
            return dl_result

        import re
        refs = re.findall(r"`(kaggle_\w+)`", dl_result)
        if not refs:
            return f"Download succeeded but no datasets were registered.\n{dl_result}"

        best_ref = refs[0]
        best_rows = 0
        for ref in refs:
            df = get_registered_dataset(ref)
            if df is not None and len(df) > best_rows:
                best_rows = len(df)
                best_ref = ref

        profile_result = profile_dataset.invoke({"dataset_ref": best_ref})

        return (
            f"Dataset registered as **{best_ref}** ({best_rows:,} rows)\n\n"
            f"Ready to use for training — reference it as `{best_ref}`.\n\n"
            f"### Profile\n{profile_result}\n\n"
            f"### All registered files\n{dl_result}"
        )

    elif source.lower() == "huggingface":
        from agents.dataset_curator.tools import download_hf_dataset, profile_dataset

        dl_result = download_hf_dataset.invoke({"dataset_id": identifier})

        if "Error" in dl_result:
            return dl_result

        import re
        refs = re.findall(r"`(hf_\w+)`", dl_result)
        if not refs:
            return f"Download succeeded but no datasets were registered.\n{dl_result}"

        best_ref = refs[0]
        best_rows = 0
        for ref in refs:
            df = get_registered_dataset(ref)
            if df is not None and len(df) > best_rows:
                best_rows = len(df)
                best_ref = ref

        profile_result = profile_dataset.invoke({"dataset_ref": best_ref})

        return (
            f"Dataset registered as **{best_ref}** ({best_rows:,} rows)\n\n"
            f"Ready to use for training — reference it as `{best_ref}`.\n\n"
            f"### Profile\n{profile_result}\n\n"
            f"### All registered files\n{dl_result}"
        )

    else:
        return f"Unknown source '{source}'. Use 'kaggle' or 'huggingface'."


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
    goal: str = Field(description="Clear, specific training objective (one or two sentences).")
    dataset_refs: list[str] = Field(
        description=(
             "Exact local dataset ref strings from search_datasets or the user's selection. "
            "Must match workspace refs — never invent names."
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
    """Finalize a training plan once you have workable dataset ref(s) and a goal clear enough to run.

    Call ONLY when:
    - The user wants to train / build a predictive model, and
    - You have at least one real local dataset ref (use search_datasets if needed).

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
| `search_datasets` | User wants to find, browse, or discover datasets — **local workspace only** (registered, SQL, catalog); Kaggle/HuggingFace disabled for now |
| `analyze_data` | Analytical questions: statistics, trends, pretrained-model inference, data exploration |
| `predict_with_model` | Run predictions on a dataset using a trained model |
| `evaluate_model` | Evaluate a trained model's performance on a labeled dataset |
| `list_trained_models` | List all trained models with their metrics |
| `get_model_info` | Get detailed info about a specific trained model |
| `propose_training_plan` | After a short dialogue (or a `[Background task` message): user wants to **train** and dataset refs are decided |

## Prediction & Model Tools
- **predict_with_model**: Run predictions on a dataset using a trained model. Use when the user wants to make predictions, score new data, or test a model on a dataset. Requires a model name and a registered dataset ref.
- **evaluate_model**: Evaluate a trained model's performance on a labeled dataset. Use when the user asks about model accuracy, performance metrics, or wants to compare how a model performs. Supports threshold optimization for imbalanced classification.
- **list_trained_models**: List all trained models with their metrics. Use when the user asks what models are available, wants to see trained models, or needs to pick a model for prediction.
- **get_model_info**: Get detailed info about a specific trained model. Use when the user asks about a specific model's features, hyperparameters, or training details.

## Decision Flow

1. **General / conversational question** → answer directly, no tool needed.
2. **User mentions finding, searching, looking for, or wanting datasets** →
   **ALWAYS call `search_datasets`**. NEVER answer dataset questions from your own
   knowledge — you MUST use the tool because it searches the **local** workspace
   for real, usable results. This includes ANY of these patterns:
   - "find me data for …", "search for datasets …", "look for … data"
   - "what datasets are good for …", "recommend a dataset for …"
   - "I need data for …", "get me some … data", "what data do I have?"
   Present the results as a numbered list and ask which local dataset ref to use.
3. **Analytical question** (e.g. "what trends …", "analyze …", "what is the distribution …")
   → `analyze_data`
4. **User wants to train / build a predictive model** → **Bias to action.** Treat plain-language goals
   (e.g. underwriting, risk, “high value” decisions, experiments to improve decisions) as **clear enough**
   to plan a supervised pipeline on workspace data — do **not** interrogate for target column, metric, or
   constraints unless the user is **explicitly** stuck or contradicts themselves.
   - If you do not have concrete local dataset **ref** names, call `search_datasets` first, then choose
     sensible refs from the results.
   - **Clarifying questions (rare):** You **may** ask **at most one** question **only** when something is
     **blocking**: e.g. no dataset ref and search returns nothing usable, empty or nonsensical goal, or
     mutually exclusive instructions. If you can make a reasonable assumption, **do not** ask.
   - If the user message starts with `[Background task` or says they plan to use **Run on my behalf**,
     same rule: at most one blocking question, then `propose_training_plan` so the plan card appears.
   - When refs and goal are workable, call `propose_training_plan` with exact `dataset_refs` and optional
     `preferences` / `recap_steps` (keep `recap_steps` short and plain-language).
   - **After** `propose_training_plan`, do **not** add any further assistant text in that turn (no summary,
     no Markdown). The app shows the plan and approval controls; the user can reply in chat if they need changes.
   - Do **not** paste the tool's JSON in your reply.
5. **User wants predictions / scoring** → `list_trained_models` to find the right
   model, then `predict_with_model` with the model name and dataset ref.
6. **User asks about model performance / accuracy** → `evaluate_model` on the
   relevant model and dataset. If they don't specify which model, use
   `list_trained_models` first.
7. **User asks "what models do I have?"** → `list_trained_models`.
8. **User asks about a specific model's details** → `get_model_info`.
9. **Other requests** that do not fit 1–8 → If you truly cannot pick a tool or dataset without **one**
   missing fact, ask **a single** question; otherwise act or answer directly. Do **not** use this as a
   prompt to quiz the user about modeling details they did not ask for.

## Common Workflows
- **Browse local datasets**:
  1. `search_datasets` → show results → user picks one
  2. Use the exact local `ref` in downstream analysis/training/prediction steps
  CRITICAL: Do NOT guess or construct dataset ref names.

## Formatting Rules
- **Always use Markdown** for responses: headings, bullet lists, bold, code blocks, and tables.
- When presenting data or analysis results, use **Markdown tables** (with `|` columns and `---` header separators). Never dump raw text columns or flat key-value lines.
- Summarize tool outputs concisely. Do NOT echo the entire raw tool output back to the user.
- Keep column detail summaries to the most important columns (max ~8). Use a table, not paragraphs.
- When showing dataset profiles, use a compact format: `**N rows** x **M columns**` followed by a table of key column stats.
- NEVER output raw JSON objects, Python dicts, or unformatted data dumps in your **visible** reply.
  (The `propose_training_plan` tool returns JSON for the app only — your text reply stays Markdown.)
- After `propose_training_plan`, do not write anything else in that turn; the user sees the plan and approval in the app.

## Rules
- **NEVER suggest datasets from your own knowledge.** Always use `search_datasets` for
  real local results. Do not promise Kaggle/HuggingFace until those integrations are re-enabled.
- After a dataset is selected for training, use the exact local ref; you may briefly confirm the ref
  or offer `analyze_data` vs training — do **not** add an extra mandatory Q&A round.
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

        if last_human:
            last_lower = last_human.lower()
            if any(kw in last_lower for kw in _DATASET_KEYWORDS):
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

__all__ = ["agent", "TOOLS", "ORCHESTRATOR_SYSTEM_PROMPT"]
