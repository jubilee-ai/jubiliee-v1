"""
Main Orchestrator Agent — Jubilee AI Chatbot

Uses langchain's create_agent to run a conversational loop that can spin up
specialised sub-agents on demand:

  1. Analysis sub-agent  (agents/analysis_agent_v2)
     → statistical analysis, pretrained-model inference, data exploration

  2. Training sub-agent  (agents/training/agent_simple)
     → full ML pipeline: model selection → data collection → cleaning →
       label/split → feature engineering → training → report

  3. Dataset Curator  (agents/dataset_curator)
     → search Kaggle / HuggingFace, download, profile, register datasets
"""

import asyncio
import json
import sys
import uuid
from typing import Any, Optional

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
# Tool 2 — Unified Dataset Search (local + Kaggle + HuggingFace)
# ============================================================================

class SearchDatasetsInput(BaseModel):
    query: str = Field(
        description="Search query for datasets (e.g. 'customer segmentation tabular', 'anomaly detection network traffic')"
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


def _search_local_datasets(query: str) -> list[dict]:
    """Search local registered datasets, SQL tables, and catalog for matches."""
    _DATA_TOOLS_DIR = _ROOT / "tools" / "data-tools"
    if str(_DATA_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_TOOLS_DIR))
    from utils import get_all_available_datasets

    all_ds = get_all_available_datasets()
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

    Searches local datasets first (already registered, SQL tables, catalog),
    then external sources (Kaggle, HuggingFace). Returns a unified numbered
    list the user can choose from.

    After the user picks a local dataset, it's ready to use immediately.
    For external datasets, use curate_dataset to download and register them.
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

    try:
        from agents.dataset_curator.agent import build_dataset_curator_agent

        async def _search_external():
            agent, client = await build_dataset_curator_agent()

            if source == "kaggle":
                prompt = (
                    f"Search Kaggle only (use search_datasets) for: {query}\n"
                    "Filter for CSV datasets. Return the top 5 results with: "
                    "title, ref (owner/slug), description (1 sentence), size_bytes, "
                    "download_count, and usability_rating."
                )
            elif source == "huggingface":
                prompt = (
                    f"Search HuggingFace only (use hub_repo_search with repo_types=['dataset']) for: {query}\n"
                    "Return the top 5 results with: id, description (1 sentence), downloads, tags."
                )
            else:
                prompt = (
                    f"Search BOTH Kaggle (search_datasets) and HuggingFace "
                    f"(hub_repo_search with repo_types=['dataset']) in parallel for: {query}\n"
                    "Return the top 3 from each source. For each result include: "
                    "name, source (kaggle/huggingface), identifier (owner/slug or org/repo), "
                    "description (1 sentence), and size or download count."
                )

            prompt += (
                f"\n\nNumber results starting from {next_num}. "
                "Do NOT download anything. Just search and return results. "
                "Format as a numbered list the user can choose from."
            )

            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            messages = result.get("messages", [])
            return messages[-1].content if messages else ""

        external_text = _run_async(_search_external())
        if external_text:
            if sections:
                sections.append("### External Datasets (Kaggle / HuggingFace)\n")
            sections.append(external_text)

    except Exception as exc:
        if sections:
            sections.append(f"\n*(External search failed: {exc})*")
        else:
            return f"Dataset search failed: {type(exc).__name__}: {exc}"

    if not sections:
        return f"No datasets found matching '{query}'."

    header = (
        "For **local** results, you can train on them immediately using the ref name. "
        "For **external** results, pick one and I'll download and register it for you.\n\n"
    )
    return header + "\n\n".join(sections)


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
    For external datasets: downloads, profiles, and registers them.

    Use this after the user picks a dataset from search_datasets results.
    Returns the registered dataset reference name that can be passed to
    train_model via linked_datasets.
    """
    _DATA_TOOLS_DIR = _ROOT / "tools" / "data-tools"
    if str(_DATA_TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(_DATA_TOOLS_DIR))
    from utils import get_registered_dataset

    if source.lower() in ("local", "local (sql)", "local (catalog)"):
        df = get_registered_dataset(identifier)
        if df is None:
            return f"Local dataset '{identifier}' not found in registry."

        from agents.dataset_curator.tools import profile_dataset
        profile_result = profile_dataset.invoke({"dataset_ref": identifier})

        return (
            f"Dataset **{identifier}** is already available locally ({len(df):,} rows, {len(df.columns)} cols).\n\n"
            f"You can train on it with: train_model(linked_datasets=[\"{identifier}\"])\n\n"
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
            f"You can now train on it with: train_model(linked_datasets=[\"{best_ref}\"])\n\n"
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
            f"You can now train on it with: train_model(linked_datasets=[\"{best_ref}\"])\n\n"
            f"### Profile\n{profile_result}\n\n"
            f"### All registered files\n{dl_result}"
        )

    else:
        return f"Unknown source '{source}'. Use 'kaggle' or 'huggingface'."


# ============================================================================
# Tool 4 — Model Training (delegates to the training LangGraph pipeline)
# ============================================================================

class TrainModelInput(BaseModel):
    goal: str = Field(
        description="Training objective (e.g. 'Train a loan-default prediction model on the Loan_default dataset')"
    )
    linked_datasets: Optional[list[str]] = Field(
        default=None,
        description="Dataset references the agent should use (e.g. ['csv_Loan_default', 'kaggle_my_data'])",
    )
    model_preference: Optional[str] = Field(
        default=None,
        description="Preferred model family: supervised | unsupervised | neural_networks",
    )
    use_external_sources: bool = Field(
        default=False,
        description="If True, the data collection step will also search Kaggle/HuggingFace "
        "when local data is insufficient. Set to True when the user wants to "
        "find data automatically or has no local datasets.",
    )


_last_training_state: dict[str, Any] = {}


def _run_training_to_completion(
    goal: str,
    linked_datasets: Optional[list[str]],
    model_preference: Optional[str],
    use_external_sources: bool = False,
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
        use_external_sources=use_external_sources,
    )

    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": goal}]}, config=config)

    final_state = dict(shared_state)

    completed_steps = [
        step for step in [
            "selected_model", "collected_dataset_ref", "cleaned_dataset_ref",
            "transformed_train_ref", "training_metrics", "report_path",
        ]
        if final_state.get(step) is not None
    ]
    print(f"[train_model] Pipeline finished. State keys populated: {completed_steps}", flush=True)

    if not final_state.get("training_metrics") and not final_state.get("selected_model"):
        msgs = result.get("messages", [])
        last_msgs = [
            m.content[:200] if hasattr(m, "content") else str(m)[:200]
            for m in msgs[-3:]
        ]
        print(f"[train_model] WARNING: Pipeline completed without training. Last messages: {last_msgs}", flush=True)

    _last_training_state.clear()
    _last_training_state.update(final_state)

    return final_state


@tool(args_schema=TrainModelInput)
def train_model(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    model_preference: Optional[str] = None,
    use_external_sources: bool = False,
) -> str:
    """Train a new ML model using the training sub-agent.

    Runs the full pipeline: model selection → data collection → cleaning →
    label/split definition → feature engineering → training → evaluation → report.

    IMPORTANT: Do NOT call this in the same turn as curate_dataset.
    If using an external dataset, first call curate_dataset, wait for it to
    return, then call train_model in a SEPARATE response using the exact
    linked_datasets ref that curate_dataset returned.

    This tool may take several minutes. Use it only after confirming with the
    user that they want to proceed with training.
    """
    try:
        result = _run_training_to_completion(goal, linked_datasets, model_preference, use_external_sources)

        selected_model = result.get("selected_model", "unknown")
        model_explanation = result.get("model_explanation", "")
        metrics = result.get("training_metrics") or {}
        report = result.get("report_path", "")
        model_weights = result.get("model_weights_path", "")

        if not metrics:
            error = result.get("error")
            return (
                f"Training pipeline did not produce results. "
                f"The agent may have stopped before reaching the training step.\n"
                f"Model family selected: {selected_model}\n"
                f"Error: {error}" if error else
                f"Training pipeline did not produce results. "
                f"The agent may have stopped before reaching the training step.\n"
                f"Model family selected: {selected_model}\n"
                f"Please try again or check LangSmith traces for details."
            )

        if not metrics.get("success", True) and not model_weights:
            summary = metrics.get("summary", "")
            recs = metrics.get("recommendations", "")
            return (
                f"Training failed.\n"
                f"  Model: {metrics.get('model_name', 'N/A')}\n"
                f"  Summary: {summary}\n"
                f"  Recommendations: {recs}\n"
                f"  Report: {report}"
            )

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
            ("silhouette_score", "Silhouette"),
            ("davies_bouldin", "Davies-Bouldin"),
            ("inertia", "Inertia"),
            ("reconstruction_loss", "Recon. Loss"),
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
| `search_datasets` | User wants to find, browse, or discover datasets — searches **local storage first**, then Kaggle and HuggingFace |
| `curate_dataset` | User picked a dataset — profiles local ones or downloads external ones |
| `analyze_data` | Analytical questions: statistics, trends, pretrained-model inference, data exploration |
| `train_model` | User explicitly wants to train / build a custom ML model |

## Decision Flow

1. **General / conversational question** → answer directly, no tool needed.
2. **User mentions finding, searching, looking for, or wanting datasets** →
   **ALWAYS call `search_datasets`**. NEVER answer dataset questions from your own
   knowledge — you MUST use the tool because it searches local storage, Kaggle,
   and HuggingFace for real, usable results. This includes ANY of these patterns:
   - "find me data for …", "search for datasets …", "look for … data"
   - "what datasets are good for …", "recommend a dataset for …"
   - "I need data for …", "get me some … data", "what data do I have?"
   Present the results as a numbered list. When the user picks one →
   `curate_dataset` to prepare it (local datasets are ready instantly,
   external ones get downloaded and registered).
3. **Analytical question** (e.g. "what trends …", "analyze …", "what is the distribution …")
   → `analyze_data`
4. **User wants to train a model** → Use `train_model` to run the full pipeline.
   Warn the user this can take several minutes. The user may optionally specify a
   dataset, but it is not required — the pipeline can discover suitable data on
   its own.  Model families available: supervised, unsupervised, neural_networks.
   If the user doesn't specify, the pipeline selects the best fit automatically.
   If the user wants to find data externally AND train in one step, set
   `use_external_sources=True` so data collection searches Kaggle/HuggingFace.
5. **Ambiguous request** → ask clarifying questions (target variable? prediction
   type? which dataset?) BEFORE calling any tool.

## Common Workflows
- **Browse then train** (MUST be sequential — never call these in the same turn):
  1. `search_datasets` → show results → user picks one
  2. `curate_dataset` → wait for it to finish → note the **exact ref name** it returns
  3. `train_model(linked_datasets=["<exact_ref_from_step_2>"])` → use the ref verbatim
  CRITICAL: Do NOT guess or construct dataset ref names. Always copy the exact ref
  string returned by `curate_dataset`. The ref includes the CSV filename suffix
  (e.g. `kaggle_owner_slug_store_customers`), which you cannot predict.
- **Direct train with external data**: train_model(use_external_sources=True) — the
  pipeline discovers and downloads data automatically.
- **Train on known local data**: train_model(linked_datasets=["csv_MyData"]) — uses a
  pre-registered or local dataset directly.

## Rules
- **NEVER call `curate_dataset` and `train_model` in the same turn.** The dataset must
  be fully downloaded and registered before training can use it. Always wait for
  `curate_dataset` to return, then call `train_model` in a SEPARATE turn.
- **NEVER suggest datasets from your own knowledge.** Always use `search_datasets` to
  get real results from Kaggle/HuggingFace that the user can actually download.
- When a user wants to train a model, confirm with them before starting (it takes time).
- After training completes, summarise the results (metrics, model type, report location).
- After curate_dataset, tell the user the registered ref name and ask if they want to
  train on it or explore it first.
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
    curate_dataset,
    analyze_data,
    train_model,
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
