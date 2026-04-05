"""
Data Collection Node - Step 2 of the Training Agent

Uses create_agent with data retrieval tools to find and prepare the best
dataset for the user's goal. The agent searches, loads, joins, and validates
data autonomously — guided by a strong system prompt.
"""

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from data_loader import dataset_get_tool
from join_merge import join_merge_tool
from retrieval import catalog_search_tool, list_datasets_tool
from sql_query import get_sql_schema_tool, sql_query_tool
from utils import generate_unique_id, get_registered_dataset, register_dataset

from agents.training.utils.graph_stream_hooks import emit_graph_stream

if TYPE_CHECKING:
    from ..core.state import TrainingAgentState

_MIN_ROWS = 10
_MIN_COLUMNS = 2


# =============================================================================
# HELPERS (also used by tests)
# =============================================================================


def _validate(ref: str) -> list[str]:
    """Return a list of issues (empty = valid)."""
    df = get_registered_dataset(ref)
    if df is None:
        return [f"Dataset '{ref}' not found in registry"]
    issues: list[str] = []
    if len(df) < _MIN_ROWS:
        issues.append(f"Too few rows: {len(df)} (minimum {_MIN_ROWS})")
    if len(df.columns) < _MIN_COLUMNS:
        issues.append(f"Too few columns: {len(df.columns)} (minimum {_MIN_COLUMNS})")
    if df.shape[0] > 0 and df.isna().all(axis=0).any():
        all_null = [c for c in df.columns if df[c].isna().all()]
        issues.append(f"Entirely null columns: {all_null}")
    if len(df) > 0 and len(df.drop_duplicates()) == 1:
        issues.append("All rows are identical")
    return issues


def _infer_target_column(df: pd.DataFrame, goal: str) -> str | None:
    """Best-effort guess at which column is the prediction target."""
    import numpy as np

    if not goal:
        return None
    goal_lower = goal.lower()
    cols = list(df.columns)
    for col in cols:
        if col.lower() in goal_lower:
            return col
    target_keywords = [
        "target", "label", "class", "churn", "price", "status",
        "outcome", "approved", "default", "fraud", "condition",
        "survived", "diagnosis", "y",
    ]
    for kw in target_keywords:
        for col in cols:
            if kw in col.lower():
                return col
    return None


def _signal_strength(df: pd.DataFrame, target_col: str) -> float:
    """Return average |correlation| between numeric features and the target."""
    import numpy as np

    if target_col not in df.columns:
        return 0.0
    target = df[target_col]
    if not np.issubdtype(target.dtype, np.number):
        try:
            target = target.astype(float)
        except (ValueError, TypeError):
            return 0.0
    numeric = df.select_dtypes(include="number").drop(columns=[target_col], errors="ignore")
    if numeric.empty:
        return 0.0
    corrs = numeric.corrwith(target).abs().dropna()
    return float(corrs.mean()) if len(corrs) > 0 else 0.0


def _goal_relevance(df: pd.DataFrame, goal: str) -> float:
    """Score how well a dataset's columns match the goal keywords. Returns 0..1."""
    if not goal:
        return 0.0
    goal_words = set(re.sub(r"[^a-z0-9 ]", " ", goal.lower()).split())
    goal_words -= {
        "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "is",
        "it", "by", "as", "at", "be", "if", "do", "from", "with", "that",
        "this", "will", "can", "like", "such", "based", "using", "train",
        "model", "predict", "build", "use", "data", "dataset", "learning",
        "whether", "score", "neural", "network", "deep",
    }
    if not goal_words:
        return 0.0
    col_words = set()
    for col in df.columns:
        col_words.update(re.sub(r"[^a-z0-9 ]", " ", col.lower()).split())
    return len(goal_words & col_words) / len(goal_words)


def _score_dataset(ref: str, goal: str = "") -> float:
    """Quality score for comparing datasets. Higher = better."""
    import numpy as np

    df = get_registered_dataset(ref)
    if df is None:
        return -1.0

    n_rows = len(df)
    null_frac = df.isnull().sum().sum() / max(df.size, 1)

    useful_cols = 0
    for c in df.columns:
        nuniq = df[c].nunique()
        if nuniq <= 1:
            continue
        is_id_like = (nuniq >= n_rows * 0.95) and not np.issubdtype(df[c].dtype, np.number)
        if not is_id_like:
            useful_cols += 1

    base = n_rows * useful_cols * (1 - null_frac)

    target_col = _infer_target_column(df, goal) if goal else None
    signal = _signal_strength(df, target_col) if target_col else 0.0
    signal_bonus = 1 + signal * 2

    relevance = _goal_relevance(df, goal) if goal else 0.5
    relevance_multiplier = 0.1 + 0.9 * relevance

    return base * signal_bonus * relevance_multiplier


# =============================================================================
# TOOLS
# =============================================================================


class MarkCollectionCompleteInput(BaseModel):
    dataset_ref: str = Field(description="The dataset reference to finalize")
    description: str = Field(description="Brief description of what this dataset contains")


def _build_mark_collection_complete(goal: str = ""):
    @tool(args_schema=MarkCollectionCompleteInput)
    def mark_collection_complete(dataset_ref: str, description: str) -> str:
        """
        Mark a dataset as the final selection for model training.

        Call this ONLY after you have found and loaded a good dataset.
        It validates the dataset meets minimum requirements before accepting.
        """
        df = get_registered_dataset(dataset_ref)
        if df is None:
            return (
                f"Dataset '{dataset_ref}' not found in registry. "
                f"Make sure you loaded it with dataset_get_tool or sql_query_tool first."
            )

        issues = []
        if len(df) < _MIN_ROWS:
            issues.append(f"Too few rows: {len(df)} (minimum {_MIN_ROWS})")
        if len(df.columns) < _MIN_COLUMNS:
            issues.append(f"Too few columns: {len(df.columns)} (minimum {_MIN_COLUMNS})")
        if df.shape[0] > 0 and df.isna().all(axis=0).any():
            all_null = [c for c in df.columns if df[c].isna().all()]
            issues.append(f"Entirely null columns: {all_null}")

        if issues:
            body = "\n".join(f"- {i}" for i in issues)
            return f"Cannot finalize — issues found:\n{body}\n\nFix these or find a better dataset."

        final_ref = generate_unique_id("collected")
        # In-memory + R2 only: skip re-materializing in SQL warehouse (same frame; avoids slow duplicate to_sql)
        register_dataset(final_ref, df, persist=True, register_sql=False)

        return (
            f"COLLECTION COMPLETE\n"
            f"Dataset: `{final_ref}`\n"
            f"Rows: {len(df):,}\n"
            f"Columns: {len(df.columns)} — {list(df.columns)}\n"
            f"Description: {description}"
        )

    return mark_collection_complete


COLLECTION_TOOLS = [
    catalog_search_tool,
    list_datasets_tool,
    dataset_get_tool,
    get_sql_schema_tool,
    sql_query_tool,
    join_merge_tool,
]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a data engineer. Find one solid training dataset for the user's goal — quickly and pragmatically.

## Workflow (keep tool calls few)

1. **Understand the goal** — What to predict? What is one row?
2. **`catalog_search_tool` first** — 1–2 searches (e.g. task + domain). Pick the best match.
3. **`list_datasets_tool` only if needed** — Use when catalog is weak, or you need SQL table names / already-loaded refs. Do not call it "just to browse."
4. **Load** — `dataset_get_tool` (catalog) or `sql_query_tool` (SQL). Use a LIMIT on first probe if the table is huge.
5. **Join** — Only if two tables clearly share a key and the join adds obvious value; otherwise skip.
6. **Finalize** — `mark_collection_complete` with the loaded `dataset_ref` and a short description.

## Tools
- `catalog_search_tool`, `list_datasets_tool`, `dataset_get_tool`, `get_sql_schema_tool`, `sql_query_tool`, `join_merge_tool`, `mark_collection_complete`

## Rules
- Prefer the fastest path: search → load → done. No extra exploration loops.
- Use the `dataset_ref` from the latest tool output when chaining.
- After `mark_collection_complete` succeeds, STOP — no summary text."""


# =============================================================================
# AGENT + RUNNER
# =============================================================================


def run_data_collection(
    goal: str,
    selected_model: str | None = None,
    linked_datasets: list[str] | None = None,
    model: str = "openai:gpt-5.4-mini",
    max_iterations: int = 8,
) -> dict:
    """
    Run the data collection agent to find the best dataset for the goal.

    Returns dict with: dataset_ref, description, messages
    """
    complete_tool = _build_mark_collection_complete(goal)
    tools = [*COLLECTION_TOOLS, complete_tool]

    agent = create_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT)

    # Build the initial message
    parts = [f"Find the best dataset for this ML task:\n\n**Goal:** {goal}"]
    if selected_model:
        parts.append(f"**Model type:** {selected_model}")
    if linked_datasets:
        parts.append(f"**Check these first:** {', '.join(linked_datasets)}")
    parts.append("\nUse the workflow above — minimal steps, then finalize.")

    initial_message = "\n".join(parts)

    # ~2 graph steps per tool round (tighter than *3 — agent still completes reliably)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": initial_message}]},
        {"recursion_limit": max(24, max_iterations * 2 + 8)},
    )

    # Extract result from messages
    dataset_ref = None
    description = None
    for msg in result.get("messages", []):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            if "COLLECTION COMPLETE" in msg.content and "Dataset:" in msg.content:
                for line in msg.content.split("\n"):
                    if "Dataset:" in line and "`" in line:
                        dataset_ref = line.split("`")[1]
                    if "Description:" in line:
                        description = line.replace("Description:", "").strip()

    return {
        "dataset_ref": dataset_ref,
        "description": description,
        "messages": result.get("messages", []),
    }


# =============================================================================
# GRAPH NODE ENTRY POINT
# =============================================================================


def data_collection(state: "TrainingAgentState") -> "TrainingAgentState":
    """Collect the best available dataset for model training."""

    # Fast path: pre-resolved dataset
    resolved = state.get("resolved_dataset_ref")
    if resolved:
        emit_graph_stream({"phase": "data_collection", "message": f"Using pre-selected dataset: {resolved}"})
        audit_trace = list(state.get("audit_trace", []))
        df = get_registered_dataset(resolved)
        if df is not None:
            audit_trace.append({
                "step": "data_collection", "action": "pre_resolved",
                "dataset_ref": resolved, "rows": len(df),
                "columns": list(df.columns), "source": "pre-resolved",
            })
        return {
            **state, "collected_dataset_ref": resolved, "data_source": "pre-resolved",
            "audit_trace": audit_trace, "current_step": "data_collection", "error": None,
        }

    goal = state.get("goal", "")
    linked_datasets = state.get("linked_datasets")
    selected_model = state.get("selected_model")
    audit_trace = list(state.get("audit_trace", []))
    explanations = list(state.get("explanations", []))

    if not goal:
        audit_trace.append({"step": "data_collection", "error": "No goal provided"})
        return {**state, "collected_dataset_ref": None, "audit_trace": audit_trace,
                "explanations": explanations + ["Data collection failed: No goal provided"],
                "current_step": "data_collection", "error": "No goal provided"}

    # Fast path: valid linked dataset
    if linked_datasets:
        for ref in linked_datasets:
            df = get_registered_dataset(ref)
            if df is not None and len(df) >= _MIN_ROWS and len(df.columns) >= _MIN_COLUMNS:
                audit_trace.append({
                    "step": "data_collection", "action": "used_linked_dataset",
                    "dataset_ref": ref, "rows": len(df),
                    "columns": list(df.columns), "source": "pre-registered",
                })
                explanations.append(f"Using linked dataset '{ref}' ({len(df):,} rows, {len(df.columns)} columns).")
                emit_graph_stream({"phase": "data_collection", "message": f"Using linked dataset `{ref}` ({len(df):,} rows)."})
                return {**state, "collected_dataset_ref": ref, "data_source": "pre-registered",
                        "audit_trace": audit_trace, "explanations": explanations,
                        "current_step": "data_collection", "error": None}

    # Run the data collection agent
    emit_graph_stream({"phase": "data_collection", "message": "Searching for the best dataset…"})

    result = run_data_collection(
        goal=goal,
        selected_model=selected_model,
        linked_datasets=linked_datasets,
    )

    best_ref = result["dataset_ref"]
    description = result.get("description", "")

    if best_ref:
        df = get_registered_dataset(best_ref)
        rows = len(df) if df is not None else "?"
        cols = len(df.columns) if df is not None else "?"
        audit_trace.append({
            "step": "data_collection", "action": "agent_retrieval",
            "dataset_ref": best_ref, "rows": rows,
            "columns": list(df.columns) if df is not None else [],
            "source": "agent", "description": description,
        })
        explanations.append(f"Data collection complete. Using '{best_ref}' ({rows} rows, {cols} columns). {description}")
        emit_graph_stream({"phase": "data_collection", "message": f"Selected `{best_ref}` ({rows} rows × {cols} cols)."})
    else:
        explanations.append("Data collection failed: no suitable dataset found.")
        emit_graph_stream({"phase": "data_collection", "message": "No suitable dataset found."})

    return {
        **state,
        "collected_dataset_ref": best_ref,
        "data_source": "agent" if best_ref else None,
        "audit_trace": audit_trace,
        "explanations": explanations,
        "current_step": "data_collection",
        "error": None if best_ref else "No suitable dataset found",
    }
