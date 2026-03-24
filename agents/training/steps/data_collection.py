"""
Data Collection Node - Step 2 of the Training Agent

Searches local sources and (optionally) external sources via the Dataset
Curator to find the best training dataset for the goal.
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

_DATA_RETRIEVAL_DIR = Path(__file__).parent.parent.parent / "data-retrieval"
if str(_DATA_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_RETRIEVAL_DIR))

from agent import DataRetrievalResult, retrieve_data
from utils import get_registered_dataset, register_dataset

from agents.training.utils.graph_stream_hooks import emit_graph_stream

if TYPE_CHECKING:
    from ..core.state import TrainingAgentState

_MIN_ROWS = 10
_MIN_COLUMNS = 2


# =============================================================================
# Helpers
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
    """Return average |correlation| between numeric features and the target.

    Falls back to 0.0 if the target is non-numeric or there are no features.
    """
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
    # Remove stop words that would match anything
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
    matches = goal_words & col_words
    return len(matches) / len(goal_words)


def _score_dataset(ref: str, goal: str = "") -> float:
    """Quality score for comparing datasets. Higher = better.

    Factors in size, completeness, feature diversity, feature–target signal,
    and goal relevance (how well column names match goal keywords).
    """
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
    signal_bonus = 1 + signal * 2  # range [1, ~3]

    # Relevance: strongly penalise datasets whose columns don't match the goal
    relevance = _goal_relevance(df, goal) if goal else 0.5
    relevance_multiplier = 0.1 + 0.9 * relevance  # range [0.1, 1.0]

    score = base * signal_bonus * relevance_multiplier
    return score


def _try_local(goal: str, selected_model: str | None,
               linked_datasets: list[str] | None) -> tuple[str | None, dict]:
    """Try the local data-retrieval agent. Returns (ref_or_None, audit_entry)."""
    model_type = selected_model or "machine learning"
    request = (
        f"I need to train a {model_type} model for the following goal:\n"
        f"'{goal}'\n\n"
        "Please find and prepare a dataset that includes:\n"
        "- A target variable suitable for this prediction task\n"
        "- Relevant features that could help predict the target\n"
        "- Enough rows for training and validation"
    )
    if linked_datasets:
        request += f"\n\nPreferred datasets: {', '.join(linked_datasets)}"

    result = retrieve_data(request)

    if isinstance(result, DataRetrievalResult):
        ref = result.dataset_ref
        if not _validate(ref):
            return ref, {
                "step": "data_collection", "action": "local_retrieval",
                "dataset_ref": ref, "rows": result.rows,
                "columns": result.columns, "source": result.source,
                "description": result.description,
            }
        return None, {"step": "data_collection", "action": "local_validation_failed",
                       "dataset_ref": ref, "issues": _validate(ref)}

    if isinstance(result, dict) and result.get("success"):
        ref = (result.get("parsed") or {}).get("dataset_ref")
        if ref and not _validate(ref):
            return ref, {"step": "data_collection", "action": "local_retrieval",
                         "dataset_ref": ref}
        return None, {"step": "data_collection", "action": "local_partial",
                       "parsed": result.get("parsed")}

    error_msg = result.get("error", "Unknown") if isinstance(result, dict) else str(result)
    return None, {"step": "data_collection", "action": "local_failed", "error": error_msg}


def _try_curator(goal: str) -> tuple[str | None, str, dict]:
    """Try the external Dataset Curator. Returns (ref_or_None, source, audit_entry)."""
    try:
        _CURATOR_DIR = Path(__file__).parent.parent.parent / "dataset_curator"
        if str(_CURATOR_DIR.parent) not in sys.path:
            sys.path.insert(0, str(_CURATOR_DIR.parent))
        from dataset_curator import DatasetCuratorResult, curate_dataset

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()

        result = asyncio.run(curate_dataset(goal))
    except Exception as exc:
        print(f"[data_collection] Curator error: {type(exc).__name__}: {exc}")
        return None, "", {"step": "data_collection", "action": "curator_failed",
                          "error": str(exc)}

    if not isinstance(result, DatasetCuratorResult):
        return None, "", {"step": "data_collection", "action": "curator_no_result"}

    try:
        df = pd.read_csv(result.csv_path)
    except Exception as exc:
        return None, "", {"step": "data_collection", "action": "curator_csv_error",
                          "error": str(exc)}

    ref = f"curator_{Path(result.csv_path).stem}"
    register_dataset(ref, df)

    if _validate(ref):
        return None, "", {"step": "data_collection", "action": "curator_validation_failed",
                          "dataset_ref": ref, "issues": _validate(ref)}

    source = "kaggle"
    if result.hf_sources:
        source = "huggingface" if not result.kaggle_sources else "kaggle+huggingface"

    return ref, source, {
        "step": "data_collection", "action": "curator_retrieval",
        "dataset_ref": ref, "csv_path": result.csv_path,
        "rows": len(df), "columns": list(df.columns), "source": source,
        "kaggle_sources": result.kaggle_sources, "hf_sources": result.hf_sources,
        "description": result.description,
    }


# =============================================================================
# Main entry point
# =============================================================================


def data_collection(state: "TrainingAgentState") -> "TrainingAgentState":
    """Collect the best available dataset for model training.

    1. Use a pre-registered linked dataset if one exists and is valid.
    2. Search local sources via the data-retrieval agent.
    3. If ``use_external_sources`` is enabled, also search Kaggle/HuggingFace
       via the Dataset Curator and keep whichever result is better.
    """
    resolved = state.get("resolved_dataset_ref")
    if resolved:
        emit_graph_stream({"phase": "data_collection", "message": f"Using pre-selected dataset: {resolved}"})
        audit_trace = list(state.get("audit_trace", []))
        df = get_registered_dataset(resolved)
        if df is not None:
            audit_trace.append({
                "step": "data_collection",
                "action": "pre_resolved",
                "dataset_ref": resolved,
                "rows": len(df),
                "columns": list(df.columns),
                "source": "pre-resolved",
            })
        return {
            **state,
            "collected_dataset_ref": resolved,
            "data_source": "pre-resolved",
            "audit_trace": audit_trace,
            "current_step": "data_collection",
            "error": None,
        }

    goal = state.get("goal", "")
    linked_datasets = state.get("linked_datasets")
    selected_model = state.get("selected_model")
    use_external = state.get("use_external_sources", False)

    audit_trace = list(state.get("audit_trace", []))
    explanations = list(state.get("explanations", []))

    if not goal:
        audit_trace.append({"step": "data_collection", "error": "No goal provided"})
        explanations.append("Data collection failed: No goal provided")
        return {**state, "collected_dataset_ref": None, "audit_trace": audit_trace,
                "explanations": explanations, "current_step": "data_collection",
                "error": "No goal provided"}

    # ----- Linked datasets (instant, no agent call) --------------------------
    if linked_datasets:
        for ref in linked_datasets:
            df = get_registered_dataset(ref)
            if df is not None and not _validate(ref):
                audit_trace.append({"step": "data_collection", "action": "used_linked_dataset",
                                    "dataset_ref": ref, "rows": len(df),
                                    "columns": list(df.columns), "source": "pre-registered"})
                explanations.append(
                    f"Using pre-registered dataset '{ref}' "
                    f"({len(df):,} rows, {len(df.columns)} columns).")
                emit_graph_stream({
                    "phase": "data_collection",
                    "message": f"Using linked dataset `{ref}` ({len(df):,} rows).",
                })
                return {**state, "collected_dataset_ref": ref,
                        "data_source": "pre-registered", "audit_trace": audit_trace,
                        "explanations": explanations, "current_step": "data_collection",
                        "error": None}

    # ----- Search local + (optionally) external, pick best ------------------
    emit_graph_stream({
        "phase": "data_collection",
        "message": "Searching locally and via retrieval agent…",
    })
    local_ref, local_audit = _try_local(goal, selected_model, linked_datasets)
    audit_trace.append(local_audit)

    ext_ref, ext_source, ext_audit = None, "", {}
    if use_external:
        emit_graph_stream({
            "phase": "data_collection",
            "message": "Searching external sources (Kaggle + Hugging Face)…",
        })
        print("[data_collection] Also searching external sources (Kaggle + HuggingFace)...")
        ext_ref, ext_source, ext_audit = _try_curator(goal)
        audit_trace.append(ext_audit)

    # Pick the best available dataset
    best_ref, best_source = None, None
    if local_ref and ext_ref:
        local_score = _score_dataset(local_ref, goal)
        ext_score = _score_dataset(ext_ref, goal)
        if ext_score > local_score:
            best_ref, best_source = ext_ref, ext_source
            print(f"[data_collection] External dataset scored higher ({ext_score:.0f} vs {local_score:.0f}) — using '{ext_ref}'")
        else:
            best_ref, best_source = local_ref, "local"
            print(f"[data_collection] Local dataset scored higher ({local_score:.0f} vs {ext_score:.0f}) — using '{local_ref}'")
    elif local_ref:
        best_ref, best_source = local_ref, "local"
    elif ext_ref:
        best_ref, best_source = ext_ref, ext_source

    if best_ref:
        df = get_registered_dataset(best_ref)
        rows = len(df) if df is not None else "?"
        cols = len(df.columns) if df is not None else "?"
        explanations.append(
            f"Data collection complete. Using '{best_ref}' "
            f"({rows} rows, {cols} columns) from {best_source}.")
        emit_graph_stream({
            "phase": "data_collection",
            "message": f"Selected dataset `{best_ref}` ({rows}×{cols}) from {best_source}.",
        })
    else:
        explanations.append("Data collection failed: no suitable dataset found.")
        emit_graph_stream({
            "phase": "data_collection",
            "message": "No suitable dataset found yet — will surface for review.",
        })

    return {
        **state,
        "collected_dataset_ref": best_ref,
        "data_source": best_source,
        "audit_trace": audit_trace,
        "explanations": explanations,
        "current_step": "data_collection",
        "error": None if best_ref else "No suitable dataset found",
    }
