"""
Data Collection Node - Step 2 of the Training Agent

Reuses the data-retrieval agent to find and prepare datasets for training.
Falls back to the Dataset Curator (Kaggle + HuggingFace) when local retrieval
fails and ``use_external_sources`` is enabled.
"""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

# Add data-tools to path for registry access
# Path: steps -> training -> agents -> root -> tools/data-tools
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

# Import data retrieval agent
# Path: steps -> training -> agents -> data-retrieval
_DATA_RETRIEVAL_DIR = Path(__file__).parent.parent.parent / "data-retrieval"
if str(_DATA_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_RETRIEVAL_DIR))

from agent import DataRetrievalResult, retrieve_data
from utils import get_registered_dataset, register_dataset

if TYPE_CHECKING:
    from ..core.state import TrainingAgentState

_MIN_ROWS = 10
_MIN_COLUMNS = 2


def _validate_collected_dataset(ref: str) -> list[str]:
    """Validate a collected dataset meets minimum requirements for training.

    Returns a list of issue strings (empty = passed).
    """
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


def _try_dataset_curator(
    goal: str,
    audit_trace: list[dict],
    explanations: list[str],
) -> dict | None:
    """Run the Dataset Curator agent as a fallback data source.

    Returns a partial state dict on success, or None on failure.
    """
    try:
        _CURATOR_DIR = Path(__file__).parent.parent.parent / "dataset_curator"
        if str(_CURATOR_DIR.parent) not in sys.path:
            sys.path.insert(0, str(_CURATOR_DIR.parent))
        from dataset_curator import DatasetCuratorResult, curate_dataset

        result = asyncio.run(curate_dataset(goal))
    except Exception as exc:
        print(f"[data_collection] Dataset curator failed: {exc}")
        audit_trace.append({
            "step": "data_collection",
            "action": "curator_failed",
            "error": str(exc),
        })
        return None

    if not isinstance(result, DatasetCuratorResult):
        print(f"[data_collection] Curator returned non-structured result, skipping.")
        return None

    csv_path = result.csv_path
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        print(f"[data_collection] Could not read curator CSV '{csv_path}': {exc}")
        return None

    ref = f"curator_{Path(csv_path).stem}"
    register_dataset(ref, df)

    validation_issues = _validate_collected_dataset(ref)
    if validation_issues:
        issues_str = "; ".join(validation_issues)
        print(f"[data_collection] Curator dataset '{ref}' failed validation: {issues_str}")
        audit_trace.append({
            "step": "data_collection",
            "action": "curator_validation_failed",
            "dataset_ref": ref,
            "issues": validation_issues,
        })
        return None

    source = "kaggle"
    if result.hf_sources:
        source = "huggingface" if not result.kaggle_sources else "kaggle+huggingface"

    explanation = (
        f"Data collection complete via Dataset Curator. "
        f"Dataset '{ref}' with {len(df):,} rows and {len(df.columns)} columns. "
        f"Source: {source}. Description: {result.description}"
    )
    explanations.append(explanation)
    audit_trace.append({
        "step": "data_collection",
        "action": "used_dataset_curator",
        "dataset_ref": ref,
        "csv_path": csv_path,
        "rows": len(df),
        "columns": list(df.columns),
        "source": source,
        "kaggle_sources": result.kaggle_sources,
        "hf_sources": result.hf_sources,
    })

    return {
        "collected_dataset_ref": ref,
        "data_source": source,
        "audit_trace": audit_trace,
        "explanations": explanations,
        "error": None,
    }


# TODO: Potentially turn this into a subagent --> for example when merging datastes
def data_collection(state: "TrainingAgentState") -> "TrainingAgentState":
    """
    Step 2: Data Collection Agent
    Tools: same as data-retrieval agent
    - Reuse data-retrieval agent from ./data-retrieval
    - Finds the right data and puts together a dataset
    - Uses the datasets provided if available
    
    Priority:
    1. If linked_datasets contains a reference that exists in the registry, use it directly
    2. Otherwise, call the data retrieval agent to find/prepare data
    """
    goal = state.get("goal", "")
    linked_datasets = state.get("linked_datasets")
    selected_model = state.get("selected_model")
    
    # Validate inputs
    if not goal:
        return {
            **state,
            "collected_dataset_ref": None,
            "audit_trace": list(state.get("audit_trace", [])) + [{
                "step": "data_collection",
                "error": "No goal provided",
            }],
            "explanations": list(state.get("explanations", [])) + ["Data collection failed: No goal provided"],
            "current_step": "data_collection",
            "error": "No goal provided",
        }
    
    # -------------------------------------------------------------------------
    # PRIORITY 1: Check if linked_datasets reference existing registered datasets
    # -------------------------------------------------------------------------
    if linked_datasets:
        for ref in linked_datasets:
            df = get_registered_dataset(ref)
            if df is not None:
                # Validate before accepting
                validation_issues = _validate_collected_dataset(ref)
                if validation_issues:
                    print(f"[data_collection] Linked dataset '{ref}' failed validation: {validation_issues}")
                    continue

                audit_trace = list(state.get("audit_trace", []))
                explanations = list(state.get("explanations", []))
                
                explanation = (
                    f"Data collection complete. Using pre-registered dataset '{ref}' "
                    f"with {len(df):,} rows and {len(df.columns)} columns. "
                    f"Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}"
                )
                explanations.append(explanation)
                audit_trace.append({
                    "step": "data_collection",
                    "action": "used_linked_dataset",
                    "dataset_ref": ref,
                    "rows": len(df),
                    "columns": list(df.columns),
                    "source": "pre-registered",
                })
                
                return {
                    **state,
                    "collected_dataset_ref": ref,
                    "data_source": "pre-registered",
                    "audit_trace": audit_trace,
                    "explanations": explanations,
                    "current_step": "data_collection",
                    "error": None,
                }
    
    # -------------------------------------------------------------------------
    # PRIORITY 2: Call data retrieval agent to find/prepare data
    # -------------------------------------------------------------------------
    
    # Build a clear, actionable request for the data retrieval agent
    model_type = selected_model or "machine learning"
    request_parts = [
        f"I need to train a {model_type} model for the following goal:",
        f"'{goal}'",
        "",
        "Please find and prepare a dataset that includes:",
        "- A target variable suitable for this prediction task",
        "- Relevant features that could help predict the target",
        "- Enough rows for training and validation",
    ]
    
    if linked_datasets:
        request_parts.append("")
        request_parts.append(f"Preferred datasets to use: {', '.join(linked_datasets)}")
        request_parts.append("Please prioritize these datasets if they exist and are suitable.")
    
    request = "\n".join(request_parts)
    
    # Call the data retrieval agent
    result = retrieve_data(request)
    
    # Process the result
    audit_trace = list(state.get("audit_trace", []))
    explanations = list(state.get("explanations", []))
    
    if isinstance(result, DataRetrievalResult):
        # Success - we got a structured result
        dataset_ref = result.dataset_ref
        explanation = (
            f"Data collection complete. Retrieved dataset '{dataset_ref}' "
            f"with {result.rows} rows and {len(result.columns)} columns. "
            f"Source: {result.source}. Description: {result.description}"
        )
        explanations.append(explanation)
        audit_trace.append({
            "step": "data_collection",
            "dataset_ref": dataset_ref,
            "description": result.description,
            "rows": result.rows,
            "columns": result.columns,
            "source": result.source,
        })
        error = None
    elif isinstance(result, dict) and result.get("success"):
        # Partial success - got a response but not structured
        parsed = result.get("parsed") or {}
        dataset_ref = parsed.get("dataset_ref")
        explanation = f"Data retrieval completed with partial result: {result.get('response', 'No response')[:200]}"
        explanations.append(explanation)
        audit_trace.append({
            "step": "data_collection",
            "raw_response": result.get("response"),
            "parsed": parsed,
        })
        error = None if dataset_ref else "Could not extract dataset_ref from response"
    else:
        # Error
        dataset_ref = None
        error_msg = result.get("error", "Unknown error") if isinstance(result, dict) else str(result)
        explanation = f"Data collection failed: {error_msg}"
        explanations.append(explanation)
        audit_trace.append({
            "step": "data_collection",
            "error": error_msg,
        })
        error = error_msg
    
    # Validate the collected dataset before passing downstream
    if dataset_ref and not error:
        validation_issues = _validate_collected_dataset(dataset_ref)
        if validation_issues:
            issues_str = "; ".join(validation_issues)
            print(f"[data_collection] Retrieved dataset '{dataset_ref}' failed validation: {issues_str}")
            audit_trace.append({
                "step": "data_collection",
                "action": "validation_failed",
                "dataset_ref": dataset_ref,
                "issues": validation_issues,
            })
            error = f"Dataset validation failed: {issues_str}"

    # -------------------------------------------------------------------------
    # PRIORITY 3: Fall back to dataset curator (Kaggle + HuggingFace)
    # -------------------------------------------------------------------------
    if (not dataset_ref or error) and state.get("use_external_sources", False):
        print("[data_collection] Local retrieval failed, trying dataset curator (Kaggle + HuggingFace)...")
        curator_result = _try_dataset_curator(goal, audit_trace, explanations)
        if curator_result:
            return {**state, **curator_result, "current_step": "data_collection"}

    return {
        **state,
        "collected_dataset_ref": dataset_ref if not error else None,
        "data_source": "local" if (dataset_ref and not error) else None,
        "audit_trace": audit_trace,
        "explanations": explanations,
        "current_step": "data_collection",
        "error": error,
    }
