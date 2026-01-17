"""
Data Collection Node - Step 2 of the Training Agent

Reuses the data-retrieval agent to find and prepare datasets for training.
"""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add data-tools to path for registry access
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

# Import data retrieval agent
_DATA_RETRIEVAL_DIR = Path(__file__).parent.parent / "data-retrieval"
if str(_DATA_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_RETRIEVAL_DIR))

from agent import DataRetrievalResult, retrieve_data
from utils import get_registered_dataset

if TYPE_CHECKING:
    from .agent import TrainingAgentState


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
                # Found a pre-registered dataset - use it directly
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
    
    return {
        **state,
        "collected_dataset_ref": dataset_ref,
        "audit_trace": audit_trace,
        "explanations": explanations,
        "current_step": "data_collection",
        "error": error,
    }
