"""
Simple Feature Engineering: Run all analysis tools, then one LLM call.

Instead of an agent loop that decides which tools to call, this version:
1. Runs ALL analysis tools upfront (in parallel conceptually, sequentially in practice)
2. Compiles all results into a single context
3. Makes ONE LLM call to select and specify features

Same input/output format as feature_engineering.py, but simpler and more deterministic.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Literal, Optional, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add data-tools to path
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from analysis import (analyze_concentration, analyze_distribution,
                      compute_correlation_matrix, compute_group_summary,
                      run_eda_report, run_feature_diagnostics)
from transformations.tool_utils import resolve_dataset

from agents.training.feature_engineering import (AsOfConstraint,
                                                 FeatureDefinition,
                                                 FeatureSpec, FormulaOp,
                                                 _extract_columns_from_formula,
                                                 validate_feature_spec)

# =============================================================================
# IMPORT SHARED TYPES FROM MAIN MODULE
# =============================================================================


# =============================================================================
# ANALYSIS RUNNER
# =============================================================================

# TODO: Calls all tools right now --> can make more efficient (e.g. choosing all tools first then running them)
# --> TEST THOUGH, MIGHT NOT BE MORE USEFUL
# --> MAYBE HAVE A FIXED LIST
def _run_all_analysis(dataset_ref: str, target_column: str, task_type: str = "classification") -> dict[str, Any]:
    """
    Run all analysis tools on the dataset and collect results.
    
    Returns a dict with results from each tool (or error message if failed).
    """
    results = {}
    df = resolve_dataset(dataset_ref)
    
    # Get column lists
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target_column]
    
    # 1. EDA Report - always run
    try:
        results["eda_report"] = run_eda_report(dataset_ref)
    except Exception as e:
        results["eda_report"] = f"Error: {e}"
    
    # 2. Correlation Matrix (all numeric columns)
    try:
        results["correlation_matrix"] = compute_correlation_matrix(
            dataset_ref=dataset_ref,
            columns=numeric_cols[:15],  # Limit columns
            threshold=0.5,
        )
    except Exception as e:
        results["correlation_matrix"] = f"Error: {e}"
    
    # 3. Feature Diagnostics - leakage detection
    try:
        if feature_cols:
            results["feature_diagnostics"] = run_feature_diagnostics(
                dataset_ref=dataset_ref,
                feature_cols=feature_cols[:10],  # Limit to first 10
                target_col=target_column,
                task_type=task_type,
            )
        else:
            results["feature_diagnostics"] = "No numeric features to analyze"
    except Exception as e:
        results["feature_diagnostics"] = f"Error: {e}"
    
    # 4. Distribution Analysis - for numeric columns
    try:
        cols_to_analyze = [c for c in numeric_cols[:10] if c != target_column]
        if cols_to_analyze:
            results["distribution_analysis"] = analyze_distribution(
                dataset_ref=dataset_ref,
                columns=cols_to_analyze,
            )
        else:
            results["distribution_analysis"] = "No numeric columns to analyze"
    except Exception as e:
        results["distribution_analysis"] = f"Error: {e}"
    
    # 5. Group Summary - for categorical columns
    try:
        if cat_cols:
            group_results = {}
            for col in cat_cols[:5]:  # Limit to first 5
                try:
                    summary = compute_group_summary(
                        dataset_ref=dataset_ref,
                        group_by=col,
                        metrics=[target_column] if target_column in numeric_cols else None,
                        agg_funcs=["mean", "count"],
                    )
                    group_results[col] = summary
                except Exception as e:
                    group_results[col] = f"Error: {e}"
            results["group_summary"] = group_results
        else:
            results["group_summary"] = "No categorical columns to analyze"
    except Exception as e:
        results["group_summary"] = f"Error: {e}"
    
    # 6. Concentration Analysis - for key numeric columns
    try:
        # Analyze concentration of a few key columns
        concentration_results = {}
        for col in feature_cols[:3]:
            try:
                conc = analyze_concentration(
                    dataset_ref=dataset_ref,
                    value_col=col,
                )
                concentration_results[col] = conc
            except Exception as e:
                concentration_results[col] = f"Error: {e}"
        results["concentration_analysis"] = concentration_results if concentration_results else "No columns to analyze"
    except Exception as e:
        results["concentration_analysis"] = f"Error: {e}"
    
    return results


def _format_analysis_results(results: dict[str, Any]) -> str:
    """Format analysis results into a readable string for the LLM."""
    sections = []
    
    for tool_name, result in results.items():
        sections.append(f"## {tool_name.replace('_', ' ').title()}")
        if isinstance(result, str):
            sections.append(result)
        elif isinstance(result, dict):
            # Pretty print dict, but truncate if too long
            formatted = json.dumps(result, indent=2, default=str)
            if len(formatted) > 3000:
                formatted = formatted[:3000] + "\n... (truncated)"
            sections.append(f"```json\n{formatted}\n```")
        elif isinstance(result, list):
            for item in result:
                formatted = json.dumps(item, indent=2, default=str)
                sections.append(formatted)
        else:
            sections.append(str(result)[:2000])
        sections.append("")
    
    return "\n".join(sections)


# =============================================================================
# SYSTEM PROMPT (simplified - no tool instructions needed)
# =============================================================================

SIMPLE_SYSTEM_PROMPT = """You are a data scientist selecting features for a machine learning model.

You have been provided with complete analysis results from multiple tools. Use this information to decide which features to include.

## Feature Formula DSL
Specify each feature using one of these operations:

| Op | Example |
|----|---------|
| passthrough | `{"op": "passthrough", "column": "credit_score"}` |
| expression | `{"op": "expression", "expression": "loan_amount / income", "source_columns": ["loan_amount", "income"]}` |
| bin | `{"op": "bin", "column": "age", "bins": 5, "strategy": "quantile"}` |
| one_hot | `{"op": "one_hot", "column": "region", "drop_first": true}` |
| ordinal | `{"op": "ordinal", "column": "education", "order": ["hs", "bs", "ms", "phd"]}` |
| group_agg | `{"op": "group_agg", "column": "amount", "agg": "mean", "group_by": ["zip"]}` |
| rolling | `{"op": "rolling", "column": "amount", "agg": "sum", "window": 30, "order_by": "date", "partition_by": ["customer_id"]}` |
| date_extract | `{"op": "date_extract", "column": "signup_date", "part": "month"}` |
| date_diff | `{"op": "date_diff", "start_column": "start", "end_column": "end", "unit": "days"}` |

## Temporal Constraints
For time-sensitive features, set `as_of_constraint`:
```json
{"source_date_column": "transaction_date", "operator": "<"}
```
For static features, set `as_of_constraint` to null.

## Rules
- Never use forbidden columns
- Explain your reasoning based on the analysis results
- List excluded columns and why"""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

TaskType = Literal["classification", "regression"]


def run_feature_engineering_simple(
    train_ref: str,
    goal: str,
    target_column: str,
    grain: str,
    val_ref: Optional[str] = None,
    test_ref: Optional[str] = None,
    task_type: TaskType = "classification",
    forbidden_columns: list[str] = None,
    as_of_cutoff: Optional[str] = None,
    prediction_horizon: Optional[str] = None,
    model: str = "openai:gpt-5-mini",
) -> dict[str, Any]:
    """
    Run feature engineering with all analysis upfront, then one LLM call.
    
    IMPORTANT: Analysis is run ONLY on training data to prevent data leakage.
    The feature spec generated is based solely on training data statistics.
    
    This function generates the feature_spec. Use execute_feature_spec_split()
    to apply the spec to train/val/test sets (fitting on train, transforming all).
    
    Args:
        train_ref: Reference to the TRAINING dataset (analysis runs on this only)
        goal: Description of the ML goal
        target_column: The target column for prediction
        grain: What one row represents (e.g., policy_id)
        val_ref: Optional reference to validation dataset (for info only, not analyzed)
        test_ref: Optional reference to test dataset (for info only, not analyzed)
        task_type: Either "classification" or "regression"
        forbidden_columns: Columns not available at prediction time
        as_of_cutoff: Column representing the observation timestamp
        prediction_horizon: How far into the future we're predicting
        model: Model to use for the LLM call
    
    Returns:
        Dict with:
        - feature_spec: The feature specification
        - validation: Validation results
        - analysis_results: Raw analysis tool outputs (for debugging)
        - dataset_refs: {"train": train_ref, "val": val_ref, "test": test_ref}
    """
    forbidden_columns = forbidden_columns or []
    
    # Step 1: Run all analysis tools ON TRAINING DATA ONLY
    print(f"Running analysis tools on training data ({train_ref})...")
    analysis_results = _run_all_analysis(train_ref, target_column, task_type)
    
    # Step 2: Format results into context
    analysis_context = _format_analysis_results(analysis_results)
    
    # Step 3: Build the prompt
    context_parts = [
        f"## Goal\n{goal}",
        f"## Task Type\n**{task_type.upper()}**" + (
            " (predict a class/category)" if task_type == "classification" 
            else " (predict a continuous value)"
        ),
        f"## Training Dataset\n`{train_ref}`",
        f"## Target Column\n`{target_column}`",
        f"## Grain\n{grain} (what one row represents)",
    ]
    
    # Include info about split structure
    split_info = [f"- Training: `{train_ref}`"]
    if val_ref:
        split_info.append(f"- Validation: `{val_ref}`")
    if test_ref:
        split_info.append(f"- Test: `{test_ref}`")
    context_parts.append(f"## Data Split\n" + "\n".join(split_info))
    context_parts.append("*Note: Analysis above is from training data only to prevent leakage.*")
    
    if forbidden_columns:
        cols_str = ", ".join([f"`{c}`" for c in forbidden_columns])
        context_parts.append(f"## Forbidden Columns (DO NOT USE)\n{cols_str}")
    
    if as_of_cutoff:
        context_parts.append(f"## As-of Cutoff Column\n`{as_of_cutoff}`")
    
    if prediction_horizon:
        context_parts.append(f"## Prediction Horizon\n{prediction_horizon}")
    
    context_parts.append(f"## Analysis Results (Training Data)\n{analysis_context}")
    context_parts.append("\n## Instructions\nBased on the analysis above, select features and specify how to build them using the formula DSL.")
    
    user_message = "\n\n".join(context_parts)
    
    # Step 4: Make ONE LLM call with structured output
    print("Making LLM call for feature selection...")
    llm = init_chat_model(model)
    llm_with_structure = llm.with_structured_output(FeatureSpec)
    
    messages = [
        SystemMessage(content=SIMPLE_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]
    
    feature_spec = llm_with_structure.invoke(messages)
    
    # Convert to dict
    if hasattr(feature_spec, "model_dump"):
        feature_spec_dict = feature_spec.model_dump()
    else:
        feature_spec_dict = feature_spec
    
    # Step 5: Validate against training data columns
    validation = None
    if feature_spec_dict:
        try:
            df = resolve_dataset(train_ref)
            available_columns = set(df.columns)
        except:
            available_columns = set()
        
        validation = validate_feature_spec(
            feature_spec=feature_spec_dict,
            available_columns=available_columns,
            forbidden_columns=set(forbidden_columns),
            target_column=target_column,
        )
    
    return {
        "feature_spec": feature_spec_dict,
        "validation": validation,
        "analysis_results": analysis_results,
        "dataset_refs": {
            "train": train_ref,
            "val": val_ref,
            "test": test_ref,
        },
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_feature_engineering_simple",
]
