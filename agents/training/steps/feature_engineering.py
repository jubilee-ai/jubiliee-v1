"""
Feature Selection & Specification Agent using LangChain's create_agent.

Step 4 of the training pipeline:
1. Agent uses analysis tools to understand the dataset
2. Agent selects features and specifies how to build them
3. Agent outputs a feature_spec (structured contract for step 5)

Uses create_agent with structured output for the final feature specification.
"""

import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from transformations.tool_utils import resolve_dataset

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add data-tools to path
# Path: steps -> training -> agents -> root -> tools/data-tools
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from analysis import (  # data_validation_tool,  # Excluded: has Pydantic schema issues with OpenAI strict mode
    concentration_analysis_tool, correlation_matrix_tool,
    distribution_analysis_tool, eda_report_tool, feature_diagnostics_tool,
    group_summary_tool, trend_analysis_tool)

# =============================================================================
# FORMULA DSL - Structured operations that map to data-tools
# =============================================================================

# Operation types that map to specific tools
OperationType = Literal[
    "passthrough",   # Use column as-is
    "expression",    # Computed expression (maps to add_column_tool)
    "bin",           # Discretize numeric (maps to bin_column_tool)
    "one_hot",       # One-hot encode categorical
    "ordinal",       # Ordinal encode with order
    "group_agg",     # Group-by aggregation (maps to agg_feature_tool)
    "rolling",       # Rolling window aggregation
    "date_extract",  # Extract datetime parts (year, month, day, etc.)
    "date_diff",     # Difference between dates
]

AggFunction = Literal["mean", "sum", "min", "max", "count", "std", "median", "nunique", "first", "last"]
DatePart = Literal["year", "month", "day", "dayofweek", "hour", "quarter", "weekofyear"]
BinStrategy = Literal["uniform", "quantile"]


class PassthroughOp(BaseModel):
    """Use a column as-is without transformation."""
    op: Literal["passthrough"] = "passthrough"
    column: str = Field(description="Source column name")


class ExpressionOp(BaseModel):
    """Compute a new column from an expression. Supports: +, -, *, /, **, comparisons, and functions (abs, sqrt, log, round, min, max)."""
    op: Literal["expression"] = "expression"
    expression: str = Field(
        description="Expression using column names. Examples: 'loan_amount / annual_income', 'age * 12', 'sqrt(income)'"
    )
    source_columns: list[str] = Field(description="Columns used in the expression")


class BinOp(BaseModel):
    """Discretize a numeric column into bins/buckets."""
    op: Literal["bin"] = "bin"
    column: str = Field(description="Numeric column to bin")
    bins: Union[int, list[float]] = Field(
        description="Number of bins (int) OR list of bin edges. Example: 5 or [0, 25, 50, 75, 100]"
    )
    strategy: BinStrategy = Field(default="quantile", description="'uniform' (equal-width) or 'quantile' (equal-frequency)")
    labels: Optional[list[str]] = Field(default=None, description="Optional labels for bins")


class OneHotOp(BaseModel):
    """One-hot encode a categorical column into multiple binary columns."""
    op: Literal["one_hot"] = "one_hot"
    column: str = Field(description="Categorical column to encode")
    drop_first: bool = Field(default=True, description="Drop first category to avoid multicollinearity")


class OrdinalOp(BaseModel):
    """Ordinal encode a categorical column with specified order."""
    op: Literal["ordinal"] = "ordinal"
    column: str = Field(description="Categorical column to encode")
    order: list[str] = Field(description="Categories in order from lowest to highest. Example: ['low', 'medium', 'high']")


class GroupAggOp(BaseModel):
    """Create an aggregated feature by computing a statistic within groups."""
    op: Literal["group_agg"] = "group_agg"
    column: str = Field(description="Column to aggregate")
    agg: AggFunction = Field(description="Aggregation function: mean, sum, min, max, count, std, median, nunique")
    group_by: list[str] = Field(description="Column(s) to group by")


class RollingOp(BaseModel):
    """Compute a rolling window aggregation over a time-ordered column."""
    op: Literal["rolling"] = "rolling"
    column: str = Field(description="Column to aggregate")
    agg: Literal["mean", "sum", "min", "max", "std"] = Field(description="Rolling aggregation function")
    window: int = Field(description="Window size (number of rows)")
    partition_by: Optional[list[str]] = Field(default=None, description="Optional column(s) to partition by before rolling")
    order_by: str = Field(description="Column to order by (usually a date/time column)")


class DateExtractOp(BaseModel):
    """Extract a part from a datetime column (year, month, day, etc.)."""
    op: Literal["date_extract"] = "date_extract"
    column: str = Field(description="Datetime column")
    part: DatePart = Field(description="Part to extract: year, month, day, dayofweek, hour, quarter, weekofyear")


class DateDiffOp(BaseModel):
    """Compute the difference between two date columns."""
    op: Literal["date_diff"] = "date_diff"
    start_column: str = Field(description="Start date column")
    end_column: str = Field(description="End date column")
    unit: Literal["days", "months", "years"] = Field(default="days", description="Unit for the difference")


# Union of all operation types
FormulaOp = Union[
    PassthroughOp,
    ExpressionOp,
    BinOp,
    OneHotOp,
    OrdinalOp,
    GroupAggOp,
    RollingOp,
    DateExtractOp,
    DateDiffOp,
]

# =============================================================================
# STRUCTURED OUTPUT SCHEMA
# =============================================================================

# TODO: Try to just run all analysis tools and then 1 LLM call to choose the features and then 1 LLM call to specify the formula

# TODO: Try to make choosing the features more deterministic or structured (eg. using a decision tree or something)


class AsOfConstraint(BaseModel):
    """
    Structured temporal constraint to prevent data leakage.
    
    Specifies that a source date column must be before the as-of cutoff date
    for this feature to be valid at prediction time.
    """
    source_date_column: str = Field(
        description="The date/timestamp column in the source data that must be checked (e.g., 'transaction_date', 'claim_date')"
    )
    operator: Literal["<", "<="] = Field(
        default="<",
        description="Comparison operator: '<' (strictly before) or '<=' (on or before)"
    )
    # Note: The reference (as_of_cutoff) is passed in as a parameter to the function,
    # so we don't need to repeat it here. The constraint means:
    # source_date_column < as_of_cutoff (the column passed to run_feature_engineering)


class FeatureDefinition(BaseModel):
    """Definition of a single feature for the ML model."""
    name: str = Field(description="Feature name (e.g., 'credit_score', 'income_to_loan_ratio', 'region_encoded')")
    formula: FormulaOp = Field(description="The operation to compute this feature. Must be one of the structured operation types.")
    grain: str = Field(description="Grain level this feature is computed at (e.g., 'loan_id', 'customer_id')")
    as_of_constraint: Optional[AsOfConstraint] = Field(
        default=None, 
        description="Temporal constraint to prevent leakage. Set if this feature uses time-sensitive data. Null if feature is static/non-temporal."
    )


class FeatureSpec(BaseModel):
    """Complete feature specification for the ML model."""
    features: list[FeatureDefinition] = Field(
        description="List of features to use in the model",
        min_length=1
    )
    reasoning: str = Field(
        description="Explanation of why these features were selected and how they relate to the prediction goal"
    )
    excluded_columns: list[str] = Field(
        default_factory=list,
        description="Columns that were intentionally excluded (forbidden, leaky, or redundant) with reasons"
    )


# All tools for the feature engineering agent (analysis only, no submission tool needed)
# Note: data_validation_tool excluded due to Pydantic schema incompatibility with OpenAI strict mode
FEATURE_ENGINEERING_TOOLS = [
    eda_report_tool,
    correlation_matrix_tool,
    distribution_analysis_tool,
    feature_diagnostics_tool,
    concentration_analysis_tool,
    group_summary_tool,
    trend_analysis_tool,
]

# =============================================================================
# SYSTEM PROMPT
# =============================================================================

FEATURE_ENGINEERING_SYSTEM_PROMPT = """You are a data scientist selecting features for a machine learning model.

## Your Goal
Analyze the dataset and decide which features to include.

## Your Tools
You have analysis tools to explore the data. Use them to understand the dataset before making feature decisions.

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
For time-sensitive features (rolling aggregations, event counts, etc.), set `as_of_constraint` to prevent leakage:
```json
{"source_date_column": "transaction_date", "operator": "<"}
```
For static features (age at signup, region), set `as_of_constraint` to null.

## Rules
- Never use forbidden columns
- Explain your reasoning
- List excluded columns and why"""

# =============================================================================
# SCHEMA HELPER
# =============================================================================

def _build_schema_context(dataset_ref: str) -> tuple[str, int]:
    """
    Build a schema summary from a dataset.
    
    Returns:
        Tuple of (schema_string, row_count)
    """
    
    try:
        df = resolve_dataset(dataset_ref)
    except Exception:
        return "", 0
    
    lines = []
    lines.append(f"**Shape:** {len(df):,} rows × {len(df.columns)} columns")
    lines.append("")
    lines.append("| Column | Type | Nulls | Unique | Sample Values |")
    lines.append("|--------|------|-------|--------|---------------|")
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        null_pct = df[col].isna().mean()
        null_str = f"{null_pct:.0%}" if null_pct > 0 else "0%"
        unique = df[col].nunique()
        
        # Get sample values
        non_null = df[col].dropna()
        if len(non_null) > 0:
            if dtype in ("object", "string", "category"):
                samples = non_null.value_counts().head(3).index.tolist()
                sample_str = ", ".join([f"'{s}'" for s in samples[:3]])
            else:
                sample_str = f"[{non_null.min():.2g}, {non_null.max():.2g}]" if "float" in dtype or "int" in dtype else str(non_null.iloc[0])[:20]
        else:
            sample_str = "all null"
        
        # Truncate long sample strings
        if len(sample_str) > 40:
            sample_str = sample_str[:37] + "..."
        
        lines.append(f"| `{col}` | {dtype} | {null_str} | {unique:,} | {sample_str} |")
    
    return "\n".join(lines), len(df)


# =============================================================================
# FEATURE VALIDATION
# =============================================================================

def _extract_columns_from_formula(formula: dict) -> set[str]:
    """Extract all column references from a formula operation."""
    cols = set()
    op = formula.get("op")
    
    if op == "passthrough":
        cols.add(formula.get("column", ""))
    elif op == "expression":
        cols.update(formula.get("source_columns", []))
    elif op == "bin":
        cols.add(formula.get("column", ""))
    elif op == "one_hot":
        cols.add(formula.get("column", ""))
    elif op == "ordinal":
        cols.add(formula.get("column", ""))
    elif op == "group_agg":
        cols.add(formula.get("column", ""))
        cols.update(formula.get("group_by", []))
    elif op == "rolling":
        cols.add(formula.get("column", ""))
        cols.add(formula.get("order_by", ""))
        cols.update(formula.get("partition_by") or [])
    elif op == "date_extract":
        cols.add(formula.get("column", ""))
    elif op == "date_diff":
        cols.add(formula.get("start_column", ""))
        cols.add(formula.get("end_column", ""))
    
    # Remove empty strings
    cols.discard("")
    return cols


def validate_feature_spec(
    feature_spec: dict,
    available_columns: set[str],
    forbidden_columns: set[str] = None,
    target_column: str = None,
) -> dict[str, Any]:
    """
    Validate that a feature specification is executable.
    
    Checks:
    1. All referenced columns exist in the dataset
    2. No forbidden columns are used
    3. Target column is not used as a feature
    4. All formulas have valid operation types
    
    Args:
        feature_spec: The feature specification dict
        available_columns: Set of column names in the dataset
        forbidden_columns: Set of forbidden column names
        target_column: The target column (should not be used as feature)
    
    Returns:
        Dict with:
        - valid: bool - whether the spec is valid
        - errors: list of critical errors (missing columns, etc.)
        - warnings: list of warnings (non-critical issues)
        - features_valid: dict mapping feature name to validation status
    """
    errors = []
    warnings = []
    features_valid = {}
    forbidden_columns = forbidden_columns or set()
    
    valid_ops = {"passthrough", "expression", "bin", "one_hot", "ordinal", 
                 "group_agg", "rolling", "date_extract", "date_diff"}
    
    features = feature_spec.get("features", [])
    
    for feat in features:
        feat_name = feat.get("name", "unknown")
        feat_errors = []
        feat_warnings = []
        
        formula = feat.get("formula", {})
        op = formula.get("op")
        
        # Check valid operation
        if op not in valid_ops:
            feat_errors.append(f"Invalid operation: '{op}'")
        
        # Extract and validate columns
        cols = _extract_columns_from_formula(formula)
        
        # Check for missing columns
        missing = cols - available_columns
        if missing:
            feat_errors.append(f"Missing columns: {missing}")
        
        # Check for forbidden columns
        used_forbidden = cols & forbidden_columns
        if used_forbidden:
            feat_errors.append(f"Uses forbidden columns: {used_forbidden}")
        
        # Check for target column usage
        if target_column and target_column in cols:
            feat_errors.append(f"Uses target column '{target_column}' as input (leakage)")
        
        # Validate as_of_constraint if present
        as_of = feat.get("as_of_constraint")
        if as_of and isinstance(as_of, dict):
            source_col = as_of.get("source_date_column")
            if source_col and source_col not in available_columns:
                feat_warnings.append(f"as_of_constraint references missing column: '{source_col}'")
        
        # Record validation status
        is_valid = len(feat_errors) == 0
        features_valid[feat_name] = {
            "valid": is_valid,
            "errors": feat_errors,
            "warnings": feat_warnings,
            "columns_used": list(cols),
        }
        
        # Aggregate errors/warnings
        for err in feat_errors:
            errors.append(f"Feature '{feat_name}': {err}")
        for warn in feat_warnings:
            warnings.append(f"Feature '{feat_name}': {warn}")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "features_valid": features_valid,
        "total_features": len(features),
        "valid_features": sum(1 for f in features_valid.values() if f["valid"]),
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

TaskType = Literal["classification", "regression"]

# TODO: Maybe add few shot prompting where based on the model we have 5 few shots each we only pass in the ones that are relevant to the model?
def run_feature_engineering(
    dataset_ref: str,
    goal: str,
    target_column: str,
    grain: str,
    task_type: TaskType = "classification",
    forbidden_columns: list[str] = None,
    as_of_cutoff: Optional[str] = None,
    prediction_horizon: Optional[str] = None,
    model: str = "openai:4",
    max_iterations: int = 10,
) -> dict[str, Any]:
    """
    Run the feature engineering agent on a dataset.
    
    Args:
        dataset_ref: Reference to the cleaned dataset
        goal: Description of the ML goal
        target_column: The target column for prediction
        grain: What one row represents (e.g., policy_id)
        task_type: Either "classification" or "regression"
        forbidden_columns: Columns not available at prediction time
        as_of_cutoff: Column representing the observation timestamp
        prediction_horizon: How far into the future we're predicting
        model: Model to use for the agent
        max_iterations: Maximum number of agent steps
    
    Returns:
        Dict with:
        - feature_spec: The feature specification (FeatureSpec model or None)
        - validation: Validation results with errors/warnings for invalid features
        - messages: All agent messages for debugging
    """
    agent = create_agent(
        model=model,
        tools=FEATURE_ENGINEERING_TOOLS,
        system_prompt=FEATURE_ENGINEERING_SYSTEM_PROMPT,
        response_format=ToolStrategy(schema=FeatureSpec),
    )
    
    forbidden_columns = forbidden_columns or []
    
    # Build the context message - keep it minimal to encourage exploration
    context_parts = [
        f"## Goal\n{goal}",
        f"## Task Type\n**{task_type.upper()}**" + (
            " (predict a class/category)" if task_type == "classification" 
            else " (predict a continuous value)"
        ),
        f"## Dataset\n`{dataset_ref}`",
        f"## Target Column\n`{target_column}`",
        f"## Grain\n{grain} (what one row represents)",
    ]
    
    if forbidden_columns:
        cols_str = ", ".join([f"`{c}`" for c in forbidden_columns])
        context_parts.append(f"## Forbidden Columns (DO NOT USE - these leak future information)\n{cols_str}")
    
    if as_of_cutoff:
        context_parts.append(f"## As-of Cutoff Column\n`{as_of_cutoff}` (observation timestamp - features must respect this)")
    
    if prediction_horizon:
        context_parts.append(f"## Prediction Horizon\n{prediction_horizon}")
    
    # Add instructions - keep them minimal to allow agent autonomy
    context_parts.append(
        "\n## Instructions\n"
        "Explore the dataset using the analysis tools. Based on what you discover, decide which features to include.\n"
        "When you're confident in your feature choices, output your final specification using the structured formula DSL."
    )
    
    initial_message = "\n\n".join(context_parts)
    
    # Run the agent
    result = agent.invoke(
        {"messages": [{"role": "user", "content": initial_message}]},
        {"recursion_limit": max_iterations},
    )
    
    # Extract the structured response (FeatureSpec)
    feature_spec = result.get("structured_response")
    
    # Convert to dict if it's a Pydantic model
    if feature_spec is not None and hasattr(feature_spec, "model_dump"):
        feature_spec_dict = feature_spec.model_dump()
    elif isinstance(feature_spec, dict):
        feature_spec_dict = feature_spec
    else:
        feature_spec_dict = None
    
    # Validate the feature spec against the dataset
    validation = None
    if feature_spec_dict is not None:
        # Get available columns from dataset
        try:
            from transformations.tool_utils import resolve_dataset
            df = resolve_dataset(dataset_ref)
            available_columns = set(df.columns)
        except Exception:
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
        "messages": result.get("messages", []),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_feature_engineering",
    "validate_feature_spec",
    "FeatureSpec",
    "FeatureDefinition",
    "AsOfConstraint",
    "FormulaOp",
    "PassthroughOp",
    "ExpressionOp",
    "BinOp",
    "OneHotOp",
    "OrdinalOp",
    "GroupAggOp",
    "RollingOp",
    "DateExtractOp",
    "DateDiffOp",
    "FEATURE_ENGINEERING_TOOLS",
]
