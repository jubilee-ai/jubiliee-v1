"""
data_validation - Fast, deterministic validation for data quality.

Checks schema/types, null thresholds, invalid ranges, uniqueness constraints,
and category drift. Produces ranked violations with severity and suggested repairs.

Run first on any new dataset or after joins/unions.
"""

import re
import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset

from .analysis_sidecar import attach_analysis_sidecar

# =============================================================================
# HELPERS
# =============================================================================

RuleType = Literal["not_null", "range", "unique", "dtype", "allowed_values", "regex"]


def _severity(pct: float, thresholds: tuple[float, float] = (0.1, 0.01)) -> str:
    """Determine severity based on violation percentage."""
    high, med = thresholds
    return "high" if pct > high else "medium" if pct > med else "low"


def _violation(
    rule: str,
    column: str,
    count: int,
    total: int,
    examples: list | None,
    action: str,
    params: dict,
    severity_thresholds: tuple[float, float] = (0.1, 0.01),
) -> dict:
    """Build a standardized violation dict."""
    pct = count / total if total > 0 else 0
    # Filter out None values from params for cleaner agent consumption
    clean_params = {k: v for k, v in params.items() if v is not None}
    return {
        "rule": rule,
        "column": column,
        "severity": _severity(pct, severity_thresholds),
        "count": int(count),
        "pct": float(round(pct, 4)),  # Ensure native float, not np.float64
        "examples": examples,
        "suggested_fix": {"action": action, "params": clean_params},
    }


def _get_examples(series: pd.Series, mask: pd.Series, n: int = 5) -> list:
    """Extract unique example values from masked series."""
    return series[mask].drop_duplicates().head(n).tolist()


def _get_value_counts_examples(series: pd.Series, mask: pd.Series, n: int = 3) -> list:
    """Extract value count examples from masked series."""
    counts = series[mask].value_counts().head(n)
    return [{"value": str(v), "count": int(c)} for v, c in counts.items()]


# =============================================================================
# VALIDATION RULES
# =============================================================================


def _check_not_null(df: pd.DataFrame, columns: list[str], threshold: float) -> list[dict]:
    """Check columns for null values exceeding threshold."""
    violations = []
    for col in columns:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) if len(df) > 0 else 0

        if null_pct > threshold:
            action = "impute" if null_pct < 0.5 else "drop_column"
            params = {"strategy": "median"} if null_pct < 0.5 else {}
            violations.append(
                _violation("not_null", col, null_count, len(df), None, action, params, (0.5, 0.2))
            )
    return violations


def _check_range(df: pd.DataFrame, column: str, min_val: Optional[float], max_val: Optional[float]) -> list[dict]:
    """Check numeric column for values outside range."""
    if column not in df.columns:
        return []

    col_data = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(col_data):
        return []

    mask = pd.Series(False, index=col_data.index)
    if min_val is not None:
        mask |= col_data < min_val
    if max_val is not None:
        mask |= col_data > max_val

    count = mask.sum()
    if count == 0:
        return []

    return [_violation(
        "range", column, count, len(df),
        _get_examples(col_data, mask),
        "clip", {"min": min_val, "max": max_val}
    )]


def _check_unique(df: pd.DataFrame, column: str, threshold: float) -> list[dict]:
    """Check column for duplicate values."""
    if column not in df.columns:
        return []

    dup_mask = df[column].duplicated(keep=False)
    dup_count = df[column].duplicated().sum()
    dup_pct = dup_count / len(df) if len(df) > 0 else 0

    if dup_pct <= threshold:
        return []

    return [_violation(
        "unique", column, dup_count, len(df),
        _get_value_counts_examples(df[column], dup_mask),
        "dedupe", {"keep": "first"}
    )]


def _check_dtype(df: pd.DataFrame, column: str, expected: str) -> list[dict]:
    """Check column dtype matches expected."""
    if column not in df.columns:
        return []

    actual = str(df[column].dtype).lower()
    expected_lower = expected.lower()

    # Dtype compatibility groups
    dtype_groups = {
        "int": {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"},
        "float": {"float16", "float32", "float64"},
        "str": {"object", "string"},
        "bool": {"bool"},
        "datetime": {"datetime64[ns]", "datetime64"},
        "category": {"category"},
    }

    # Find if expected is a group alias or actual dtype
    for group, types in dtype_groups.items():
        if expected_lower == group or expected_lower in types:
            if actual in types:
                return []
            break

    # Substring match fallback
    if expected_lower in actual or actual in expected_lower:
        return []

    return [_violation(
        "dtype", column, len(df), len(df),
        [f"expected {expected}, got {df[column].dtype}"],
        "cast", {"dtype": expected},
        (0.5, 0.5)  # Always medium for dtype mismatch
    )]


def _check_allowed_values(df: pd.DataFrame, column: str, values: list) -> list[dict]:
    """Check column values are in allowed set."""
    if column not in df.columns:
        return []

    col_data = df[column].dropna()
    invalid_mask = ~col_data.isin(values)
    count = invalid_mask.sum()

    if count == 0:
        return []

    return [_violation(
        "allowed_values", column, count, len(df),
        _get_value_counts_examples(col_data, invalid_mask),
        "replace_values", {"mapping": "review invalid values"}
    )]


def _check_regex(df: pd.DataFrame, column: str, pattern: str) -> list[dict]:
    """Check column values match regex pattern."""
    if column not in df.columns:
        return []

    col_data = df[column].dropna().astype(str)

    try:
        # Use vectorized str.match (faster than apply)
        matches = col_data.str.match(pattern, na=False)
    except re.error:
        return [_violation(
            "regex", column, 0, len(df),
            [f"Invalid regex pattern: {pattern}"],
            "fix_pattern", {},
            (1.0, 1.0)  # Always low
        )]

    invalid_mask = ~matches
    count = invalid_mask.sum()

    if count == 0:
        return []

    return [_violation(
        "regex", column, count, len(df),
        _get_examples(col_data, invalid_mask),
        "regex_replace", {"pattern": pattern, "action": "review"}
    )]


def _infer_schema(df: pd.DataFrame) -> list[dict]:
    """Infer schema from DataFrame."""
    return [
        {
            "column": col,
            "dtype": str(df[col].dtype),
            "nullable": bool(df[col].isna().any()),
            "unique_count": int(df[col].nunique()),
        }
        for col in df.columns
    ]


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def validate_dataset(
    dataset_ref: str,
    rules: list[dict],
    thresholds: Optional[dict] = None,
    sample_n: Optional[int] = None,
) -> dict:
    """
    Validate a dataset against a set of rules.
    
    Args:
        dataset_ref: Reference to dataset
        rules: List of validation rules
        thresholds: Optional thresholds for null_pct, dup_pct
        sample_n: Optional sample size for large datasets
    
    Returns:
        Validation result dict with violations and suggestions
    """
    df = resolve_dataset(dataset_ref)
    
    # Sample if needed
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    
    # Default thresholds
    thresholds = thresholds or {}
    max_null_pct = thresholds.get("max_null_pct", 0.0)
    max_dup_pct = thresholds.get("max_dup_pct", 0.0)
    
    violations = []
    rules_passed = 0
    rules_failed = 0
    
    for rule in rules:
        rule_type = rule.get("type")
        rule_violations = []
        
        if rule_type == "not_null":
            columns = rule.get("columns", [])
            rule_violations = _check_not_null(df, columns, max_null_pct)
            
        elif rule_type == "range":
            column = rule.get("column")
            min_val = rule.get("min")
            max_val = rule.get("max")
            rule_violations = _check_range(df, column, min_val, max_val)
            
        elif rule_type == "unique":
            column = rule.get("column")
            rule_violations = _check_unique(df, column, max_dup_pct)
            
        elif rule_type == "dtype":
            column = rule.get("column")
            expected = rule.get("expected")
            rule_violations = _check_dtype(df, column, expected)
            
        elif rule_type == "allowed_values":
            column = rule.get("column")
            values = rule.get("values", [])
            rule_violations = _check_allowed_values(df, column, values)
            
        elif rule_type == "regex":
            column = rule.get("column")
            pattern = rule.get("pattern", "")
            rule_violations = _check_regex(df, column, pattern)
        
        if rule_violations:
            violations.extend(rule_violations)
            rules_failed += 1
        else:
            rules_passed += 1
    
    # Sort violations by severity
    severity_order = {"high": 0, "medium": 1, "low": 2}
    violations.sort(key=lambda v: severity_order.get(v["severity"], 3))
    
    # Generate summary
    top_actions = []
    for v in violations[:3]:
        action = v["suggested_fix"]["action"]
        col = v["column"]
        if action == "clip":
            params = v["suggested_fix"]["params"]
            top_actions.append(f"clip {col} to [{params.get('min')}, {params.get('max')}]")
        elif action == "impute":
            top_actions.append(f"impute {col} (median)")
        elif action == "drop_column":
            top_actions.append(f"drop {col} (>50% null)")
        elif action == "dedupe":
            top_actions.append(f"dedupe on {col}")
        elif action == "cast":
            top_actions.append(f"cast {col} to {v['suggested_fix']['params'].get('dtype')}")
        else:
            top_actions.append(f"{action} on {col}")
    
    return {
        "passed": len(violations) == 0,
        "violations": violations,
        "schema_inferred": _infer_schema(df),
        "summary": {
            "rows_checked": len(df),
            "total_rules": len(rules),
            "passed": rules_passed,
            "failed": rules_failed,
            "top_action": ", then ".join(top_actions) if top_actions else "No issues found"
        }
    }


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class ValidationRule(BaseModel):
    """A single validation rule."""
    type: RuleType = Field(description="Rule type: not_null, range, unique, dtype, allowed_values, regex")
    column: Optional[str] = Field(default=None, description="Column to validate (for single-column rules)")
    columns: Optional[list[str]] = Field(default=None, description="Columns to validate (for not_null)")
    min: Optional[float] = Field(default=None, description="Minimum value (for range)")
    max: Optional[float] = Field(default=None, description="Maximum value (for range)")
    expected: Optional[str] = Field(default=None, description="Expected dtype (for dtype)")
    values: Optional[list[str]] = Field(default=None, description="Allowed values (for allowed_values)")
    pattern: Optional[str] = Field(default=None, description="Regex pattern (for regex)")


class ValidationThresholds(BaseModel):
    """Thresholds for validation."""
    max_null_pct: float = Field(default=0.0, description="Max allowed null percentage (0-1)")
    max_dup_pct: float = Field(default=0.0, description="Max allowed duplicate percentage (0-1)")


class DataValidationInput(BaseModel):
    """Input schema for data_validation tool."""
    dataset_ref: str = Field(description="Dataset reference from previous operation or table name")
    rules: list[ValidationRule] = Field(
        description="List of validation rules. Each rule has a type and type-specific parameters."
    )
    thresholds: Optional[ValidationThresholds] = Field(
        default=None,
        description="Thresholds for null and duplicate percentages"
    )
    sample_n: Optional[int] = Field(
        default=None,
        description="Sample size for large datasets. None = use full dataset."
    )


@tool(args_schema=DataValidationInput)
def data_validation_tool(
    dataset_ref: str,
    rules: list[dict],
    thresholds: Optional[dict] = None,
    sample_n: Optional[int] = None,
) -> str:
    """
    Validate a dataset for data quality issues.
    
    Run this FIRST on any new dataset to catch problems before analysis or training.
    
    RULE TYPES:
    - not_null: Check columns have no/few nulls
    - range: Check numeric values are within bounds
    - unique: Check column has unique values (for IDs)
    - dtype: Check column has expected type
    - allowed_values: Check values are in allowed set
    - regex: Check string values match pattern
    
    EXAMPLES:
    ```
    data_validation(
        dataset_ref="loan_data",
        rules=[
            {"type": "not_null", "columns": ["age", "income"]},
            {"type": "range", "column": "age", "min": 0, "max": 120},
            {"type": "unique", "column": "customer_id"},
            {"type": "allowed_values", "column": "status", "values": ["active", "closed"]}
        ],
        thresholds={"max_null_pct": 0.1}
    )
    ```
    
    OUTPUT:
    - passed: Whether all rules passed
    - violations: List of issues with severity and suggested fixes
    - schema_inferred: Detected column types
    - summary: Quick overview with top actions to take
    
    Use the suggested_fix in violations to decide which transformation to apply.
    """
    try:
        # Convert Pydantic models to dicts if needed
        rules_dicts = [r if isinstance(r, dict) else r.model_dump() for r in rules]
        thresholds_dict = thresholds if isinstance(thresholds, dict) else (thresholds.model_dump() if thresholds else None)
        
        result = validate_dataset(
            dataset_ref=dataset_ref,
            rules=rules_dicts,
            thresholds=thresholds_dict,
            sample_n=sample_n,
        )
        
        # Format output for agent
        lines = []
        
        if result["passed"]:
            lines.append("✓ VALIDATION PASSED")
        else:
            lines.append("✗ VALIDATION FAILED")
        
        lines.append("")
        lines.append(f"**Summary:** {result['summary']['passed']}/{result['summary']['total_rules']} rules passed")
        lines.append(f"**Rows checked:** {result['summary']['rows_checked']:,}")
        
        if result["violations"]:
            lines.append("")
            lines.append("### Violations (by severity)")
            for v in result["violations"]:
                severity_icon = "🔴" if v["severity"] == "high" else "🟡" if v["severity"] == "medium" else "🟢"
                lines.append(f"{severity_icon} **{v['rule']}** on `{v['column']}`: {v['count']:,} ({v['pct']:.1%})")
                if v.get("examples"):
                    examples_str = str(v["examples"][:3])
                    lines.append(f"   Examples: {examples_str}")
                fix = v["suggested_fix"]
                lines.append(f"   → Fix: `{fix['action']}` {fix.get('params', {})}")
            
            lines.append("")
            lines.append(f"**Top actions:** {result['summary']['top_action']}")
        
        lines.append("")
        lines.append("### Schema")
        for col_info in result["schema_inferred"][:10]:
            nullable = "nullable" if col_info["nullable"] else "not null"
            lines.append(f"- `{col_info['column']}`: {col_info['dtype']} ({nullable})")
        if len(result["schema_inferred"]) > 10:
            lines.append(f"- ... +{len(result['schema_inferred']) - 10} more columns")
        
        md = "\n".join(lines)
        payload = {
            "passed": result["passed"],
            "violations": result["violations"][:40],
            "schema_inferred": result["schema_inferred"][:40],
            "summary": result["summary"],
        }
        summary_line = (
            "passed" if result["passed"] else f"{len(result['violations'])} violation(s)"
        )
        return attach_analysis_sidecar(
            md,
            kind="data_validation",
            tool="data_validation_tool",
            summary=summary_line,
            payload=payload,
        )
        
    except Exception as e:
        return f"✗ Validation failed: {type(e).__name__}: {e}"


# Export
validation_tools = [data_validation_tool]

__all__ = [
    "data_validation_tool",
    "validate_dataset",
    "validation_tools",
]

