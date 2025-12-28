"""
join_merge - Join two datasets safely with join diagnostics.

Provides comprehensive diagnostics for AI agents:
- Match rate: percentage of rows that matched
- Unmatched counts from each side  
- Row explosion detection (many-to-many joins)
- Column collision handling and reporting
"""

import json
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .utils import (build_schema_from_dataframe, fuzzy_suggest,
                    generate_unique_id, get_registered_dataset,
                    get_similar_matches, list_all_dataset_refs,
                    register_dataset, utc_timestamp)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MAX_ROWS = 10_000
EXPLOSION_THRESHOLD = 2.0  # Warn if result rows > threshold * max(left, right)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class JoinReport:
    """Diagnostic report for a join operation."""
    left_rows: int
    right_rows: int
    result_rows: int
    match_rate_left: float
    match_rate_right: float
    unmatched_left: int
    unmatched_right: int
    row_explosion: bool
    explosion_factor: float
    collisions: list[str]
    key_columns: list[str]
    join_type: str
    
    def to_dict(self) -> dict:
        return {
            "left_rows": self.left_rows,
            "right_rows": self.right_rows,
            "result_rows": self.result_rows,
            "match_rate": {
                "left": round(self.match_rate_left, 4),
                "right": round(self.match_rate_right, 4),
            },
            "unmatched": {
                "left": self.unmatched_left,
                "right": self.unmatched_right,
            },
            "row_explosion": self.row_explosion,
            "explosion_factor": round(self.explosion_factor, 2),
            "collisions": self.collisions,
            "key_columns": self.key_columns,
            "join_type": self.join_type,
        }


@dataclass  
class JoinResult:
    """Result from join_merge operation."""
    dataset_ref: str
    schema: list[dict]
    join_report: JoinReport
    data: list[dict]
    stats: dict
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "dataset_ref": self.dataset_ref,
            "schema": self.schema,
            "join_report": self.join_report.to_dict(),
            "data": self.data,
            "stats": self.stats,
            "warnings": self.warnings,
        }


# =============================================================================
# HELPERS
# =============================================================================

def _parse_keys(keys: Union[str, list[str], dict[str, str]]) -> tuple[list[str], list[str]]:
    """Parse join keys into (left_keys, right_keys) lists."""
    if isinstance(keys, str):
        return [keys], [keys]
    if isinstance(keys, list):
        return keys, keys
    if isinstance(keys, dict):
        return list(keys.keys()), list(keys.values())
    raise ValueError(f"Invalid keys type: {type(keys)}. Expected str, list, or dict.")


def _validate_keys(df: pd.DataFrame, keys: list[str], side: str) -> None:
    """Validate keys exist in DataFrame, raise helpful error if not."""
    missing = [k for k in keys if k not in df.columns]
    if missing:
        suggestions = [
            f"'{k}' (did you mean: {', '.join(get_similar_matches(k, df.columns))}?)"
            if get_similar_matches(k, df.columns) else f"'{k}'"
            for k in missing
        ]
        raise ValueError(
            f"{side} key(s) not found: {', '.join(suggestions)}\n"
            f"Available columns: {list(df.columns)}"
        )


def _resolve_dataset(ref: str) -> pd.DataFrame:
    """Resolve a dataset reference to a DataFrame."""
    # Check registry first (for chained operations)
    if (df := get_registered_dataset(ref)) is not None:
        return df.copy()
    
    # Try SQL warehouse
    try:
        from .sql_query import get_warehouse
        if (table := get_warehouse().get_table_info(ref)):
            df, _ = get_warehouse().execute_query(f"SELECT * FROM {table.name}")
            return df
    except Exception:
        pass
    
    # Try catalog
    try:
        from .data_loader import dataset_get
        return pd.DataFrame(dataset_get(asset_id=ref, limit=-1).data)
    except Exception:
        pass
    
    # Not found
    available = list_all_dataset_refs()
    hint = fuzzy_suggest(ref, available) if available else "No datasets loaded. Use sql_query_tool or dataset_get_tool first."
    raise ValueError(f"Dataset '{ref}' not found. {hint}")


def _detect_collisions(left_df: pd.DataFrame, right_df: pd.DataFrame, 
                       left_keys: list[str], right_keys: list[str]) -> list[str]:
    """Find non-key columns that exist in both DataFrames."""
    left_cols = set(left_df.columns) - set(left_keys)
    right_cols = set(right_df.columns) - set(right_keys)
    return sorted(left_cols & right_cols)


# =============================================================================
# MAIN JOIN FUNCTION
# =============================================================================

def join_merge(
    left_ref: str,
    right_ref: str,
    keys: Union[str, list[str], dict[str, str]],
    join_type: Literal["inner", "left", "right", "outer"] = "inner",
    suffixes: tuple[str, str] = ("_left", "_right"),
    dedupe_strategy: Optional[Literal["first", "last"]] = None,
    validate: Optional[Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]] = None,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> JoinResult:
    """
    Join two datasets safely with comprehensive diagnostics.
    
    Args:
        left_ref: Reference to left dataset (table name, asset_id, or registered ref)
        right_ref: Reference to right dataset
        keys: Join key(s) - str, list[str], or dict mapping left to right cols
        join_type: Type of join - 'inner', 'left', 'right', 'outer'
        suffixes: Suffixes for overlapping column names
        dedupe_strategy: 'first' or 'last' to dedupe before join
        validate: Cardinality validation ('one_to_one', 'one_to_many', etc.)
        max_rows: Maximum rows in result
    
    Returns:
        JoinResult with joined data and diagnostics
    """
    # Resolve and parse
    left_df = _resolve_dataset(left_ref)
    right_df = _resolve_dataset(right_ref)
    left_keys, right_keys = _parse_keys(keys)
    
    # Validate keys exist
    _validate_keys(left_df, left_keys, "Left")
    _validate_keys(right_df, right_keys, "Right")
    
    warnings = []
    left_original, right_original = len(left_df), len(right_df)
    
    # Deduplicate if requested
    if dedupe_strategy:
        left_df = left_df.drop_duplicates(subset=left_keys, keep=dedupe_strategy)
        right_df = right_df.drop_duplicates(subset=right_keys, keep=dedupe_strategy)
        if len(left_df) < left_original:
            warnings.append(f"Deduplicated left: {left_original:,} → {len(left_df):,} rows")
        if len(right_df) < right_original:
            warnings.append(f"Deduplicated right: {right_original:,} → {len(right_df):,} rows")
    
    left_rows, right_rows = len(left_df), len(right_df)
    collisions = _detect_collisions(left_df, right_df, left_keys, right_keys)
    
    # Perform merge with indicator for diagnostics
    try:
        merge_kwargs = dict(
            how=join_type,
            suffixes=suffixes,
            validate=validate,
            indicator=True,  # Pandas built-in: adds _merge column
        )
        if left_keys == right_keys:
            result_df = pd.merge(left_df, right_df, on=left_keys, **merge_kwargs)
        else:
            result_df = pd.merge(left_df, right_df, left_on=left_keys, right_on=right_keys, **merge_kwargs)
    except pd.errors.MergeError as e:
        raise ValueError(f"Join validation failed: {e}")
    
    result_rows = len(result_df)
    
    # Use pandas indicator for match diagnostics (built-in!)
    merge_indicator = result_df["_merge"]
    matched_both = (merge_indicator == "both").sum()
    left_only = (merge_indicator == "left_only").sum()
    right_only = (merge_indicator == "right_only").sum()
    
    # Calculate match rates based on join type
    if join_type == "inner":
        match_rate_left = 1.0 if left_rows == 0 else matched_both / left_rows
        match_rate_right = 1.0 if right_rows == 0 else matched_both / right_rows
    elif join_type == "left":
        match_rate_left = 1.0
        match_rate_right = 1.0 if right_rows == 0 else matched_both / right_rows
    elif join_type == "right":
        match_rate_left = 1.0 if left_rows == 0 else matched_both / left_rows
        match_rate_right = 1.0
    else:  # outer
        match_rate_left = 1.0 if left_rows == 0 else matched_both / left_rows
        match_rate_right = 1.0 if right_rows == 0 else matched_both / right_rows
    
    # Unmatched counts (using pre-join key comparison for accuracy)
    left_key_idx = pd.MultiIndex.from_frame(left_df[left_keys]) if len(left_keys) > 1 else left_df[left_keys[0]]
    right_key_idx = pd.MultiIndex.from_frame(right_df[right_keys]) if len(right_keys) > 1 else right_df[right_keys[0]]
    unmatched_left = int((~left_key_idx.isin(right_key_idx)).sum())
    unmatched_right = int((~right_key_idx.isin(left_key_idx)).sum())
    
    # Drop the indicator column
    result_df = result_df.drop(columns=["_merge"])
    
    # Row explosion detection
    max_input = max(left_rows, right_rows)
    explosion_factor = result_rows / max_input if max_input > 0 else 1.0
    row_explosion = explosion_factor >= EXPLOSION_THRESHOLD
    
    if row_explosion:
        warnings.append(
            f"Row explosion detected! {result_rows:,} rows from {left_rows:,}+{right_rows:,} input "
            f"(factor: {explosion_factor:.1f}x). Consider dedupe_strategy='first'."
        )
    
    if collisions:
        renamed = [f"{c}{suffixes[0]}, {c}{suffixes[1]}" for c in collisions]
        warnings.append(f"Column collisions renamed: {', '.join(renamed)}")
    
    # Apply row limit
    total_rows = result_rows
    if result_rows > max_rows:
        result_df = result_df.head(max_rows)
        warnings.append(f"Result truncated from {result_rows:,} to {max_rows:,} rows.")
    
    # Generate reference and register for chaining
    result_ref = generate_unique_id("join", left_ref, right_ref, str(keys), join_type)
    register_dataset(result_ref, result_df)
    
    return JoinResult(
        dataset_ref=result_ref,
        schema=build_schema_from_dataframe(result_df),
        join_report=JoinReport(
            left_rows=left_rows,
            right_rows=right_rows,
            result_rows=len(result_df),
            match_rate_left=match_rate_left,
            match_rate_right=match_rate_right,
            unmatched_left=unmatched_left,
            unmatched_right=unmatched_right,
            row_explosion=row_explosion,
            explosion_factor=explosion_factor,
            collisions=collisions,
            key_columns=left_keys if left_keys == right_keys else list(zip(left_keys, right_keys)),
            join_type=join_type,
        ),
        data=result_df.where(pd.notna(result_df), None).to_dict(orient="records"),
        stats={
            "left_source": left_ref,
            "right_source": right_ref,
            "execution_time": utc_timestamp(),
            "total_result_rows": total_rows,
            "rows_returned": len(result_df),
            "columns_returned": len(result_df.columns),
        },
        warnings=warnings,
    )


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class JoinMergeInput(BaseModel):
    """Input schema for join_merge tool."""
    left_ref: str = Field(description="Reference to left dataset (table name, asset_id, or previous join ref)")
    right_ref: str = Field(description="Reference to right dataset")
    keys: Union[str, list[str], dict[str, str]] = Field(
        description="Join key(s): single column, list of columns, or dict mapping left→right columns"
    )
    join_type: Literal["inner", "left", "right", "outer"] = Field(
        default="inner", description="Join type: inner, left, right, or outer"
    )
    suffixes: Optional[tuple[str, str]] = Field(
        default=("_left", "_right"), description="Suffixes for overlapping column names"
    )
    dedupe_strategy: Optional[Literal["first", "last"]] = Field(
        default=None, description="Dedupe by key before join: 'first' or 'last'"
    )
    cardinality: Optional[Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]] = Field(
        default=None, description="Validate join cardinality"
    )


@tool(args_schema=JoinMergeInput)
def join_merge_tool(
    left_ref: str,
    right_ref: str,
    keys: Union[str, list[str], dict[str, str]],
    join_type: Literal["inner", "left", "right", "outer"] = "inner",
    suffixes: Optional[tuple[str, str]] = ("_left", "_right"),
    dedupe_strategy: Optional[Literal["first", "last"]] = None,
    cardinality: Optional[Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]] = None,
) -> str:
    """
    Join two datasets with diagnostics (match rate, row explosion, collisions).
    
    First load data with sql_query_tool or dataset_get_tool, then join.
    The returned dataset_ref can be used in subsequent operations.
    
    Examples:
    - join_merge_tool('customers', 'orders', 'customer_id')
    - join_merge_tool('a', 'b', {'cust_id': 'customer_id'}, 'left')
    - join_merge_tool('a', 'b', 'id', dedupe_strategy='first')
    """
    try:
        result = join_merge(
            left_ref=left_ref,
            right_ref=right_ref,
            keys=keys,
            join_type=join_type,
            suffixes=tuple(suffixes) if isinstance(suffixes, list) else suffixes or ("_left", "_right"),
            dedupe_strategy=dedupe_strategy,
            validate=cardinality,
        )
        return _format_result(result, left_ref, right_ref, join_type)
    except ValueError as e:
        return f"**Join Error:** {e}\n\n**Available datasets:** {', '.join(list_all_dataset_refs()) or 'None loaded'}"
    except Exception as e:
        return f"**Error:** {type(e).__name__}: {e}"


def _format_result(result: JoinResult, left_ref: str, right_ref: str, join_type: str) -> str:
    """Format join result for AI agent consumption."""
    r = result.to_dict()
    report = r["join_report"]
    
    lines = [
        f"## Join Result: {left_ref} ⟕ {right_ref}",
        f"**Reference:** `{r['dataset_ref']}`",
        f"**Type:** {join_type.upper()} | **Keys:** {report['key_columns']}",
        "",
    ]
    
    if r["warnings"]:
        lines.extend(["### ⚠️ Warnings", *[f"- {w}" for w in r["warnings"]], ""])
    
    lines.extend([
        "### Diagnostics",
        f"| Metric | Left | Right |",
        f"|--------|------|-------|",
        f"| Rows | {report['left_rows']:,} | {report['right_rows']:,} |",
        f"| Match Rate | {report['match_rate']['left']:.1%} | {report['match_rate']['right']:.1%} |",
        f"| Unmatched | {report['unmatched']['left']:,} | {report['unmatched']['right']:,} |",
        f"",
        f"**Result:** {report['result_rows']:,} rows, {len(r['schema'])} columns",
    ])
    
    if report["row_explosion"]:
        lines.append(f"**⚠️ Row explosion:** {report['explosion_factor']}x")
    if report["collisions"]:
        lines.append(f"**Collisions:** {', '.join(report['collisions'])}")
    lines.append("")
    
    # Schema preview
    lines.append("### Schema")
    for col in r["schema"][:10]:
        lines.append(f"- `{col['name']}` ({col['type']})")
    if len(r["schema"]) > 10:
        lines.append(f"- *...+{len(r['schema']) - 10} more*")
    lines.append("")
    
    # Data preview
    if r["data"]:
        lines.extend([
            "### Preview",
            "```json",
            json.dumps(r["data"][:5], indent=2, default=str),
            "```",
        ])
        if len(r["data"]) > 5:
            lines.append(f"*...{len(r['data']) - 5} more rows*")
        lines.append("")
    
    # Full data
    lines.extend([
        "<data>",
        json.dumps(r["data"], default=str),
        "</data>",
    ])
    
    return "\n".join(lines)


# Exports
join_merge_tools = [join_merge_tool]

__all__ = [
    "join_merge",
    "join_merge_tool", 
    "join_merge_tools",
    "JoinResult",
    "JoinReport",
]
