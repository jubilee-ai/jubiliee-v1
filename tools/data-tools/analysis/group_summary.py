"""
group_summary - Aggregate metrics by categorical columns.

The most fundamental analysis operation: "What's the average X by Y?"
Powers slice-and-dice analysis for business insights.
"""

import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset

# =============================================================================
# CORE FUNCTION
# =============================================================================

def compute_group_summary(
    dataset_ref: str,
    group_by: str | list[str],
    metrics: Optional[list[str]] = None,
    agg_funcs: list[str] = ["mean", "median", "count"],
    top_n: Optional[int] = None,
    sort_by: Optional[str] = None,
    sort_ascending: bool = False,
) -> dict:
    """
    Compute summary statistics grouped by categorical column(s).
    
    Args:
        dataset_ref: Reference to dataset
        group_by: Column(s) to group by
        metrics: Numeric columns to aggregate. If None, uses all numeric columns.
        agg_funcs: Aggregation functions to apply
        top_n: Limit to top N groups (by first metric's first agg)
        sort_by: Column to sort results by
        sort_ascending: Sort direction
    
    Returns:
        Dict with grouped statistics and insights
    """
    df = resolve_dataset(dataset_ref)
    
    # Normalize group_by to list
    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    
    # Validate group columns exist
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        return {"error": f"Group columns not found: {missing}"}
    
    # Select metric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if metrics:
        metric_cols = [c for c in metrics if c in numeric_cols]
        if not metric_cols:
            return {"error": f"No valid numeric columns in: {metrics}"}
    else:
        # Exclude group columns from metrics if they're numeric
        metric_cols = [c for c in numeric_cols if c not in group_cols]
    
    if not metric_cols:
        return {"error": "No numeric columns available for aggregation"}
    
    # Build aggregation dict
    agg_dict = {col: agg_funcs for col in metric_cols}
    
    # Perform groupby aggregation using pandas
    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)
    
    # Flatten column names
    grouped.columns = [f"{col}_{agg}" for col, agg in grouped.columns]
    grouped = grouped.reset_index()
    
    # Sort if specified
    if sort_by and sort_by in grouped.columns:
        grouped = grouped.sort_values(sort_by, ascending=sort_ascending)
    elif not sort_by and len(grouped.columns) > len(group_cols):
        # Default: sort by first metric descending
        first_metric_col = grouped.columns[len(group_cols)]
        grouped = grouped.sort_values(first_metric_col, ascending=False)
    
    # Limit to top N
    if top_n:
        grouped = grouped.head(top_n)
    
    # Convert to records
    records = grouped.to_dict(orient="records")
    
    # Round numeric values
    for record in records:
        for key, value in record.items():
            if isinstance(value, float):
                record[key] = round(value, 2)
    
    # Compute overall stats for comparison
    overall_stats = {}
    for col in metric_cols:
        overall_stats[col] = {
            "mean": round(df[col].mean(), 2),
            "median": round(df[col].median(), 2),
            "count": int(df[col].count()),
        }
    
    # Group metadata
    group_counts = df.groupby(group_cols, dropna=False).size()
    
    return {
        "group_by": group_cols,
        "metrics": metric_cols,
        "agg_funcs": agg_funcs,
        "n_groups": len(grouped),
        "total_rows": len(df),
        "groups": records,
        "overall": overall_stats,
        "group_sizes": {
            "min": int(group_counts.min()),
            "max": int(group_counts.max()),
            "mean": round(group_counts.mean(), 1),
        },
    }


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class GroupSummaryInput(BaseModel):
    """Input schema for group_summary_tool."""
    dataset_ref: str = Field(
        description="Dataset reference from previous operation or table name"
    )
    group_by: str | list[str] = Field(
        description="Column name or list of column names to group by"
    )
    metrics: Optional[list[str]] = Field(
        default=None,
        description="Numeric columns to aggregate. If not provided, uses all numeric columns."
    )
    agg_funcs: list[str] = Field(
        default=["mean", "median", "count"],
        description="Aggregation functions: 'mean', 'median', 'sum', 'count', 'min', 'max', 'std'"
    )
    top_n: Optional[int] = Field(
        default=None,
        description="Limit results to top N groups. Useful for high-cardinality group columns."
    )
    sort_by: Optional[str] = Field(
        default=None,
        description="Column to sort results by (use format 'metric_agg', e.g., 'income_mean')"
    )


@tool(args_schema=GroupSummaryInput)
def group_summary_tool(
    dataset_ref: str,
    group_by: str | list[str],
    metrics: Optional[list[str]] = None,
    agg_funcs: list[str] = ["mean", "median", "count"],
    top_n: Optional[int] = None,
    sort_by: Optional[str] = None,
) -> str:
    """
    Compute summary statistics grouped by categorical column(s).
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Answer "What's the average X by Y?" questions
    - Compare metrics across categories (regions, segments, groups)
    - Find which groups have highest/lowest values
    - Understand how a metric varies across segments
    
    EXAMPLES:
    ```
    # Average income by region
    group_summary(dataset_ref="customers", group_by="region", metrics=["income"])
    
    # Default rate by age group and gender
    group_summary(dataset_ref="loans", group_by=["age_group", "gender"], metrics=["default"])
    
    # Top 10 products by total sales
    group_summary(dataset_ref="sales", group_by="product", metrics=["revenue"], 
                  agg_funcs=["sum", "count"], top_n=10)
    
    # Multiple metrics with all common aggregations
    group_summary(dataset_ref="data", group_by="category", 
                  agg_funcs=["mean", "median", "min", "max", "std"])
    ```
    
    RETURNS:
    - Grouped statistics table
    - Comparison to overall averages
    - Group size distribution
    """
    try:
        result = compute_group_summary(
            dataset_ref=dataset_ref,
            group_by=group_by,
            metrics=metrics,
            agg_funcs=agg_funcs,
            top_n=top_n,
            sort_by=sort_by,
        )
        
        if "error" in result:
            return f"✗ {result['error']}"
        
        # Format output for agent
        lines = []
        
        lines.append("# Group Summary Analysis")
        lines.append("")
        group_str = ", ".join(f"`{c}`" for c in result["group_by"])
        lines.append(f"**Grouped by:** {group_str}")
        lines.append(f"**Metrics:** {', '.join(f'`{m}`' for m in result['metrics'])}")
        lines.append(f"**Aggregations:** {', '.join(result['agg_funcs'])}")
        lines.append(f"**Groups found:** {result['n_groups']} (from {result['total_rows']:,} rows)")
        lines.append("")
        
        # Overall stats for context
        lines.append("## Overall Averages (for comparison)")
        for metric, stats in result["overall"].items():
            lines.append(f"- `{metric}`: mean={stats['mean']:,.2f}, median={stats['median']:,.2f}")
        lines.append("")
        
        # Group results table
        lines.append("## Results by Group")
        
        if result["groups"]:
            # Build header
            first_row = result["groups"][0]
            headers = list(first_row.keys())
            
            # Truncate column names for display
            display_headers = []
            for h in headers:
                if len(h) > 12:
                    display_headers.append(h[:10] + "..")
                else:
                    display_headers.append(h)
            
            header_row = "| " + " | ".join(display_headers) + " |"
            separator = "|" + "|".join(["---"] * len(headers)) + "|"
            lines.append(header_row)
            lines.append(separator)
            
            # Data rows (limit to 20 for readability)
            display_rows = result["groups"][:20]
            for row in display_rows:
                values = []
                for h in headers:
                    val = row[h]
                    if isinstance(val, float):
                        values.append(f"{val:,.2f}")
                    elif isinstance(val, int):
                        values.append(f"{val:,}")
                    elif val is None or (isinstance(val, float) and pd.isna(val)):
                        values.append("(null)")
                    else:
                        val_str = str(val)
                        values.append(val_str[:15] if len(val_str) > 15 else val_str)
                lines.append("| " + " | ".join(values) + " |")
            
            if len(result["groups"]) > 20:
                lines.append(f"\n*Showing 20 of {len(result['groups'])} groups. Use `top_n` to limit.*")
        
        lines.append("")
        
        # Group size info
        sizes = result["group_sizes"]
        lines.append("## Group Sizes")
        lines.append(f"- Min: {sizes['min']:,} rows")
        lines.append(f"- Max: {sizes['max']:,} rows")
        lines.append(f"- Avg: {sizes['mean']:,.1f} rows")
        
        # Add insight hints
        if result["groups"] and len(result["groups"]) >= 2:
            lines.append("")
            lines.append("## Quick Insights")
            
            # Find column with biggest variance across groups
            for metric in result["metrics"][:2]:
                mean_col = f"{metric}_mean"
                if mean_col in result["groups"][0]:
                    values = [g[mean_col] for g in result["groups"] if g.get(mean_col) is not None]
                    if values and len(values) >= 2:
                        max_val = max(values)
                        min_val = min(values)
                        if min_val > 0:
                            ratio = max_val / min_val
                            if ratio > 1.5:
                                max_group = next(g for g in result["groups"] if g.get(mean_col) == max_val)
                                min_group = next(g for g in result["groups"] if g.get(mean_col) == min_val)
                                group_key = result["group_by"][0]
                                lines.append(
                                    f"- `{metric}` varies {ratio:.1f}x across groups "
                                    f"(highest: {max_group[group_key]}, lowest: {min_group[group_key]})"
                                )
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"✗ Group summary failed: {type(e).__name__}: {e}"


# =============================================================================
# EXPORTS
# =============================================================================

group_summary_tools = [group_summary_tool]

__all__ = [
    "group_summary_tool",
    "compute_group_summary",
    "group_summary_tools",
]

