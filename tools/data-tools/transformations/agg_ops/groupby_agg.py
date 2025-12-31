"""Group by and aggregate tool."""

from typing import Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result

# Map user-friendly names to pandas aggregation functions
AGG_FUNCS = {
    "count": "count",
    "count_distinct": "nunique",
    "sum": "sum",
    "avg": "mean",
    "mean": "mean",
    "min": "min",
    "max": "max",
    "std": "std",
    "var": "var",
    "median": "median",
    "first": "first",
    "last": "last",
}


class AggSpec(BaseModel):
    """Single aggregation specification."""
    col: str = Field(description="Column to aggregate")
    fn: str = Field(
        description="Aggregation function: count, count_distinct, sum, avg, min, max, "
                    "std, var, median, first, last, quantile:N (e.g. quantile:0.75), "
                    "count_if:EXPR (e.g. count_if:status=='active')"
    )
    output: Optional[str] = Field(
        default=None,
        description="Output column name. Defaults to '{col}_{fn}'"
    )


class GroupByAggInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    keys: Union[str, list[str]] = Field(
        description="Column(s) to group by. Use list for multiple keys."
    )
    aggs: list[AggSpec] = Field(
        description="List of aggregations. Each has 'col', 'fn', and optional 'output'. "
                    "Example: [{'col': 'amount', 'fn': 'sum'}, {'col': 'id', 'fn': 'count'}]"
    )


def _parse_agg_fn(fn: str, df: pd.DataFrame):
    """Parse aggregation function string. Returns pandas-compatible function."""
    fn_lower = fn.lower().strip()
    
    if fn_lower in AGG_FUNCS:
        return AGG_FUNCS[fn_lower]
    
    if fn_lower.startswith("quantile:"):
        q = float(fn_lower.split(":")[1])
        return lambda x: x.quantile(q)
    
    if fn_lower.startswith("count_if:"):
        expr = fn_lower.split(":", 1)[1]
        return lambda x: df.loc[x.index].eval(expr).sum()
    
    raise ValueError(f"Unknown function: '{fn}'. Use: {', '.join(AGG_FUNCS.keys())}, quantile:N, or count_if:EXPR")


def _output_name(spec: AggSpec) -> str:
    """Generate output column name from spec."""
    if spec.output:
        return spec.output
    fn_name = spec.fn.lower().split(":")[0]
    return f"{spec.col}_{fn_name}"


@tool(args_schema=GroupByAggInput)
def groupby_agg_tool(
    dataset_ref: str,
    keys: Union[str, list[str]],
    aggs: list[dict]
) -> str:
    """
    Group rows by key column(s) and compute aggregations.
    
    Supported functions:
    - count, count_distinct, sum, avg, min, max, std, var, median, first, last
    - quantile:N (e.g. 'quantile:0.5' for median, 'quantile:0.95' for 95th percentile)
    - count_if:EXPR (e.g. 'count_if:status=="active"' to count matching rows)
    
    Example: Group sales by region and product, compute total and average:
      keys: ["region", "product"]
      aggs: [{"col": "amount", "fn": "sum", "output": "total"},
             {"col": "amount", "fn": "avg", "output": "avg_amount"}]
    """
    try:
        df = resolve_dataset(dataset_ref)
        key_cols = [keys] if isinstance(keys, str) else list(keys)
        
        # Validate keys
        missing_keys = [k for k in key_cols if k not in df.columns]
        if missing_keys:
            return f"✗ Key column(s) not found: {missing_keys}. Available: {list(df.columns)}"
        
        # Parse specs
        specs = [AggSpec(**a) if isinstance(a, dict) else a for a in aggs]
        if not specs:
            return "✗ No aggregations specified. Provide at least one in 'aggs'."
        
        # Validate agg columns
        missing_cols = [s.col for s in specs if s.col not in df.columns]
        if missing_cols:
            return f"✗ Aggregation column(s) not found: {missing_cols}. Available: {list(df.columns)}"
        
        # Build named aggregations using pandas NamedAgg
        named_aggs = {}
        for spec in specs:
            try:
                agg_fn = _parse_agg_fn(spec.fn, df)
                out_name = _output_name(spec)
                named_aggs[out_name] = pd.NamedAgg(column=spec.col, aggfunc=agg_fn)
            except ValueError as e:
                return f"✗ {e}"
        
        # Execute groupby with named aggregations
        result = df.groupby(key_cols, as_index=False).agg(**named_aggs)
        
        ref = save_result(result, "agg")
        detail = f"{len(df)} rows → {len(result)} groups, {len(specs)} aggs"
        return format_result(ref, result, "groupby_agg", detail)
        
    except Exception as e:
        return f"✗ groupby_agg failed: {e}"
