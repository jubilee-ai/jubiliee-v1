"""Pivot tool - reshape long to wide format."""

from typing import Optional, Union, Literal

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


AGG_FUNCTIONS = {
    "sum": "sum", "mean": "mean", "count": "count", "min": "min", "max": "max",
    "first": "first", "last": "last", "median": "median", "std": "std", "var": "var"
}


class PivotInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    index: Union[str, list[str]] = Field(
        description="Column(s) to use as row identifiers"
    )
    columns: str = Field(
        description="Column whose unique values become new columns"
    )
    values: Union[str, list[str]] = Field(
        description="Column(s) to aggregate into the pivot cells"
    )
    agg_func: str = Field(
        default="mean",
        description="Aggregation: sum, mean, count, min, max, first, last, median, std, var"
    )
    fill_value: Optional[float] = Field(
        default=None, description="Value to fill missing cells (default: NaN)"
    )


@tool(args_schema=PivotInput)
def pivot_tool(
    dataset_ref: str,
    index: Union[str, list[str]],
    columns: str,
    values: Union[str, list[str]],
    agg_func: str = "mean",
    fill_value: Optional[float] = None
) -> str:
    """
    Reshape data from long to wide format using pivot_table.
    Groups by index, spreads column values horizontally, aggregates values.
    Example: sales by (region, product) → regions as rows, products as columns.
    """
    try:
        df = resolve_dataset(dataset_ref)
        warnings = []
        
        # Validate columns
        index_cols = [index] if isinstance(index, str) else list(index)
        value_cols = [values] if isinstance(values, str) else list(values)
        all_cols = index_cols + [columns] + value_cols
        
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            return f"✗ Columns not found: {missing}. Available: {list(df.columns)}"
        
        if agg_func not in AGG_FUNCTIONS:
            return f"✗ Unknown agg_func '{agg_func}'. Use: {list(AGG_FUNCTIONS.keys())}"
        
        # Pivot
        result = pd.pivot_table(
            df,
            index=index_cols,
            columns=columns,
            values=value_cols,
            aggfunc=AGG_FUNCTIONS[agg_func],
            fill_value=fill_value
        )
        
        # Flatten multi-level column names
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = ['_'.join(str(c) for c in col).strip('_') for col in result.columns]
        else:
            result.columns = [str(c) for c in result.columns]
        
        result = result.reset_index()
        
        ref = save_result(result, "pvt")
        n_pivoted = len(df[columns].unique())
        detail = f"{index} × {columns} → {n_pivoted} new cols (agg={agg_func})"
        return format_result(ref, result, "pivot", detail, warnings if warnings else None)
        
    except Exception as e:
        return f"✗ pivot failed: {e}"

