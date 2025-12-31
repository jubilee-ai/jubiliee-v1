"""Rolling sum tool - compute moving sum."""

from typing import Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class RollingSumInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Numeric column to compute rolling sum on")
    window: int = Field(description="Window size (number of rows)")
    partition_by: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to partition by. Computes rolling sum within each group."
    )
    order_by: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to order by before computing rolling sum"
    )
    min_periods: Optional[int] = Field(
        default=None,
        description="Minimum observations required. Defaults to window size (strict). Use 1 for partial windows."
    )
    output_column: Optional[str] = Field(
        default=None,
        description="Name for output column. Defaults to '{column}_rsum_{window}'"
    )


@tool(args_schema=RollingSumInput)
def rolling_sum_tool(
    dataset_ref: str,
    column: str,
    window: int,
    partition_by: Optional[Union[str, list[str]]] = None,
    order_by: Optional[Union[str, list[str]]] = None,
    min_periods: Optional[int] = None,
    output_column: Optional[str] = None
) -> str:
    """
    Compute rolling sum over a window of rows.
    Useful for cumulative metrics, trailing totals.
    Default: strict windows (NaN until window is full). Set min_periods=1 for partial windows.
    """
    try:
        df = resolve_dataset(dataset_ref)
        warnings = []
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        # Validate numeric column
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"✗ Column '{column}' is not numeric (dtype: {df[column].dtype}). Rolling sum requires numeric data."
        
        result = df.copy()
        out_col = output_column or f"{column}_rsum_{window}"
        
        # Default to strict windows (min_periods = window)
        effective_min_periods = min_periods if min_periods is not None else window
        
        # Sort if order_by specified
        if order_by:
            order_cols = [order_by] if isinstance(order_by, str) else list(order_by)
            result = result.sort_values(order_cols)
        
        # Compute rolling sum
        if partition_by:
            part_cols = [partition_by] if isinstance(partition_by, str) else list(partition_by)
            result[out_col] = result.groupby(part_cols)[column].transform(
                lambda x: x.rolling(window=window, min_periods=effective_min_periods).sum()
            )
        else:
            result[out_col] = result[column].rolling(window=window, min_periods=effective_min_periods).sum()
        
        result = result.reset_index(drop=True)
        
        # Warn about NaN values from incomplete windows
        nan_count = result[out_col].isna().sum()
        if nan_count > 0:
            warnings.append(f"{nan_count} NaN values (incomplete windows)")
        
        ref = save_result(result, "rsm")
        detail = f"{column} → {out_col} (window={window})"
        if partition_by:
            detail += f", by={partition_by}"
        return format_result(ref, result, "rolling_sum", detail, warnings if warnings else None)
        
    except Exception as e:
        return f"✗ rolling_sum failed: {e}"

