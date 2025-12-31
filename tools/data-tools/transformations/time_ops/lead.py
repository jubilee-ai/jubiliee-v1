"""Lead tool - get next row values within partitions."""

from typing import Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class LeadInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Column to get next values from")
    periods: int = Field(default=1, description="Number of periods to lead (1 = next row)")
    partition_by: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to partition by. Example: 'user_id' or ['user_id', 'category']"
    )
    order_by: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to order by before computing lead. Example: 'timestamp'"
    )
    output_column: Optional[str] = Field(
        default=None,
        description="Name for lead column. Defaults to '{column}_lead_{periods}'"
    )


@tool(args_schema=LeadInput)
def lead_tool(
    dataset_ref: str,
    column: str,
    periods: int = 1,
    partition_by: Optional[Union[str, list[str]]] = None,
    order_by: Optional[Union[str, list[str]]] = None,
    output_column: Optional[str] = None
) -> str:
    """
    Get next row value(s) for a column, optionally within partitions.
    Useful for computing time-to-next-event, next transaction amount, etc.
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        result = df.copy()
        out_col = output_column or f"{column}_lead_{periods}"
        
        # Sort if order_by specified
        if order_by:
            order_cols = [order_by] if isinstance(order_by, str) else list(order_by)
            result = result.sort_values(order_cols)
        
        # Compute lead (negative shift)
        if partition_by:
            part_cols = [partition_by] if isinstance(partition_by, str) else list(partition_by)
            result[out_col] = result.groupby(part_cols)[column].shift(-periods)
        else:
            result[out_col] = result[column].shift(-periods)
        
        # Reset index after sort
        result = result.reset_index(drop=True)
        
        ref = save_result(result, "led")
        detail = f"{column} → {out_col} (lead={periods})"
        if partition_by:
            detail += f", by={partition_by}"
        return format_result(ref, result, "lead", detail, None)
        
    except Exception as e:
        return f"✗ lead failed: {e}"

