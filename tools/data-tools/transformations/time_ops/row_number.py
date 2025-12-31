"""Row number tool - assign sequential numbers within partitions."""

from typing import Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class RowNumberInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    partition_by: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to partition by. Numbers restart for each group."
    )
    order_by: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to order by within partitions"
    )
    ascending: Union[bool, list[bool]] = Field(
        default=True,
        description="Sort direction for order_by columns"
    )
    output_column: str = Field(
        default="row_num",
        description="Name for row number column"
    )


@tool(args_schema=RowNumberInput)
def row_number_tool(
    dataset_ref: str,
    partition_by: Optional[Union[str, list[str]]] = None,
    order_by: Optional[Union[str, list[str]]] = None,
    ascending: Union[bool, list[bool]] = True,
    output_column: str = "row_num"
) -> str:
    """
    Add row numbers, optionally within partitions and ordered.
    Useful for ranking, selecting first/last N per group.
    """
    try:
        df = resolve_dataset(dataset_ref)
        result = df.copy()
        
        # Sort if order_by specified
        if order_by:
            order_cols = [order_by] if isinstance(order_by, str) else list(order_by)
            result = result.sort_values(order_cols, ascending=ascending)
        
        # Compute row numbers
        if partition_by:
            part_cols = [partition_by] if isinstance(partition_by, str) else list(partition_by)
            result[output_column] = result.groupby(part_cols).cumcount() + 1
        else:
            result[output_column] = range(1, len(result) + 1)
        
        result = result.reset_index(drop=True)
        
        ref = save_result(result, "rn")
        detail = f"→ {output_column}"
        if partition_by:
            detail += f", by={partition_by}"
        if order_by:
            detail += f", order={order_by}"
        return format_result(ref, result, "row_number", detail, None)
        
    except Exception as e:
        return f"✗ row_number failed: {e}"

