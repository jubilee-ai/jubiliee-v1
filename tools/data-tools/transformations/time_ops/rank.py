"""Rank tool - assign rank based on values."""

from typing import Optional, Union, Literal

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class RankInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Column to rank by")
    partition_by: Optional[Union[str, list[str]]] = Field(
        default=None, description="Column(s) to partition by (rank within each group)"
    )
    ascending: bool = Field(default=True, description="True=lowest value gets rank 1")
    method: Literal["average", "min", "max", "first", "dense"] = Field(
        default="average",
        description="How to handle ties: average, min, max, first, dense"
    )
    output_column: Optional[str] = Field(
        default=None, description="Output column name. Defaults to '{column}_rank'"
    )


@tool(args_schema=RankInput)
def rank_tool(
    dataset_ref: str,
    column: str,
    partition_by: Optional[Union[str, list[str]]] = None,
    ascending: bool = True,
    method: str = "average",
    output_column: Optional[str] = None
) -> str:
    """
    Assign rank to rows based on column values.
    Methods: average (default), min, max, first (by position), dense (no gaps).
    Unlike row_number, rank handles ties according to the method.
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        result = df.copy()
        out_col = output_column or f"{column}_rank"
        
        if partition_by:
            part_cols = [partition_by] if isinstance(partition_by, str) else list(partition_by)
            result[out_col] = result.groupby(part_cols)[column].rank(
                method=method, ascending=ascending
            )
        else:
            result[out_col] = result[column].rank(method=method, ascending=ascending)
        
        ref = save_result(result, "rnk")
        detail = f"{column} → {out_col} (method={method})"
        if partition_by:
            detail += f", by={partition_by}"
        if not ascending:
            detail += ", desc"
        return format_result(ref, result, "rank", detail, None)
        
    except Exception as e:
        return f"✗ rank failed: {e}"


