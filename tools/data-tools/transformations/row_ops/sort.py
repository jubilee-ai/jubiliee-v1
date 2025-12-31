"""Sort rows tool - order rows by column(s)."""

from typing import Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class SortRowsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    by: Union[str, list[str]] = Field(description="Column(s) to sort by. Example: 'date' or ['category', 'amount']")
    ascending: Union[bool, list[bool]] = Field(default=True, description="Sort direction. Example: False or [True, False]")


@tool(args_schema=SortRowsInput)
def sort_rows_tool(dataset_ref: str, by: Union[str, list[str]], ascending: Union[bool, list[bool]] = True) -> str:
    """Sort rows by one or more columns."""
    try:
        df = resolve_dataset(dataset_ref)
        
        by_list = [by] if isinstance(by, str) else list(by)
        missing = [c for c in by_list if c not in df.columns]
        if missing:
            return f"✗ Columns not found: {missing}. Available: {list(df.columns)}"
        
        result = df.sort_values(by=by_list, ascending=ascending).reset_index(drop=True)
        
        ref = save_result(result, "srt")
        direction = "asc" if ascending is True else "desc" if ascending is False else "mixed"
        return format_result(ref, result, "sort", f"by {by_list} {direction}", None)
        
    except Exception as e:
        return f"✗ sort failed: {e}"

