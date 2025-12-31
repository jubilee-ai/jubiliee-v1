"""Limit rows tool - return first n rows."""

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class LimitRowsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    n: int = Field(description="Maximum number of rows to return")


@tool(args_schema=LimitRowsInput)
def limit_rows_tool(dataset_ref: str, n: int) -> str:
    """Return only the first n rows."""
    try:
        df = resolve_dataset(dataset_ref)
        
        result = df.head(n).reset_index(drop=True)
        
        ref = save_result(result, "lim")
        warnings = [f"Requested {n} but only {len(df)} available"] if n > len(df) else None
        return format_result(ref, result, "limit", f"first {len(result)} of {len(df)}", warnings)
        
    except Exception as e:
        return f"✗ limit failed: {e}"

