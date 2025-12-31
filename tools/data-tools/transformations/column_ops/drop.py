"""Drop columns tool - remove specified columns."""

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class DropColumnsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    columns: list[str] = Field(description="Columns to drop. Example: ['temp_col', 'debug_info']")


@tool(args_schema=DropColumnsInput)
def drop_columns_tool(dataset_ref: str, columns: list[str]) -> str:
    """Drop the specified columns, keeping all others."""
    try:
        df = resolve_dataset(dataset_ref)
        
        existing = [c for c in columns if c in df.columns]
        missing = [c for c in columns if c not in df.columns]
        
        result = df.drop(columns=existing, errors='ignore').copy()
        ref = save_result(result, "drp")
        
        warnings = [f"Not found: {missing}"] if missing else None
        return format_result(ref, result, "drop", f"removed {len(existing)}", warnings)
        
    except Exception as e:
        return f"✗ drop failed: {e}"

