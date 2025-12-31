"""Select columns tool - keep only specified columns."""

from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class SelectColumnsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    columns: list[str] = Field(description="Columns to keep. Example: ['id', 'name', 'amount']")


@tool(args_schema=SelectColumnsInput)
def select_columns_tool(dataset_ref: str, columns: list[str]) -> str:
    """Keep only the specified columns, dropping all others."""
    try:
        df = resolve_dataset(dataset_ref)
        
        # Validate columns exist
        missing = [c for c in columns if c not in df.columns]
        valid = [c for c in columns if c in df.columns]
        
        if not valid:
            return f"✗ No valid columns. Requested: {columns}, Available: {list(df.columns)}"
        
        result = df[valid].copy()
        ref = save_result(result, "sel")
        
        warnings = [f"Missing columns ignored: {missing}"] if missing else None
        return format_result(ref, result, "select", f"kept {len(valid)} of {len(columns)}", warnings)
        
    except Exception as e:
        return f"✗ select failed: {e}"

