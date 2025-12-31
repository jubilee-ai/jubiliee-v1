"""Parse datetime tool - convert string column to datetime."""

from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class ParseDatetimeInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Column to parse as datetime")
    format: Optional[str] = Field(default=None, description="Date format (e.g., '%Y-%m-%d'). Auto-detected if not provided.")


@tool(args_schema=ParseDatetimeInput)
def parse_datetime_tool(dataset_ref: str, column: str, format: Optional[str] = None) -> str:
    """Parse a string column to datetime type."""
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        result = df.copy()
        result[column] = pd.to_datetime(result[column], format=format or "mixed", errors='coerce')
        
        nulls = result[column].isna().sum() - df[column].isna().sum()
        ref = save_result(result, "pdt")
        
        warnings = [f"{nulls} values failed to parse"] if nulls > 0 else None
        return format_result(ref, result, "parse_datetime", column, warnings)
        
    except Exception as e:
        return f"✗ parse_datetime failed: {e}"

