"""Dedupe tool - remove duplicate rows."""

from typing import Literal, Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class DedupeInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    by: Optional[Union[str, list[str]]] = Field(
        default=None, 
        description="Column(s) to check for duplicates. None = all columns. Example: 'user_id' or ['email', 'date']"
    )
    keep: Literal["first", "last"] = Field(default="first", description="Which duplicate to keep: 'first' or 'last'")


@tool(args_schema=DedupeInput)
def dedupe_tool(dataset_ref: str, by: Optional[Union[str, list[str]]] = None, keep: str = "first") -> str:
    """Remove duplicate rows, keeping first or last occurrence."""
    try:
        df = resolve_dataset(dataset_ref)
        
        subset = [by] if isinstance(by, str) else by
        if subset:
            missing = [c for c in subset if c not in df.columns]
            if missing:
                return f"✗ Columns not found: {missing}. Available: {list(df.columns)}"
        
        result = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        
        removed = len(df) - len(result)
        ref = save_result(result, "dup")
        
        warnings = None
        if removed == 0:
            warnings = ["No duplicates found"]
        return format_result(ref, result, "dedupe", f"removed {removed} duplicates", warnings)
        
    except Exception as e:
        return f"✗ dedupe failed: {e}"

