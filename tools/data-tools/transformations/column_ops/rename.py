"""Rename columns tool - rename columns using a mapping."""

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class RenameColumnsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    mapping: dict[str, str] = Field(description="Old to new name mapping. Example: {'old_name': 'new_name'}")


@tool(args_schema=RenameColumnsInput)
def rename_columns_tool(dataset_ref: str, mapping: dict[str, str]) -> str:
    """Rename columns according to the mapping."""
    try:
        df = resolve_dataset(dataset_ref)
        
        valid = {k: v for k, v in mapping.items() if k in df.columns}
        missing = [k for k in mapping if k not in df.columns]
        
        if not valid:
            return f"✗ No columns to rename. Requested: {list(mapping.keys())}, Available: {list(df.columns)}"
        
        result = df.rename(columns=valid).copy()
        ref = save_result(result, "ren")
        
        warnings = [f"Not found: {missing}"] if missing else None
        return format_result(ref, result, "rename", f"renamed {len(valid)}", warnings)
        
    except Exception as e:
        return f"✗ rename failed: {e}"

