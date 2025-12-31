"""Union tool - combine multiple datasets vertically."""

from typing import Literal

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class UnionInput(BaseModel):
    dataset_refs: list[str] = Field(
        description="List of dataset references to combine (2 or more)"
    )
    mode: Literal["by_name", "by_position"] = Field(
        default="by_name",
        description="by_name: align columns by name. by_position: align by column order."
    )
    ignore_index: bool = Field(
        default=True, description="Reset index in result (recommended)"
    )


@tool(args_schema=UnionInput)
def union_tool(
    dataset_refs: list[str],
    mode: str = "by_name",
    ignore_index: bool = True
) -> str:
    """
    Combine multiple datasets by stacking rows vertically (SQL UNION).
    by_name: columns matched by name, missing filled with NaN.
    by_position: columns matched by position (faster, requires same column count).
    """
    try:
        if len(dataset_refs) < 2:
            return "✗ union requires at least 2 datasets"
        
        warnings = []
        dfs = []
        
        for ref in dataset_refs:
            df = resolve_dataset(ref)
            dfs.append(df)
        
        # Check column alignment
        col_counts = [len(df.columns) for df in dfs]
        col_names = [set(df.columns) for df in dfs]
        
        if mode == "by_position":
            if len(set(col_counts)) > 1:
                return f"✗ by_position requires same column count. Found: {col_counts}"
            # Rename columns to match first df for positional alignment
            base_cols = list(dfs[0].columns)
            aligned_dfs = [dfs[0]]
            for df in dfs[1:]:
                df_copy = df.copy()
                df_copy.columns = base_cols
                aligned_dfs.append(df_copy)
            result = pd.concat(aligned_dfs, ignore_index=ignore_index)
        else:  # by_name
            # Check for column mismatches
            all_cols = set().union(*col_names)
            for i, cols in enumerate(col_names):
                missing = all_cols - cols
                if missing:
                    warnings.append(f"Dataset {i+1} missing: {list(missing)[:3]}...")
            
            result = pd.concat(dfs, ignore_index=ignore_index)
        
        total_rows = sum(len(df) for df in dfs)
        ref = save_result(result, "unn")
        detail = f"{len(dfs)} datasets → {total_rows} rows (mode={mode})"
        return format_result(ref, result, "union", detail, warnings if warnings else None)
        
    except Exception as e:
        return f"✗ union failed: {e}"

