"""Impute missing values with statistical strategies."""

from typing import Literal, Optional, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class ImputeInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Column to impute")
    strategy: Literal["mean", "median", "mode", "ffill", "bfill", "group_median"] = Field(
        description="Imputation strategy: mean, median, mode (most frequent), "
                    "ffill (forward fill), bfill (backward fill), group_median (median per group)"
    )
    group_by: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to group by for group_median strategy. Required for group_median."
    )


@tool(args_schema=ImputeInput)
def impute_tool(
    dataset_ref: str,
    column: str,
    strategy: Literal["mean", "median", "mode", "ffill", "bfill", "group_median"],
    group_by: Optional[Union[str, list[str]]] = None
) -> str:
    """
    Impute missing values using statistical strategies.
    
    Strategies:
    - mean: Fill with column mean (numeric only)
    - median: Fill with column median (numeric only)
    - mode: Fill with most frequent value
    - ffill: Forward fill (propagate last valid value)
    - bfill: Backward fill (propagate next valid value)
    - group_median: Fill with median per group (requires group_by)
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        result = df.copy()
        null_before = result[column].isna().sum()
        
        if strategy == "mean":
            result[column] = result[column].fillna(result[column].mean())
        
        elif strategy == "median":
            result[column] = result[column].fillna(result[column].median())
        
        elif strategy == "mode":
            mode_val = result[column].mode()
            if len(mode_val) > 0:
                result[column] = result[column].fillna(mode_val.iloc[0])
        
        elif strategy == "ffill":
            result[column] = result[column].ffill()
        
        elif strategy == "bfill":
            result[column] = result[column].bfill()
        
        elif strategy == "group_median":
            if not group_by:
                return "✗ group_median requires 'group_by' parameter"
            grp_cols = [group_by] if isinstance(group_by, str) else list(group_by)
            missing_grp = [c for c in grp_cols if c not in df.columns]
            if missing_grp:
                return f"✗ Group columns not found: {missing_grp}. Available: {list(df.columns)}"
            
            result[column] = result.groupby(grp_cols)[column].transform(
                lambda x: x.fillna(x.median())
            )
        
        null_after = result[column].isna().sum()
        filled = null_before - null_after
        
        ref = save_result(result, "imp")
        detail = f"filled {filled} nulls in '{column}' using {strategy}"
        if group_by:
            detail += f" (by {group_by})"
        
        warnings = None
        if null_after > 0:
            warnings = [f"{null_after} nulls remain (no valid fill value)"]
        elif null_before == 0:
            warnings = ["No nulls found"]
        
        return format_result(ref, result, "impute", detail, warnings)
        
    except Exception as e:
        return f"✗ impute failed: {e}"

