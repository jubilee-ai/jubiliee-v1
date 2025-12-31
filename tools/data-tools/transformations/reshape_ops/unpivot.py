"""Unpivot (melt) tool - reshape wide to long format."""

from typing import Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class UnpivotInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    id_cols: Union[str, list[str]] = Field(
        description="Column(s) to keep as identifiers (not unpivoted)"
    )
    value_cols: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to unpivot. If None, uses all non-id columns."
    )
    var_name: str = Field(
        default="variable", description="Name for the new column holding original column names"
    )
    value_name: str = Field(
        default="value", description="Name for the new column holding the values"
    )


@tool(args_schema=UnpivotInput)
def unpivot_tool(
    dataset_ref: str,
    id_cols: Union[str, list[str]],
    value_cols: Optional[Union[str, list[str]]] = None,
    var_name: str = "variable",
    value_name: str = "value"
) -> str:
    """
    Reshape data from wide to long format using melt.
    Keeps id_cols fixed, unpivots value_cols into (variable, value) pairs.
    Example: revenue_q1, revenue_q2, revenue_q3 → (quarter, revenue) rows.
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        # Normalize to lists
        id_list = [id_cols] if isinstance(id_cols, str) else list(id_cols)
        
        # Validate id columns
        missing_id = [c for c in id_list if c not in df.columns]
        if missing_id:
            return f"✗ ID columns not found: {missing_id}. Available: {list(df.columns)}"
        
        # Determine value columns
        if value_cols:
            val_list = [value_cols] if isinstance(value_cols, str) else list(value_cols)
            missing_val = [c for c in val_list if c not in df.columns]
            if missing_val:
                return f"✗ Value columns not found: {missing_val}. Available: {list(df.columns)}"
        else:
            val_list = [c for c in df.columns if c not in id_list]
            if not val_list:
                return "✗ No columns to unpivot after excluding id_cols"
        
        # Melt
        result = pd.melt(
            df,
            id_vars=id_list,
            value_vars=val_list,
            var_name=var_name,
            value_name=value_name
        )
        
        ref = save_result(result, "upt")
        detail = f"{len(val_list)} cols → ({var_name}, {value_name})"
        return format_result(ref, result, "unpivot", detail, None)
        
    except Exception as e:
        return f"✗ unpivot failed: {e}"

