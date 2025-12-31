"""Cast column tool - convert column data type."""

from typing import Literal

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result

DTYPE_MAP = {
    "integer": "Int64", "int": "Int64",
    "float": "float64", "number": "float64",
    "string": "string", "text": "string",
    "boolean": "boolean", "bool": "boolean",
    "category": "category",
}


class CastInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Column to cast")
    dtype: str = Field(description="Target type: integer, float, string, boolean, category")


@tool(args_schema=CastInput)
def cast_tool(dataset_ref: str, column: str, dtype: str) -> str:
    """Cast a column to a different data type."""
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        dtype_lower = dtype.lower()
        if dtype_lower not in DTYPE_MAP:
            return f"✗ Unknown dtype '{dtype}'. Use: {list(DTYPE_MAP.keys())}"
        
        result = df.copy()
        target = DTYPE_MAP[dtype_lower]
        
        if target in ("Int64", "float64"):
            result[column] = pd.to_numeric(result[column], errors='coerce')
            if target == "Int64":
                result[column] = result[column].astype("Int64")
        elif target == "boolean":
            if result[column].dtype == object:
                lower = result[column].str.lower()
                result[column] = lower.isin(['true', 'yes', '1', 't', 'y'])
            else:
                result[column] = result[column].astype("boolean")
        else:
            result[column] = result[column].astype(target)
        
        ref = save_result(result, "cst")
        return format_result(ref, result, "cast", f"{column} → {dtype}", None)
        
    except Exception as e:
        return f"✗ cast failed: {e}"

