"""Fill null values with a constant."""

from typing import Any, Optional, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class FillNullInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: Union[str, list[str]] = Field(
        description="Column(s) to fill. Use list for multiple columns with same value."
    )
    value: Any = Field(description="Value to fill nulls with (number, string, etc.)")


@tool(args_schema=FillNullInput)
def fill_null_tool(
    dataset_ref: str,
    column: Union[str, list[str]],
    value: Any
) -> str:
    """
    Fill null values in column(s) with a constant value.
    
    Examples:
    - Fill nulls in 'age' with 0: column='age', value=0
    - Fill nulls in 'status' with 'unknown': column='status', value='unknown'
    - Fill multiple columns: column=['a', 'b'], value=0
    """
    try:
        df = resolve_dataset(dataset_ref)
        cols = [column] if isinstance(column, str) else list(column)
        
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return f"✗ Columns not found: {missing}. Available: {list(df.columns)}"
        
        result = df.copy()
        filled_count = 0
        for col in cols:
            null_count = result[col].isna().sum()
            filled_count += null_count
            result[col] = result[col].fillna(value)
        
        ref = save_result(result, "fln")
        detail = f"filled {filled_count} nulls in {cols} with {repr(value)}"
        
        warnings = [f"No nulls found in {cols}"] if filled_count == 0 else None
        return format_result(ref, result, "fill_null", detail, warnings)
        
    except Exception as e:
        return f"✗ fill_null failed: {e}"

