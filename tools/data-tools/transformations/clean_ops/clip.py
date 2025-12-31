"""Clip column values to a range."""

from typing import Optional, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class ClipInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Numeric column to clip")
    min_val: Optional[float] = Field(
        default=None,
        description="Minimum value. Values below this will be set to this value."
    )
    max_val: Optional[float] = Field(
        default=None,
        description="Maximum value. Values above this will be set to this value."
    )


@tool(args_schema=ClipInput)
def clip_tool(
    dataset_ref: str,
    column: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> str:
    """
    Clip values in a column to a specified range.
    
    Examples:
    - Clip 'age' to [0, 120]: column='age', min_val=0, max_val=120
    - Cap 'score' at 100: column='score', max_val=100
    - Set floor at 0: column='amount', min_val=0
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        if min_val is None and max_val is None:
            return "✗ Provide at least one of 'min_val' or 'max_val'"
        
        result = df.copy()
        
        # Count values that will be clipped
        clipped_low = 0
        clipped_high = 0
        if min_val is not None:
            clipped_low = (result[column] < min_val).sum()
        if max_val is not None:
            clipped_high = (result[column] > max_val).sum()
        
        result[column] = result[column].clip(lower=min_val, upper=max_val)
        
        ref = save_result(result, "clp")
        bounds = f"[{min_val if min_val is not None else '-∞'}, {max_val if max_val is not None else '∞'}]"
        detail = f"'{column}' to {bounds}, clipped {clipped_low + clipped_high} values"
        
        warnings = None
        if clipped_low + clipped_high == 0:
            warnings = ["No values were outside the range"]
        
        return format_result(ref, result, "clip", detail, warnings)
        
    except Exception as e:
        return f"✗ clip failed: {e}"

