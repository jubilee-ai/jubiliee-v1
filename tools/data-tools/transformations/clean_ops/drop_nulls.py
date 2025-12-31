"""Drop rows with null values."""

from typing import Literal, Optional, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class DropNullsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    columns: Optional[Union[str, list[str]]] = Field(
        default=None,
        description="Column(s) to check for nulls. If not specified, checks all columns."
    )
    how: Literal["any", "all"] = Field(
        default="any",
        description="'any': drop if any specified column is null. 'all': drop only if all are null."
    )


@tool(args_schema=DropNullsInput)
def drop_nulls_tool(
    dataset_ref: str,
    columns: Optional[Union[str, list[str]]] = None,
    how: Literal["any", "all"] = "any"
) -> str:
    """
    Drop rows containing null values.
    
    Examples:
    - Drop rows where any column is null: columns=None, how='any'
    - Drop rows where 'age' is null: columns='age'
    - Drop rows where both 'a' and 'b' are null: columns=['a', 'b'], how='all'
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        subset = None
        if columns:
            subset = [columns] if isinstance(columns, str) else list(columns)
            missing = [c for c in subset if c not in df.columns]
            if missing:
                return f"✗ Columns not found: {missing}. Available: {list(df.columns)}"
        
        result = df.dropna(subset=subset, how=how).reset_index(drop=True)
        
        dropped = len(df) - len(result)
        ref = save_result(result, "drp")
        
        col_desc = subset if subset else "all"
        detail = f"dropped {dropped} rows (how={how}, cols={col_desc})"
        
        warnings = None
        if len(result) == 0:
            warnings = ["Result is empty!"]
        elif dropped == 0:
            warnings = ["No nulls found"]
        
        return format_result(ref, result, "drop_nulls", detail, warnings)
        
    except Exception as e:
        return f"✗ drop_nulls failed: {e}"

