"""Replace values using a mapping."""

from typing import Any, Union

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class ReplaceValuesInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Column to apply replacements to")
    mapping: dict[Any, Any] = Field(
        description="Mapping of old values to new values. "
                    "Example: {'M': 'Male', 'F': 'Female'} or {-1: None, 999: None}"
    )


@tool(args_schema=ReplaceValuesInput)
def replace_values_tool(
    dataset_ref: str,
    column: str,
    mapping: dict
) -> str:
    """
    Replace values in a column using a mapping dictionary.
    
    Examples:
    - Expand codes: column='gender', mapping={'M': 'Male', 'F': 'Female'}
    - Replace sentinel values: column='age', mapping={-1: None, 999: None}
    - Recode categories: column='status', mapping={'A': 'Active', 'I': 'Inactive'}
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        if not mapping:
            return "✗ Mapping is empty. Provide at least one key-value pair."
        
        result = df.copy()
        
        # Count replacements
        replaced = result[column].isin(mapping.keys()).sum()
        result[column] = result[column].replace(mapping)
        
        ref = save_result(result, "rpl")
        detail = f"'{column}': replaced {replaced} values using {len(mapping)} mappings"
        
        warnings = None
        if replaced == 0:
            warnings = ["No values matched the mapping keys"]
        
        return format_result(ref, result, "replace_values", detail, warnings)
        
    except Exception as e:
        return f"✗ replace_values failed: {e}"

