"""Replace values using regex patterns."""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class RegexReplaceInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="String column to apply regex replacement to")
    pattern: str = Field(description="Regex pattern to match. Example: r'\\d+' matches digits.")
    replacement: str = Field(
        description="Replacement string. Use \\1, \\2 for capture groups. "
                    "Example: r'\\1-\\2' reformats matched groups."
    )


@tool(args_schema=RegexReplaceInput)
def regex_replace_tool(
    dataset_ref: str,
    column: str,
    pattern: str,
    replacement: str
) -> str:
    """
    Replace values in a column using regex pattern matching.
    
    Examples:
    - Remove non-digits: pattern=r'[^0-9]', replacement=''
    - Extract domain: pattern=r'.*@(.+)', replacement=r'\\1'
    - Format phone: pattern=r'(\\d{3})(\\d{3})(\\d{4})', replacement=r'(\\1) \\2-\\3'
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        result = df.copy()
        
        # Count rows that will be modified (suppress group warning)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            matches = result[column].astype(str).str.contains(pattern, regex=True, na=False).sum()
        result[column] = result[column].astype(str).str.replace(pattern, replacement, regex=True)
        
        ref = save_result(result, "rgx")
        detail = f"'{column}': pattern '{pattern}' matched {matches} rows"
        
        warnings = None
        if matches == 0:
            warnings = ["Pattern did not match any values"]
        
        return format_result(ref, result, "regex_replace", detail, warnings)
        
    except Exception as e:
        return f"✗ regex_replace failed: {e}"

