"""Filter rows tool - keep rows matching a predicate."""

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..column_ops.add_column import ExpressionEvaluator
from ..tool_utils import format_result, resolve_dataset, save_result


class FilterRowsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    predicate: str = Field(
        description="Condition to filter rows. Examples: 'age >= 18', "
                    "'status == \"active\"', 'amount > 0 and category == \"A\"'"
    )


@tool(args_schema=FilterRowsInput)
def filter_rows_tool(dataset_ref: str, predicate: str) -> str:
    """Keep only rows where the predicate is true."""
    try:
        df = resolve_dataset(dataset_ref)
        
        evaluator = ExpressionEvaluator(df)
        mask = evaluator.evaluate(predicate)
        result = df.loc[mask].reset_index(drop=True)
        
        removed = len(df) - len(result)
        ref = save_result(result, "flt")
        
        warnings = []
        if len(result) == 0:
            warnings.append("Result is empty! Check your predicate.")
        elif len(result) < len(df) * 0.01 and len(df) > 100:
            warnings.append(f"Only {len(result)/len(df)*100:.1f}% of rows matched")
        
        return format_result(ref, result, "filter", f"kept {len(result)}, removed {removed}", warnings)
        
    except Exception as e:
        return f"✗ filter failed: {e}"

