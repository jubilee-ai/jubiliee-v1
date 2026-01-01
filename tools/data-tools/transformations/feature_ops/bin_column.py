"""Bin column tool - discretize continuous variables into bins/buckets."""

from typing import Literal, Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class BinColumnInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Numeric column to bin/discretize")
    bins: Union[int, list[float]] = Field(
        description="Number of bins (int) OR list of bin edges. "
                    "Examples: 5 (creates 5 bins), [0, 25, 50, 75, 100] (custom edges)"
    )
    labels: Optional[list[str]] = Field(default=None, description="Labels for bins (len = bins-1 if edges)")
    strategy: Literal["uniform", "quantile"] = Field(default="uniform", description="'uniform' or 'quantile' (equal-frequency)")
    output_column: Optional[str] = Field(default=None, description="Output column name. Defaults to '{column}_binned'")


def _generate_labels(n_bins: int) -> list[str]:
    """Generate default labels: bin_1, bin_2, ..."""
    return [f"bin_{i+1}" for i in range(n_bins)]


@tool(args_schema=BinColumnInput)
def bin_column_tool(
    dataset_ref: str,
    column: str,
    bins: Union[int, list[float]],
    labels: Optional[list[str]] = None,
    strategy: Literal["uniform", "quantile"] = "uniform",
    output_column: Optional[str] = None,
) -> str:
    """
    Discretize a continuous column into bins/buckets.
    
    Use 'quantile' for skewed data (income), 'uniform' for normal data (age).
    
    Examples:
    - bin_column(col='age', bins=5, strategy='quantile') → 5 age quintiles
    - bin_column(col='credit_score', bins=[300, 580, 670, 740, 850], labels=['poor', 'fair', 'good', 'excellent'])
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        if not pd.api.types.is_numeric_dtype(df[column]):
            return f"✗ Column '{column}' is not numeric (dtype: {df[column].dtype})"
        
        result = df.copy()
        out_col = output_column or f"{column}_binned"
        
        # Generate default labels if not provided
        if labels is None:
            n_bins = bins if isinstance(bins, int) else len(bins) - 1
            labels = _generate_labels(n_bins)
        
        # Use pandas cut/qcut
        if isinstance(bins, int) and strategy == "quantile":
            result[out_col] = pd.qcut(df[column], q=bins, labels=labels, duplicates='drop')
        else:
            result[out_col] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
        
        ref = save_result(result, "bin")
        return format_result(ref, result, "bin_column", f"{column} → {out_col}")
        
    except Exception as e:
        return f"✗ bin_column failed: {e}"
