"""Sample rows tool - random subset of rows."""

from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


class SampleRowsInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    n: Optional[int] = Field(default=None, description="Number of rows to sample")
    frac: Optional[float] = Field(default=None, description="Fraction of rows (0-1). Example: 0.1 for 10%")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


@tool(args_schema=SampleRowsInput)
def sample_rows_tool(dataset_ref: str, n: Optional[int] = None, frac: Optional[float] = None, seed: Optional[int] = None) -> str:
    """Take a random sample of rows."""
    try:
        if n is None and frac is None:
            return "✗ Provide either 'n' (count) or 'frac' (fraction)"
        if n is not None and frac is not None:
            return "✗ Provide 'n' or 'frac', not both"
        
        df = resolve_dataset(dataset_ref)
        
        warnings = None
        if n is not None:
            actual_n = min(n, len(df))
            if actual_n < n:
                warnings = [f"Requested {n} but only {len(df)} available"]
            result = df.sample(n=actual_n, random_state=seed).reset_index(drop=True)
            detail = f"n={actual_n}"
        else:
            result = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
            detail = f"frac={frac}"
        
        ref = save_result(result, "smp")
        return format_result(ref, result, "sample", detail, warnings)
        
    except Exception as e:
        return f"✗ sample failed: {e}"

