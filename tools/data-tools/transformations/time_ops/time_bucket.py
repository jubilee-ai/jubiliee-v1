"""Time bucket tool - aggregate timestamps into time periods."""

from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result

# Fixed frequencies for dt.floor
FLOOR_FREQS = {
    "second": "s",
    "minute": "min",
    "hour": "h",
    "day": "D",
}

# Non-fixed frequencies need special handling
PERIOD_FREQS = {
    "week": "W",
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}


class TimeBucketInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Datetime column to bucket")
    granularity: str = Field(
        description="Time granularity: second, minute, hour, day, week, month, quarter, year"
    )
    output_column: Optional[str] = Field(
        default=None,
        description="Name for bucketed column. Defaults to '{column}_{granularity}'"
    )


@tool(args_schema=TimeBucketInput)
def time_bucket_tool(
    dataset_ref: str,
    column: str,
    granularity: str,
    output_column: Optional[str] = None
) -> str:
    """
    Bucket a datetime column into time periods (hour, day, week, month, etc.).
    Creates a new column with the period start timestamp.
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        granularity_lower = granularity.lower()
        all_granularities = list(FLOOR_FREQS.keys()) + list(PERIOD_FREQS.keys())
        if granularity_lower not in all_granularities:
            return f"✗ Unknown granularity '{granularity}'. Use: {all_granularities}"
        
        result = df.copy()
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(result[column]):
            result[column] = pd.to_datetime(result[column], errors='coerce')
        
        # Create bucketed column
        out_col = output_column or f"{column}_{granularity_lower}"
        
        if granularity_lower in FLOOR_FREQS:
            # Use floor for fixed frequencies
            result[out_col] = result[column].dt.floor(FLOOR_FREQS[granularity_lower])
        else:
            # Use period conversion for non-fixed frequencies
            freq = PERIOD_FREQS[granularity_lower]
            result[out_col] = result[column].dt.to_period(freq).dt.start_time
        
        ref = save_result(result, "tbk")
        return format_result(ref, result, "time_bucket", f"{column} → {out_col} ({granularity_lower})", None)
        
    except Exception as e:
        return f"✗ time_bucket failed: {e}"

