"""Extract datetime features tool - pull date/time components from datetime columns."""

from typing import Literal, Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


Feature = Literal[
    "year", "month", "day", "hour", "minute", "second",
    "dayofweek", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "is_month_start", "is_month_end"
]

# Map feature names to pandas dt accessor attributes/methods
FEATURE_MAP = {
    "year": lambda dt: dt.year,
    "month": lambda dt: dt.month,
    "day": lambda dt: dt.day,
    "hour": lambda dt: dt.hour,
    "minute": lambda dt: dt.minute,
    "second": lambda dt: dt.second,
    "dayofweek": lambda dt: dt.dayofweek,
    "dayofyear": lambda dt: dt.dayofyear,
    "weekofyear": lambda dt: dt.isocalendar().week.astype(int),
    "quarter": lambda dt: dt.quarter,
    "is_weekend": lambda dt: (dt.dayofweek >= 5).astype(int),
    "is_month_start": lambda dt: dt.is_month_start.astype(int),
    "is_month_end": lambda dt: dt.is_month_end.astype(int),
}


class ExtractDatetimeInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Datetime column to extract features from")
    features: list[Feature] = Field(
        description="Features: year, month, day, hour, minute, second, dayofweek, dayofyear, weekofyear, quarter, is_weekend, is_month_start, is_month_end"
    )
    prefix: Optional[str] = Field(default=None, description="Prefix for output columns (defaults to column name)")


@tool(args_schema=ExtractDatetimeInput)
def extract_datetime_tool(
    dataset_ref: str,
    column: str,
    features: list[Feature],
    prefix: Optional[str] = None,
) -> str:
    """
    Extract date/time components from a datetime column.
    
    Examples:
    - extract_datetime(col='date', features=['month', 'dayofweek', 'is_weekend'])
    - extract_datetime(col='timestamp', features=['hour', 'minute'], prefix='ts')
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        result = df.copy()
        col_dt = pd.to_datetime(df[column])
        col_prefix = prefix or column
        
        for feat in features:
            if feat not in FEATURE_MAP:
                return f"✗ Unknown feature '{feat}'. Available: {list(FEATURE_MAP.keys())}"
            result[f"{col_prefix}_{feat}"] = FEATURE_MAP[feat](col_dt.dt)
        
        ref = save_result(result, "dt")
        return format_result(ref, result, "extract_datetime", f"{column} → {len(features)} features")
        
    except Exception as e:
        return f"✗ extract_datetime failed: {e}"


class DateDiffInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    start_column: str = Field(description="Start datetime column")
    end_column: str = Field(description="End datetime column")
    unit: Literal["days", "hours", "minutes", "seconds"] = Field(default="days", description="Unit for difference")
    output_column: Optional[str] = Field(default=None, description="Output column name")


@tool(args_schema=DateDiffInput)
def date_diff_tool(
    dataset_ref: str,
    start_column: str,
    end_column: str,
    unit: Literal["days", "hours", "minutes", "seconds"] = "days",
    output_column: Optional[str] = None,
) -> str:
    """
    Calculate difference between two datetime columns.
    
    Examples:
    - date_diff(start='signup_date', end='purchase_date', unit='days')
    - date_diff(start='birth_date', end='now', unit='days') then divide by 365 for age
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        for col in [start_column, end_column]:
            if col not in df.columns:
                return f"✗ Column '{col}' not found"
        
        result = df.copy()
        out_col = output_column or f"{unit}_{start_column}_to_{end_column}"
        
        delta = pd.to_datetime(df[end_column]) - pd.to_datetime(df[start_column])
        
        divisors = {"days": 86400, "hours": 3600, "minutes": 60, "seconds": 1}
        result[out_col] = delta.dt.total_seconds() / divisors[unit]
        
        ref = save_result(result, "ddiff")
        return format_result(ref, result, "date_diff", f"({end_column} - {start_column}) in {unit}")
        
    except Exception as e:
        return f"✗ date_diff failed: {e}"
