"""Aggregate feature tool - create features by groupby + join-back pattern."""

from typing import Literal, Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result


AggFunc = Literal["mean", "sum", "min", "max", "count", "std", "median", "nunique", "first", "last"]

# Aggs that work well with transform() vs need merge pattern
TRANSFORM_AGGS = {"mean", "sum", "min", "max", "count", "std", "first", "last"}
MERGE_AGGS = {"median", "nunique"}  # These are slow/unsupported via transform


class AggFeatureInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    column: str = Field(description="Column to aggregate")
    agg: AggFunc = Field(description="Aggregation: mean, sum, min, max, count, std, median, nunique, first, last")
    group_by: Union[str, list[str]] = Field(description="Column(s) to group by")
    output_column: Optional[str] = Field(default=None, description="Output column name")


@tool(args_schema=AggFeatureInput)
def agg_feature_tool(
    dataset_ref: str,
    column: str,
    agg: AggFunc,
    group_by: Union[str, list[str]],
    output_column: Optional[str] = None,
) -> str:
    """
    Create an aggregated feature by computing a statistic within groups and joining back.
    
    Examples:
    - agg_feature(column='income', agg='mean', group_by='zip_code') → avg income per zip
    - agg_feature(column='claim_amount', agg='sum', group_by='customer_id') → total claims per customer
    """
    try:
        df = resolve_dataset(dataset_ref)
        
        if column not in df.columns:
            return f"✗ Column '{column}' not found. Available: {list(df.columns)}"
        
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            return f"✗ Group column(s) not found: {missing}"
        
        out_col = output_column or f"{agg}_{column}_by_{'_'.join(group_cols)}"
        result = df.copy()
        
        # Use transform for fast aggs, merge pattern for median/nunique
        if agg in TRANSFORM_AGGS:
            result[out_col] = df.groupby(group_cols)[column].transform(agg)
        else:
            # Merge pattern for median, nunique
            agg_df = df.groupby(group_cols, as_index=False)[column].agg(agg)
            agg_df = agg_df.rename(columns={column: out_col})
            result = result.merge(agg_df, on=group_cols, how="left")
        
        ref = save_result(result, "agg")
        return format_result(ref, result, "agg_feature", f"{agg}({column}) by {group_cols} → {out_col}")
        
    except Exception as e:
        return f"✗ agg_feature failed: {e}"


class MultiAggFeatureInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    aggs: dict[str, AggFunc] = Field(description="Dict of {column: agg_function}")
    group_by: Union[str, list[str]] = Field(description="Column(s) to group by")


@tool(args_schema=MultiAggFeatureInput)
def multi_agg_feature_tool(
    dataset_ref: str,
    aggs: dict[str, AggFunc],
    group_by: Union[str, list[str]],
) -> str:
    """
    Create multiple aggregated features at once.
    
    Example:
    multi_agg_feature(aggs={'income': 'mean', 'age': 'mean', 'debt': 'sum'}, group_by='zip_code')
    """
    try:
        df = resolve_dataset(dataset_ref)
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
        
        result = df.copy()
        group_suffix = '_'.join(group_cols)
        
        for col, agg_func in aggs.items():
            if col not in df.columns:
                return f"✗ Column '{col}' not found"
            out_col = f"{agg_func}_{col}_by_{group_suffix}"
            
            if agg_func in TRANSFORM_AGGS:
                result[out_col] = df.groupby(group_cols)[col].transform(agg_func)
            else:
                agg_df = df.groupby(group_cols, as_index=False)[col].agg(agg_func)
                agg_df = agg_df.rename(columns={col: out_col})
                result = result.merge(agg_df, on=group_cols, how="left")
        
        ref = save_result(result, "magg")
        return format_result(ref, result, "multi_agg_feature", f"{len(aggs)} features by {group_cols}")
        
    except Exception as e:
        return f"✗ multi_agg_feature failed: {e}"
