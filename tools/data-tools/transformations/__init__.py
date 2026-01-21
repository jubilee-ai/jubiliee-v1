"""
Transformation tools for AI agents.

Usage:
    from transformations import transform_tools
    # Returns list of all LangChain tools for agent binding
"""

from .agg_ops import agg_tools, groupby_agg_tool
from .clean_ops import (
    clean_tools,
    clip_tool,
    drop_nulls_tool,
    fill_null_tool,
    impute_tool,
    regex_replace_tool,
    replace_values_tool,
)
from .column_ops import (
    add_column_tool,
    cast_tool,
    column_tools,
    drop_columns_tool,
    parse_datetime_tool,
    rename_columns_tool,
    select_columns_tool,
)
from .feature_ops import (
    agg_feature_tool,
    bin_column_tool,
    extract_datetime_tool,
    feature_tools,
)
from .row_ops import (
    dedupe_tool,
    filter_rows_tool,
    limit_rows_tool,
    row_tools,
    sample_rows_tool,
    sort_rows_tool,
)
from .time_ops import (
    lag_tool,
    lead_tool,
    rank_tool,
    rolling_max_tool,
    rolling_mean_tool,
    rolling_min_tool,
    rolling_sum_tool,
    row_number_tool,
    time_bucket_tool,
    time_tools,
)
from .reshape_ops import (
    pivot_tool,
    reshape_tools,
    union_tool,
    unpivot_tool,
)
from .tool_utils import cleanup_datasets, format_result, resolve_dataset, save_result
from .utility_tools import cleanup_datasets_tool, list_datasets_tool, utility_tools
from .apply_transformations import (
    apply_transformations,
    apply_transformations_tool,
    get_registered_tools,
    get_tool_descriptions,
)

# All tools for agent binding
transform_tools = column_tools + row_tools + time_tools + reshape_tools + agg_tools + clean_tools + feature_tools + utility_tools + [apply_transformations_tool]

__all__ = [
    # All tools combined
    "transform_tools",
    
    # Column tools
    "column_tools",
    "select_columns_tool",
    "drop_columns_tool",
    "rename_columns_tool",
    "cast_tool",
    "parse_datetime_tool",
    "add_column_tool",
    
    # Row tools
    "row_tools",
    "filter_rows_tool",
    "sort_rows_tool",
    "dedupe_tool",
    "sample_rows_tool",
    "limit_rows_tool",
    
    # Time/window tools
    "time_tools",
    "time_bucket_tool",
    "lag_tool",
    "lead_tool",
    "rolling_mean_tool",
    "rolling_sum_tool",
    "rolling_min_tool",
    "rolling_max_tool",
    "row_number_tool",
    "rank_tool",
    
    # Reshape tools
    "reshape_tools",
    "pivot_tool",
    "unpivot_tool",
    "union_tool",
    
    # Aggregation tools
    "agg_tools",
    "groupby_agg_tool",
    
    # Cleaning tools
    "clean_tools",
    "drop_nulls_tool",
    "fill_null_tool",
    "impute_tool",
    "clip_tool",
    "replace_values_tool",
    "regex_replace_tool",
    
    # Feature engineering tools
    "feature_tools",
    "bin_column_tool",
    "agg_feature_tool",
    "extract_datetime_tool",
    
    # Utility tools
    "utility_tools",
    "cleanup_datasets_tool",
    "list_datasets_tool",
    
    # Internal utilities
    "resolve_dataset",
    "save_result",
    "format_result",
    "cleanup_datasets",
    
    # Batch transformations
    "apply_transformations",
    "apply_transformations_tool",
    "get_registered_tools",
    "get_tool_descriptions",
]
