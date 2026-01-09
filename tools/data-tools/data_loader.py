"""
dataset_get - Retrieve a dataset (or slice) from an asset_id.

Inputs: asset_id, columns?, filters?, limit?, sample_strategy?
Outputs: {dataset_ref, schema, provenance, data, stats}

Optimized for AI agent consumption with:
- Smart defaults for data slicing
- Rich schema information
- Statistical summaries
- Provenance tracking
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .utils import (DATASETS_DIR, get_dataset_info, infer_simple_dtype,
                    utc_timestamp)


@dataclass
class ColumnSchema:
    """Schema information for a single column."""
    name: str
    dtype: str
    nullable: bool = True
    unique_count: Optional[int] = None
    sample_values: list[Any] = field(default_factory=list)
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    
    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "dtype": self.dtype,
            "nullable": self.nullable,
        }
        if self.unique_count is not None:
            result["unique_count"] = self.unique_count
        if self.sample_values:
            result["sample_values"] = self.sample_values[:5]
        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        if self.mean_value is not None:
            result["mean"] = round(self.mean_value, 4)
        return result


@dataclass
class DatasetResult:
    """Result from dataset_get operation."""
    dataset_ref: str
    schema: list[ColumnSchema]
    provenance: dict
    data: list[dict]
    stats: dict
    
    def to_dict(self) -> dict:
        return {
            "dataset_ref": self.dataset_ref,
            "schema": [col.to_dict() for col in self.schema],
            "provenance": self.provenance,
            "data": self.data,
            "stats": self.stats,
        }


def _load_csv_dataset(asset_id: str) -> "pd.DataFrame":
    """Load a CSV dataset."""
    import pandas as pd
    
    file_path = DATASETS_DIR / asset_id
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {asset_id}")
    
    return pd.read_csv(file_path)


def _load_huggingface_dataset(source: str) -> "pd.DataFrame":
    """Load a HuggingFace dataset."""
    import pandas as pd
    
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")
    
    ds = load_dataset(source)
    
    # Handle different dataset structures
    if hasattr(ds, 'keys'):
        # DatasetDict - get the first split (usually 'train')
        split_name = list(ds.keys())[0]
        return ds[split_name].to_pandas()
    else:
        return ds.to_pandas()


def _infer_column_schema(df: "pd.DataFrame", column: str) -> ColumnSchema:
    """Infer schema information for a column."""
    col_data = df[column]
    dtype = infer_simple_dtype(str(col_data.dtype))
    
    schema = ColumnSchema(
        name=column,
        dtype=dtype,
        nullable=col_data.isna().any(),
    )
    
    # Get unique count for non-numeric columns or low-cardinality numeric
    non_null = col_data.dropna()
    if len(non_null) > 0:
        schema.unique_count = int(non_null.nunique())
        
        # Sample values (for categorical/string columns)
        if dtype == "string" or schema.unique_count <= 20:
            schema.sample_values = [str(v) for v in non_null.unique()[:5].tolist()]
        
        # Numeric stats
        if dtype in ("integer", "float"):
            try:
                schema.min_value = float(non_null.min())
                schema.max_value = float(non_null.max())
                schema.mean_value = float(non_null.mean())
            except (TypeError, ValueError):
                pass
    
    return schema


def _apply_filters(df: "pd.DataFrame", filters: dict) -> "pd.DataFrame":
    """Apply filters to dataframe.
    
    Supports:
    - Exact match: {"column": value}
    - Comparison: {"column": {"$gt": 5, "$lt": 10}}
    - In list: {"column": {"$in": [1, 2, 3]}}
    - Not equal: {"column": {"$ne": value}}
    """
    result = df.copy()
    
    for column, condition in filters.items():
        if column not in result.columns:
            continue
        
        if isinstance(condition, dict):
            # Complex filter
            for op, value in condition.items():
                if op == "$gt":
                    result = result[result[column] > value]
                elif op == "$gte":
                    result = result[result[column] >= value]
                elif op == "$lt":
                    result = result[result[column] < value]
                elif op == "$lte":
                    result = result[result[column] <= value]
                elif op == "$eq":
                    result = result[result[column] == value]
                elif op == "$ne":
                    result = result[result[column] != value]
                elif op == "$in":
                    result = result[result[column].isin(value)]
                elif op == "$nin":
                    result = result[~result[column].isin(value)]
                elif op == "$contains":
                    result = result[result[column].astype(str).str.contains(str(value), case=False, na=False)]
        else:
            # Simple equality filter
            result = result[result[column] == condition]
    
    return result


def dataset_get(
    asset_id: str,
    columns: Optional[list[str]] = None,
    filters: Optional[dict] = None,
    limit: int = 100,
    sample_strategy: Literal["head", "tail", "random"] = "head",
    include_stats: bool = True,
) -> DatasetResult:
    """
    Retrieve a dataset or slice from an asset_id.
    
    Args:
        asset_id: The dataset file identifier (e.g., 'Loan_default.csv')
        columns: List of columns to retrieve. None = all columns.
        filters: Dict of column filters to apply (e.g., {"Age": {"$gt": 30}})
        limit: Maximum number of rows to return. Use -1 for all rows.
        sample_strategy: How to sample rows - 'head', 'tail', or 'random'
        include_stats: Whether to compute summary statistics
    
    Returns:
        DatasetResult with schema, data, provenance, and stats
    """
    import pandas as pd

    # Get dataset info from catalog
    ds_info = get_dataset_info(asset_id)
    if ds_info is None:
        raise ValueError(f"Dataset not found in catalog: {asset_id}")
    
    # Load the dataset
    fmt = ds_info.get("format", "")
    source = ds_info.get("source")
    file_path = ds_info.get("file", asset_id)  # Use catalog path or fallback to asset_id
    
    if "HuggingFace" in fmt and source:
        df = _load_huggingface_dataset(source)
    elif "CSV" in fmt:
        df = _load_csv_dataset(file_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")
    
    original_rows = len(df)
    
    # Apply filters
    if filters:
        df = _apply_filters(df, filters)
    
    filtered_rows = len(df)
    
    # Select columns
    if columns:
        available_cols = [c for c in columns if c in df.columns]
        missing_cols = [c for c in columns if c not in df.columns]
        if available_cols:
            df = df[available_cols]
    else:
        available_cols = list(df.columns)
        missing_cols = []
    
    # Apply limit and sampling
    if limit > 0 and len(df) > limit:
        if sample_strategy == "head":
            df_sample = df.head(limit)
        elif sample_strategy == "tail":
            df_sample = df.tail(limit)
        elif sample_strategy == "random":
            df_sample = df.sample(n=limit, random_state=42)
        else:
            df_sample = df.head(limit)
    else:
        df_sample = df
    
    # Build schema
    schema = [_infer_column_schema(df, col) for col in df_sample.columns]
    
    # Convert data to records (handle NaN for JSON)
    data = df_sample.where(pd.notnull(df_sample), None).to_dict(orient="records")
    
    # Build stats
    stats = {
        "total_rows_in_dataset": original_rows,
        "rows_after_filter": filtered_rows,
        "rows_returned": len(data),
        "columns_returned": len(df_sample.columns),
    }
    
    if missing_cols:
        stats["missing_columns"] = missing_cols
    
    if include_stats and len(df_sample) > 0:
        # Add numeric column summaries
        numeric_cols = df_sample.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            stats["numeric_summary"] = {}
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                stats["numeric_summary"][col] = {
                    "mean": round(df_sample[col].mean(), 4) if not df_sample[col].isna().all() else None,
                    "std": round(df_sample[col].std(), 4) if not df_sample[col].isna().all() else None,
                    "min": float(df_sample[col].min()) if not df_sample[col].isna().all() else None,
                    "max": float(df_sample[col].max()) if not df_sample[col].isna().all() else None,
                }
    
    # Build provenance
    provenance = {
        "asset_id": asset_id,
        "name": ds_info.get("name", ""),
        "source": source or "local",
        "format": fmt,
        "retrieved_at": utc_timestamp(),
        "filters_applied": filters or {},
        "columns_requested": columns,
        "sample_strategy": sample_strategy if limit > 0 else "all",
        "limit": limit,
    }
    
    return DatasetResult(
        dataset_ref=asset_id,
        schema=schema,
        provenance=provenance,
        data=data,
        stats=stats,
    )


# =============================================================================
# LANGCHAIN TOOL DEFINITION
# =============================================================================

class DatasetGetInput(BaseModel):
    """Input schema for dataset_get tool."""
    
    asset_id: str = Field(
        description="The dataset file identifier from catalog search results. "
                    "Examples: 'Loan_default.csv', 'insurance.csv', 'credit_card_risk_dataset.py'"
    )
    columns: Optional[list[str]] = Field(
        default=None,
        description="List of specific columns to retrieve. If not provided, all columns are returned. "
                    "Example: ['Age', 'Income', 'CreditScore', 'Default']"
    )
    filters: Optional[dict] = Field(
        default=None,
        description="Filters to apply to the data. Supports: "
                    "exact match {'column': value}, "
                    "comparisons {'column': {'$gt': 5, '$lt': 10}}, "
                    "in list {'column': {'$in': [1, 2, 3]}}, "
                    "contains {'column': {'$contains': 'text'}}. "
                    "Example: {'Age': {'$gte': 30}, 'Default': 1}"
    )
    limit: int = Field(
        default=5,
        description="Maximum number of rows to return. Default is 5 for quick preview. "
                    "Use -1 for all rows (caution: large datasets)."
    )
    sample_strategy: Literal["head", "tail", "random"] = Field(
        default="head",
        description="How to sample rows when limit is applied: "
                    "'head' = first N rows, 'tail' = last N rows, 'random' = random sample"
    )


@tool(args_schema=DatasetGetInput)
def dataset_get_tool(
    asset_id: str,
    columns: Optional[list[str]] = None,
    filters: Optional[dict] = None,
    limit: int = 5,
    sample_strategy: Literal["head", "tail", "random"] = "head",
) -> str:
    """
    Retrieve a sample of data from a catalog dataset.
    
    Returns schema info and a data preview (default 5 rows, max 10).
    Use this to inspect dataset structure and sample values.
    For full data queries, use sql_query_tool on SQL tables.
    
    Args:
        asset_id: Dataset ID from catalog_search_tool (e.g., "csv/insurance.csv")
        columns: Optional list of columns to return
        filters: Optional dict of {column: value} filters
        limit: Number of rows to return (default 5, max 10)
        sample_strategy: "head", "tail", or "random"
    """
    limit = min(limit, 10)  # Cap at 10 rows
    try:
        result = dataset_get(
            asset_id=asset_id,
            columns=columns,
            filters=filters,
            limit=limit,
            sample_strategy=sample_strategy,
            include_stats=True,
        )
        
        result_dict = result.to_dict()
        
        # Format output for agent consumption
        output_lines = []
        
        # Header
        output_lines.append(f"## Dataset: {result_dict['provenance']['name']}")
        output_lines.append(f"**Asset ID:** `{asset_id}`")
        output_lines.append("")
        
        # Stats summary
        stats = result_dict['stats']
        output_lines.append("### Summary")
        output_lines.append(f"- Total rows in dataset: {stats['total_rows_in_dataset']:,}")
        if stats.get('rows_after_filter') != stats['total_rows_in_dataset']:
            output_lines.append(f"- Rows after filtering: {stats['rows_after_filter']:,}")
        output_lines.append(f"- Rows returned: {stats['rows_returned']}")
        output_lines.append(f"- Columns returned: {stats['columns_returned']}")
        output_lines.append("")
        
        # Schema
        output_lines.append("### Schema")
        for col in result_dict['schema']:
            col_info = f"- **{col['name']}** ({col['dtype']})"
            if col.get('sample_values'):
                samples = ", ".join(str(v) for v in col['sample_values'][:3])
                col_info += f" — samples: {samples}"
            elif col.get('min') is not None:
                col_info += f" — range: [{col['min']}, {col['max']}]"
            output_lines.append(col_info)
        output_lines.append("")
        
        # Data preview (show all returned rows, already capped at 10)
        data = result_dict['data']
        if data:
            output_lines.append("### Data Preview")
            output_lines.append("```json")
            output_lines.append(json.dumps(data, indent=2, default=str))
            output_lines.append("```")
        output_lines.append("")
        
        return "\n".join(output_lines)
        
    except FileNotFoundError as e:
        return f"Error: Dataset file not found. {e}"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error loading dataset: {type(e).__name__}: {e}"


# Create a list of tools for easy import
data_loader_tools = [dataset_get_tool]

