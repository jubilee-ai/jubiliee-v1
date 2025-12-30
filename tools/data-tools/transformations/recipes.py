"""Transformation recipes - predefined patterns for common data prep tasks."""

from typing import Callable

import pandas as pd

# Recipe type: function that takes df and optional config, returns list of ops
Recipe = Callable[[pd.DataFrame, dict], list[dict]]


def _get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=['number']).columns.tolist()


def _get_string_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=['object', 'string']).columns.tolist()


def _get_datetime_candidates(df: pd.DataFrame) -> list[str]:
    """Find string columns that look like dates."""
    candidates = []
    for col in _get_string_columns(df):
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
        try:
            pd.to_datetime(sample, format="mixed", errors='raise')
            candidates.append(col)
        except:
            pass
    return candidates


def _get_high_null_columns(df: pd.DataFrame, threshold: float = 0.5) -> list[str]:
    """Find columns with null rate above threshold."""
    null_rates = df.isnull().mean()
    return null_rates[null_rates > threshold].index.tolist()


def _get_low_cardinality_columns(df: pd.DataFrame, threshold: int = 20) -> list[str]:
    """Find string columns with low unique values (good for category)."""
    result = []
    for col in _get_string_columns(df):
        if df[col].nunique() <= threshold:
            result.append(col)
    return result


def _get_numeric_string_columns(df: pd.DataFrame) -> list[str]:
    """Find string columns that look like numbers."""
    result = []
    for col in _get_string_columns(df):
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
        try:
            pd.to_numeric(sample, errors='raise')
            result.append(col)
        except:
            pass
    return result


# =============================================================================
# RECIPES
# =============================================================================

def clean_for_ml(df: pd.DataFrame, config: dict = None) -> list[dict]:
    """
    Prepare data for machine learning.
    - Drop high-null columns
    - Parse date columns
    - Cast numeric-looking strings to numbers
    - Cast low-cardinality strings to category
    """
    config = config or {}
    null_threshold = config.get("null_threshold", 0.5)
    ops = []
    
    # Drop high-null columns
    high_null = _get_high_null_columns(df, null_threshold)
    if high_null:
        ops.append({"op": "drop", "columns": high_null})
    
    # Parse datetime columns
    date_cols = _get_datetime_candidates(df)
    for col in date_cols:
        if col not in high_null:
            ops.append({"op": "parse_datetime", "column": col})
    
    # Cast numeric-looking strings to float (before category check!)
    numeric_str_cols = _get_numeric_string_columns(df)
    for col in numeric_str_cols:
        if col not in high_null:
            ops.append({"op": "cast", "column": col, "dtype": "float"})
    
    # Cast low-cardinality to category (exclude numeric strings)
    cat_cols = _get_low_cardinality_columns(df)
    for col in cat_cols:
        if col not in high_null and col not in date_cols and col not in numeric_str_cols:
            ops.append({"op": "cast", "column": col, "dtype": "category"})
    
    return ops


def clean_basic(df: pd.DataFrame, config: dict = None) -> list[dict]:
    """
    Basic cleaning.
    - Parse obvious date columns
    - Infer and cast types
    """
    ops = []
    
    # Parse datetime columns
    for col in _get_datetime_candidates(df):
        ops.append({"op": "parse_datetime", "column": col})
    
    return ops


def prepare_for_join(df: pd.DataFrame, config: dict = None) -> list[dict]:
    """
    Prepare data for joining.
    - Cast join keys to consistent types
    - Strip whitespace from string keys
    """
    config = config or {}
    key_columns = config.get("key_columns", [])
    ops = []
    
    for col in key_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                # Add column that strips whitespace
                ops.append({"op": "add_column", "name": col, "expression": f"strip({col})", "overwrite": True})
    
    return ops


def select_features(df: pd.DataFrame, config: dict = None) -> list[dict]:
    """
    Select only specified feature columns.
    """
    config = config or {}
    columns = config.get("columns", [])
    
    if columns:
        return [{"op": "select", "columns": columns}]
    return []


def normalize_column_names(df: pd.DataFrame, config: dict = None) -> list[dict]:
    """
    Normalize column names to snake_case. Handles acronyms properly:
    CustomerID -> customer_id, HTTPServer -> http_server
    """
    import re
    mapping = {}
    for col in df.columns:
        # Handle acronyms: consecutive capitals stay together
        # "CustomerID" -> "Customer_ID", "HTTPServer" -> "HTTP_Server"
        new_name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', str(col))
        # Handle normal camelCase: "firstName" -> "first_Name"
        new_name = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', new_name)
        new_name = new_name.lower()
        # Clean up non-alphanumeric chars
        new_name = re.sub(r'[^a-z0-9_]', '_', new_name)
        new_name = re.sub(r'_+', '_', new_name).strip('_')
        if new_name != col:
            mapping[col] = new_name
    
    if mapping:
        return [{"op": "rename", "mapping": mapping}]
    return []


# Recipe registry
RECIPES: dict[str, Recipe] = {
    "clean_for_ml": clean_for_ml,
    "clean_basic": clean_basic,
    "prepare_for_join": prepare_for_join,
    "select_features": select_features,
    "normalize_names": normalize_column_names,
}


def get_recipe(name: str) -> Recipe:
    """Get a recipe by name."""
    if name not in RECIPES:
        raise ValueError(f"Unknown recipe: '{name}'. Available: {list(RECIPES.keys())}")
    return RECIPES[name]


def list_recipes() -> dict[str, str]:
    """List available recipes with descriptions."""
    return {
        "clean_for_ml": "Drop high-null columns, parse dates, cast categories",
        "clean_basic": "Parse date columns, basic type inference",
        "prepare_for_join": "Normalize join keys (strip whitespace, consistent types)",
        "select_features": "Keep only specified columns",
        "normalize_names": "Convert column names to snake_case",
    }

