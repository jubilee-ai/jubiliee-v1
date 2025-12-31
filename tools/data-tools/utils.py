"""
Shared utility functions for data tools.

Centralizes common functionality:
- OpenAI client management
- Text tokenization and similarity
- Fuzzy matching and suggestions
- Type conversions
- ID generation
- Catalog access
"""

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load .env file from project root
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

# Base paths
DATA_TOOLS_DIR = Path(__file__).parent
DATASETS_DIR = DATA_TOOLS_DIR.parent.parent / "datasets"
CATALOG_PATH = DATASETS_DIR / "catalog.json"
SQL_DIR = DATASETS_DIR / "sql"


# =============================================================================
# OPENAI CLIENT
# =============================================================================

_openai_client = None


def get_openai_client():
    """
    Lazy load OpenAI client.
    
    Returns a singleton instance of the OpenAI client.
    Requires OPENAI_API_KEY environment variable to be set.
    """
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


# =============================================================================
# TEXT UTILITIES
# =============================================================================

def tokenize_text(text: str) -> list[str]:
    """
    Simple tokenization: lowercase, split on non-alphanumeric.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of lowercase tokens
    """
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score between -1 and 1
    """
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


# =============================================================================
# FUZZY MATCHING & SUGGESTIONS
# =============================================================================

def fuzzy_suggest(
    query: str, 
    options: list[str], 
    n: int = 3, 
    cutoff: float = 0.4
) -> str:
    """
    Find similar options and format as suggestion string.
    
    Args:
        query: The string to find matches for
        options: List of valid options to match against
        n: Maximum number of suggestions
        cutoff: Minimum similarity score (0-1)
    
    Returns:
        Formatted suggestion string like "Did you mean: option1, option2?"
        or "Available: option1, option2, ..." if no close matches
    """
    similar = get_close_matches(query.lower(), [o.lower() for o in options], n=n, cutoff=cutoff)
    
    if similar:
        # Map back to original case
        original_case = []
        for s in similar:
            for o in options:
                if o.lower() == s:
                    original_case.append(o)
                    break
        return f"Did you mean: {', '.join(original_case)}?"
    elif options:
        display = options[:10]
        suffix = f" (+{len(options) - 10} more)" if len(options) > 10 else ""
        return f"Available: {', '.join(display)}{suffix}"
    else:
        return "No options available."


def get_similar_matches(
    query: str, 
    options: list[str], 
    n: int = 3, 
    cutoff: float = 0.4
) -> list[str]:
    """
    Get similar matches for a query string.
    
    Args:
        query: The string to find matches for
        options: List of valid options to match against
        n: Maximum number of matches
        cutoff: Minimum similarity score (0-1)
    
    Returns:
        List of similar options (may be empty)
    """
    return get_close_matches(query.lower(), [o.lower() for o in options], n=n, cutoff=cutoff)


# =============================================================================
# TYPE CONVERSIONS
# =============================================================================

def pandas_dtype_to_sql_type(dtype_str: str) -> str:
    """
    Convert pandas dtype string to simplified SQL type name.
    
    Args:
        dtype_str: Pandas dtype as string (e.g., 'int64', 'float64', 'object')
    
    Returns:
        Simplified type name: 'integer', 'float', 'boolean', 'datetime', or 'text'
    """
    dtype_str = str(dtype_str).lower()
    
    if "int" in dtype_str:
        return "integer"
    elif "float" in dtype_str:
        return "float"
    elif "bool" in dtype_str:
        return "boolean"
    elif "datetime" in dtype_str:
        return "datetime"
    else:
        return "text"


def infer_simple_dtype(dtype_str: str) -> str:
    """
    Convert pandas dtype to simple type name for schema display.
    
    Args:
        dtype_str: Pandas dtype as string
    
    Returns:
        Simple type name: 'integer', 'float', 'boolean', 'datetime', 'string', or the original
    """
    dtype_str = str(dtype_str).lower()
    
    if "int" in dtype_str:
        return "integer"
    elif "float" in dtype_str:
        return "float"
    elif "bool" in dtype_str:
        return "boolean"
    elif "datetime" in dtype_str:
        return "datetime"
    elif "object" in dtype_str or "string" in dtype_str:
        return "string"
    else:
        return dtype_str


# =============================================================================
# ID GENERATION
# =============================================================================

def generate_unique_id(prefix: str, *parts) -> str:
    """
    Generate a unique ID with a prefix and optional content-based hash.
    
    Args:
        prefix: Prefix for the ID (e.g., 'q', 'join', 'ds')
        *parts: Optional strings to hash for content-based component
    
    Returns:
        Unique ID like 'q_abc123_def456' or 'join_abc123_def456'
    """
    if parts:
        content = ":".join(str(p) for p in parts)
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    else:
        content_hash = uuid.uuid4().hex[:8]
    
    unique_part = uuid.uuid4().hex[:6]
    return f"{prefix}_{content_hash}_{unique_part}"


def utc_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    
    Returns:
        ISO timestamp string like '2024-01-15T10:30:00Z'
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# =============================================================================
# CATALOG ACCESS
# =============================================================================

_catalog_cache: Optional[dict] = None


def get_catalog() -> dict:
    """
    Load and cache the dataset catalog.
    
    Returns:
        Catalog dict with 'datasets' and 'summary' keys
    """
    global _catalog_cache
    if _catalog_cache is None:
        with open(CATALOG_PATH, "r") as f:
            _catalog_cache = json.load(f)
    return _catalog_cache


def get_dataset_info(asset_id: str) -> Optional[dict]:
    """
    Get dataset metadata from catalog by asset_id.
    
    Args:
        asset_id: The dataset file identifier (e.g., 'Loan_default.csv' or 'csv/Loan_default.csv')
    
    Returns:
        Dataset metadata dict or None if not found
    """
    catalog = get_catalog()
    for ds in catalog.get("datasets", []):
        file_path = ds.get("file", "")
        # Match on full path or just the filename
        if file_path == asset_id or file_path.endswith(f"/{asset_id}"):
            return ds
    return None


def list_catalog_assets() -> list[str]:
    """
    List all asset IDs in the catalog.
    
    Returns:
        List of asset_id strings
    """
    catalog = get_catalog()
    return [ds.get("file", "") for ds in catalog.get("datasets", []) if ds.get("file")]


# =============================================================================
# SCHEMA UTILITIES
# =============================================================================

def build_schema_from_dataframe(df) -> list[dict]:
    """
    Build a schema list from a pandas DataFrame.
    
    Args:
        df: Pandas DataFrame
    
    Returns:
        List of dicts with 'name' and 'type' keys
    """
    schema = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        sql_type = pandas_dtype_to_sql_type(dtype)
        schema.append({"name": col, "type": sql_type})
    return schema


def truncate_columns_display(columns: list[str], max_cols: int = 10) -> list[str]:
    """
    Truncate column list for display, adding count of remaining.
    
    Args:
        columns: Full list of column names
        max_cols: Maximum columns to show
    
    Returns:
        Truncated list with '... +N more' suffix if needed
    """
    if len(columns) <= max_cols:
        return columns
    return columns[:max_cols] + [f"... +{len(columns) - max_cols} more"]


# =============================================================================
# IN-MEMORY DATASET REGISTRY
# =============================================================================

# Global registry for storing DataFrames from operations (joins, queries, etc.)
# This allows chaining operations by referencing previous results
_dataset_registry: dict = {}


def register_dataset(ref: str, df) -> str:
    """
    Register a DataFrame in the in-memory registry.
    
    Args:
        ref: Unique reference ID for this dataset
        df: Pandas DataFrame to store
    
    Returns:
        The reference ID
    """
    _dataset_registry[ref] = df
    return ref


def get_registered_dataset(ref: str):
    """
    Retrieve a DataFrame from the registry.
    
    Args:
        ref: Reference ID of the dataset
    
    Returns:
        The DataFrame or None if not found
    """
    return _dataset_registry.get(ref)


def list_registered_datasets() -> list[str]:
    """
    List all registered dataset references.
    
    Returns:
        List of reference IDs
    """
    return list(_dataset_registry.keys())


def clear_registry() -> None:
    """Clear all registered datasets from memory."""
    _dataset_registry.clear()


def clear_dataset_registry(prefix: str = None) -> int:
    """
    Clear registered datasets, optionally filtering by prefix.
    
    Args:
        prefix: If provided, only clear datasets starting with this prefix
    
    Returns:
        Number of datasets cleared
    """
    if prefix is None:
        count = len(_dataset_registry)
        _dataset_registry.clear()
        return count
    
    to_remove = [k for k in _dataset_registry if k.startswith(prefix)]
    for k in to_remove:
        del _dataset_registry[k]
    return len(to_remove)


def get_registered_dataset_info(ref: str) -> Optional[dict]:
    """
    Get info about a registered dataset.
    
    Args:
        ref: Reference ID of the dataset
    
    Returns:
        Dict with 'rows' and 'columns' count, or None if not found
    """
    df = _dataset_registry.get(ref)
    if df is not None:
        return {"rows": len(df), "columns": len(df.columns)}
    return None


# =============================================================================
# UNIFIED DATASET DISCOVERY
# =============================================================================

def get_all_available_datasets() -> dict[str, list[dict]]:
    """
    Get all available datasets from all sources.
    
    Returns:
        Dict with keys 'registered', 'sql_tables', 'catalog' containing
        lists of dataset info dicts.
    """
    result = {
        "registered": [],
        "sql_tables": [],
        "catalog": [],
    }
    
    # Registered datasets (from previous operations)
    for ref in list_registered_datasets():
        info = get_registered_dataset_info(ref)
        if info:
            result["registered"].append({
                "ref": ref,
                "rows": info["rows"],
                "columns": info["columns"],
            })
    
    # SQL warehouse tables (lazy import to avoid circular deps)
    try:
        from .sql_query import get_warehouse
        warehouse = get_warehouse()
        for t in warehouse.get_all_tables():
            result["sql_tables"].append({
                "ref": t.name,
                "rows": t.row_count,
                "columns": len(t.columns),
                "column_names": [c["name"] for c in t.columns],
            })
    except Exception:
        pass
    
    # Catalog assets
    for asset_id in list_catalog_assets():
        info = get_dataset_info(asset_id)
        if info:
            result["catalog"].append({
                "ref": asset_id,
                "name": info.get("name", ""),
                "rows": info.get("rows"),
                "columns": info.get("columns", []),
            })
    
    return result


def list_all_dataset_refs() -> list[str]:
    """
    Get a flat list of all available dataset reference IDs.
    
    Returns:
        List of reference IDs from all sources (registered, SQL, catalog)
    """
    refs = list_registered_datasets()
    
    # SQL tables
    try:
        from .sql_query import list_tables
        refs.extend(list_tables())
    except Exception:
        pass
    
    # Catalog assets
    refs.extend(list_catalog_assets())
    
    return refs

