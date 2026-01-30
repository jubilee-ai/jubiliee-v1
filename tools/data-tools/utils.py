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
DERIVED_DATASETS_DIR = DATASETS_DIR / "derived"  # For newly created/joined datasets

# Ensure derived datasets directory exists
DERIVED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)


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
# DATASET REGISTRY (In-Memory + File Persistence)
# =============================================================================

# Global registry for storing DataFrames from operations (joins, queries, etc.)
# This allows chaining operations by referencing previous results
# Datasets are also persisted to DERIVED_DATASETS_DIR for durability
_dataset_registry: dict = {}


def _sanitize_ref_for_filename(ref: str) -> str:
    """Convert a dataset ref to a safe filename."""
    if ref is None:
        raise ValueError("Dataset reference cannot be None")
    # Replace problematic characters with underscores
    safe = re.sub(r'[^\w\-.]', '_', ref)
    return safe


def _sanitize_ref_for_sql(ref: str) -> str:
    """Convert a dataset ref to a valid SQL table name."""
    # SQL table names: start with letter/underscore, alphanumeric + underscore only
    safe = re.sub(r'[^\w]', '_', ref)
    if safe and safe[0].isdigit():
        safe = '_' + safe
    return safe.lower()


def _get_derived_path(ref: str) -> Path:
    """Get the file path for a derived dataset."""
    safe_name = _sanitize_ref_for_filename(ref)
    return DERIVED_DATASETS_DIR / f"{safe_name}.parquet"


# Track which tables we've added to the warehouse (for cleanup)
_derived_sql_tables: set = set()


def register_dataset(ref: str, df, persist: bool = True, register_sql: bool = True) -> str:
    """
    Register a DataFrame in the registry, persist to disk, and add to SQL warehouse.
    
    Derived datasets are:
    1. Stored in memory for fast access
    2. Saved to datasets/derived/ as parquet files for durability
    3. Added to the SQL warehouse as tables for SQL querying
    
    Args:
        ref: Unique reference ID for this dataset
        df: Pandas DataFrame to store
        persist: If True, save to disk as parquet (default True)
        register_sql: If True, add to SQL warehouse for SQL queries (default True)
    
    Returns:
        The reference ID
    """
    import pandas as pd
    
    _dataset_registry[ref] = df
    
    # Persist to disk if requested
    if persist and isinstance(df, pd.DataFrame):
        try:
            file_path = _get_derived_path(ref)
            df.to_parquet(file_path, index=False)
        except Exception as e:
            # Log but don't fail - in-memory is still available
            print(f"Warning: Failed to persist dataset '{ref}' to disk: {e}")
    
    # Register in SQL warehouse for SQL queries
    if register_sql and isinstance(df, pd.DataFrame):
        try:
            _register_in_sql_warehouse(ref, df)
        except Exception as e:
            # Log but don't fail - other access methods still work
            print(f"Warning: Failed to register '{ref}' in SQL warehouse: {e}")
    
    return ref


def _dedupe_column_names(df) -> "pd.DataFrame":
    """
    Deduplicate column names for SQL compatibility.
    SQLite is case-insensitive, so 'Age' and 'age' are duplicates.
    """
    import pandas as pd
    
    cols = list(df.columns)
    seen_lower = {}
    new_cols = []
    
    for col in cols:
        col_lower = col.lower()
        if col_lower in seen_lower:
            # Duplicate found - add suffix
            count = seen_lower[col_lower]
            seen_lower[col_lower] = count + 1
            new_col = f"{col}_{count}"
            new_cols.append(new_col)
        else:
            seen_lower[col_lower] = 1
            new_cols.append(col)
    
    if new_cols != cols:
        df = df.copy()
        df.columns = new_cols
    
    return df


def _register_in_sql_warehouse(ref: str, df) -> None:
    """Register a DataFrame as a table in the SQL warehouse."""
    from sqlalchemy import text
    
    # Import here to avoid circular imports
    from .sql_query import get_warehouse
    
    warehouse = get_warehouse()
    table_name = _sanitize_ref_for_sql(ref)
    
    # Dedupe column names for SQL (SQLite is case-insensitive)
    df = _dedupe_column_names(df)
    
    # Write DataFrame to SQL - replace if exists
    df.to_sql(table_name, warehouse.engine, if_exists='replace', index=False)
    
    # Update warehouse's table cache
    from sqlalchemy import inspect
    inspector = inspect(warehouse.engine)
    columns = [
        {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
        for c in inspector.get_columns(table_name)
    ]
    
    from .sql_query import TableInfo
    warehouse.tables[table_name] = TableInfo(
        name=table_name,
        columns=columns,
        row_count=len(df)
    )
    
    # Track for cleanup
    _derived_sql_tables.add(table_name)


def get_registered_dataset(ref: str):
    """
    Retrieve a DataFrame from the registry or disk.
    
    Checks in-memory registry first, then falls back to disk.
    
    Args:
        ref: Reference ID of the dataset
    
    Returns:
        The DataFrame or None if not found
    """
    import pandas as pd
    
    # Check in-memory first
    if ref in _dataset_registry:
        return _dataset_registry[ref]
    
    # Check disk (derived datasets)
    file_path = _get_derived_path(ref)
    if file_path.exists():
        try:
            df = pd.read_parquet(file_path)
            # Cache in memory for faster subsequent access
            _dataset_registry[ref] = df
            return df
        except Exception:
            pass
    
    return None


def list_registered_datasets() -> list[str]:
    """
    List all registered dataset references (in-memory and on disk).
    
    Returns:
        List of reference IDs
    """
    # Get in-memory refs
    refs = set(_dataset_registry.keys())
    
    # Add refs from disk
    if DERIVED_DATASETS_DIR.exists():
        for f in DERIVED_DATASETS_DIR.glob("*.parquet"):
            refs.add(f.stem)
    
    return sorted(refs)


def clear_registry(clear_disk: bool = False, clear_sql: bool = True) -> None:
    """
    Clear all registered datasets from memory, disk, and SQL warehouse.
    
    Args:
        clear_disk: If True, also delete files from datasets/derived/
        clear_sql: If True, also drop derived tables from SQL warehouse (default True)
    """
    global _derived_sql_tables
    
    _dataset_registry.clear()
    
    # Clear SQL tables
    if clear_sql and _derived_sql_tables:
        try:
            from sqlalchemy import text
            from .sql_query import get_warehouse
            warehouse = get_warehouse()
            with warehouse.engine.connect() as conn:
                for table_name in list(_derived_sql_tables):
                    try:
                        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                        if table_name in warehouse.tables:
                            del warehouse.tables[table_name]
                    except Exception:
                        pass
                conn.commit()
            _derived_sql_tables.clear()
        except Exception:
            pass
    
    # Clear disk files
    if clear_disk and DERIVED_DATASETS_DIR.exists():
        for f in DERIVED_DATASETS_DIR.glob("*.parquet"):
            try:
                f.unlink()
            except Exception:
                pass


def clear_dataset_registry(prefix: str = None, clear_disk: bool = False, clear_sql: bool = True) -> int:
    """
    Clear registered datasets, optionally filtering by prefix.
    
    Args:
        prefix: If provided, only clear datasets starting with this prefix
        clear_disk: If True, also delete files from datasets/derived/
        clear_sql: If True, also drop derived tables from SQL warehouse
    
    Returns:
        Number of datasets cleared
    """
    global _derived_sql_tables
    
    if prefix is None:
        count = len(_dataset_registry)
        _dataset_registry.clear()
        
        # Clear all SQL tables
        if clear_sql and _derived_sql_tables:
            try:
                from sqlalchemy import text
                from .sql_query import get_warehouse
                warehouse = get_warehouse()
                with warehouse.engine.connect() as conn:
                    for table_name in list(_derived_sql_tables):
                        try:
                            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                            if table_name in warehouse.tables:
                                del warehouse.tables[table_name]
                            count += 1
                        except Exception:
                            pass
                    conn.commit()
                _derived_sql_tables.clear()
            except Exception:
                pass
        
        if clear_disk and DERIVED_DATASETS_DIR.exists():
            for f in DERIVED_DATASETS_DIR.glob("*.parquet"):
                try:
                    f.unlink()
                    count += 1
                except Exception:
                    pass
        return count
    
    count = 0
    to_remove = [k for k in _dataset_registry if k.startswith(prefix)]
    for k in to_remove:
        del _dataset_registry[k]
        count += 1
    
    # Clear matching SQL tables
    if clear_sql:
        try:
            from sqlalchemy import text
            from .sql_query import get_warehouse
            warehouse = get_warehouse()
            safe_prefix = _sanitize_ref_for_sql(prefix)
            with warehouse.engine.connect() as conn:
                for table_name in list(_derived_sql_tables):
                    if table_name.startswith(safe_prefix):
                        try:
                            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                            if table_name in warehouse.tables:
                                del warehouse.tables[table_name]
                            _derived_sql_tables.discard(table_name)
                            count += 1
                        except Exception:
                            pass
                conn.commit()
        except Exception:
            pass
    
    if clear_disk and DERIVED_DATASETS_DIR.exists():
        for f in DERIVED_DATASETS_DIR.glob(f"{prefix}*.parquet"):
            try:
                f.unlink()
                count += 1
            except Exception:
                pass
    
    return count


def get_registered_dataset_info(ref: str) -> Optional[dict]:
    """
    Get info about a registered dataset.
    
    Args:
        ref: Reference ID of the dataset
    
    Returns:
        Dict with 'rows', 'columns', and 'persisted' status, or None if not found
    """
    df = get_registered_dataset(ref)
    if df is not None:
        file_path = _get_derived_path(ref)
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "persisted": file_path.exists(),
            "file_path": str(file_path) if file_path.exists() else None,
        }
    return None


def list_derived_datasets() -> list[dict]:
    """
    List all derived datasets stored on disk.
    
    Returns:
        List of dicts with ref, file_path, and size_mb
    """
    datasets = []
    if DERIVED_DATASETS_DIR.exists():
        for f in DERIVED_DATASETS_DIR.glob("*.parquet"):
            datasets.append({
                "ref": f.stem,
                "file_path": str(f),
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            })
    return datasets


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

