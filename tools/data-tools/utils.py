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
import logging
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _sql_query_mod():
    """Load sql_query whether ``utils`` is a package submodule or top-level (sys.path)."""
    try:
        from . import sql_query as sq
    except ImportError:
        import sql_query as sq
    return sq


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
# R2 ARTIFACT HELPERS
# =============================================================================


def _get_r2_store():
    """Get the R2 artifact store (R2 in production, local FS in dev)."""
    try:
        from backend.shared.artifact_store import get_artifact_store
        return get_artifact_store()
    except Exception:
        return None


def _upload_parquet_to_r2(df, storage_key: str) -> bool:
    """Upload a DataFrame as parquet to R2. Returns True on success."""
    store = _get_r2_store()
    if not store:
        return False
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name, index=False)
            tmp_path = f.name
        store.upload(Path(tmp_path), storage_key)
        return True
    except Exception as e:
        logger.warning(f"R2 upload failed for {storage_key}: {e}")
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _download_parquet_from_r2(storage_key: str):
    """Download a parquet file from R2 and return as DataFrame. Returns None on failure."""
    store = _get_r2_store()
    if not store:
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        store.download(storage_key, Path(tmp_path))
        import pandas as pd
        return pd.read_parquet(tmp_path)
    except Exception as e:
        logger.warning(f"R2 download failed for {storage_key}: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# =============================================================================
# DATASET REGISTRY (In-Memory + R2 Persistence)
# =============================================================================

import threading as _threading
_dataset_registry: dict = {}
_registry_lock = _threading.Lock()


def _sanitize_ref_for_sql(ref: str) -> str:
    """Convert a dataset ref to a valid SQL table name."""
    # SQL table names: start with letter/underscore, alphanumeric + underscore only
    safe = re.sub(r'[^\w]', '_', ref)
    if safe and safe[0].isdigit():
        safe = '_' + safe
    return safe.lower()


# Track which tables we've added to the warehouse (for cleanup)
_derived_sql_tables: set = set()


def register_dataset(
    ref: str,
    df,
    persist: bool = True,
    register_sql: bool = True,
    source_type: str = "derived",
) -> str:
    """
    Register a DataFrame in the registry, persist to R2, and add to SQL warehouse.
    
    Derived datasets are:
    1. Stored in memory for fast access
    2. Uploaded to R2 as parquet for durability
    3. Added to the SQL warehouse as tables for SQL querying
    4. Registered in Postgres datasets table (metadata + storage_key)
    
    Args:
        ref: Unique reference ID for this dataset
        df: Pandas DataFrame to store
        persist: If True, upload to R2 as parquet (default True)
        register_sql: If True, add to SQL warehouse for SQL queries (default True)
        source_type: Dataset origin -- "derived", "local", "kaggle", "huggingface"
    
    Returns:
        The reference ID
    """
    import pandas as pd
    
    with _registry_lock:
        _dataset_registry[ref] = df
    
    storage_key = f"datasets/{source_type}/{ref}.parquet"

    # Persist to R2 if requested
    if persist and isinstance(df, pd.DataFrame):
        if not _upload_parquet_to_r2(df, storage_key):
            logger.warning(f"Dataset '{ref}' not persisted to R2")
    
    # Register in SQL warehouse for SQL queries
    if register_sql and isinstance(df, pd.DataFrame):
        try:
            _register_in_sql_warehouse(ref, df)
        except Exception as e:
            print(f"Warning: Failed to register '{ref}' in SQL warehouse: {e}")

    # Register metadata in Postgres (non-blocking)
    if isinstance(df, pd.DataFrame):
        try:
            _register_in_postgres(ref, df, source_type, storage_key)
        except Exception as e:
            logger.error(f"Postgres registration failed for '{ref}': {e}")

    return ref


def _register_in_postgres(ref: str, df, source_type: str, storage_key: str) -> None:
    """Write dataset metadata to Postgres datasets table. Best-effort."""
    try:
        from backend.shared.database import get_db_session
        from backend.shared.models import Dataset
    except ImportError:
        return

    import pandas as pd

    properties: dict = {
        "format": "parquet",
        "storage_key": storage_key,
        "row_count": len(df),
        "columns": list(df.columns),
    }
    try:
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        properties["schema_info"] = dtypes
    except Exception:
        pass

    with get_db_session() as session:
        existing = session.query(Dataset).filter(Dataset.name == ref).first()
        if existing:
            existing.properties = {**(existing.properties or {}), **properties}
        else:
            session.add(Dataset(name=ref, source_type=source_type, properties=properties))


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

    sq = _sql_query_mod()
    warehouse = sq.get_warehouse()
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
    
    warehouse.tables[table_name] = sq.TableInfo(
        name=table_name,
        columns=columns,
        row_count=len(df)
    )
    
    # Track for cleanup
    _derived_sql_tables.add(table_name)


def get_registered_dataset(ref: str):
    """
    Retrieve a DataFrame from the registry or R2.
    
    Checks in-memory registry first, then falls back to Postgres metadata + R2.
    
    Args:
        ref: Reference ID of the dataset
    
    Returns:
        The DataFrame or None if not found
    """
    with _registry_lock:
        if ref in _dataset_registry:
            return _dataset_registry[ref]
    
    # Try Postgres + R2
    try:
        from backend.shared.database import get_db_session
        from backend.shared.models import Dataset

        with get_db_session() as session:
            row = session.query(Dataset).filter(Dataset.name == ref).first()
            if row and row.properties and row.properties.get("storage_key"):
                df = _download_parquet_from_r2(row.properties["storage_key"])
                if df is not None:
                    with _registry_lock:
                        _dataset_registry[ref] = df
                    return df
    except Exception as e:
        logger.debug(f"Postgres/R2 lookup for {ref}: {e}")
    
    return None


def list_registered_datasets() -> list[str]:
    """
    List all registered dataset references (in-memory + Postgres).
    
    Returns:
        List of reference IDs
    """
    with _registry_lock:
        refs = set(_dataset_registry.keys())

    try:
        from backend.shared.database import get_db_session
        from backend.shared.models import Dataset

        with get_db_session() as session:
            for row in session.query(Dataset.name).all():
                refs.add(row.name)
    except Exception:
        pass
    
    return sorted(refs)


def clear_registry(clear_sql: bool = True) -> None:
    """
    Clear all registered datasets from memory and SQL warehouse.
    
    Args:
        clear_sql: If True, also drop derived tables from SQL warehouse (default True)
    """
    global _derived_sql_tables
    
    with _registry_lock:
        _dataset_registry.clear()
    
    if clear_sql and _derived_sql_tables:
        try:
            from sqlalchemy import text
            warehouse = _sql_query_mod().get_warehouse()
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


def clear_dataset_registry(prefix: str = None, clear_sql: bool = True) -> int:
    """
    Clear registered datasets, optionally filtering by prefix.
    
    Args:
        prefix: If provided, only clear datasets starting with this prefix
        clear_sql: If True, also drop derived tables from SQL warehouse
    
    Returns:
        Number of datasets cleared
    """
    global _derived_sql_tables
    
    if prefix is None:
        with _registry_lock:
            count = len(_dataset_registry)
            _dataset_registry.clear()
        
        if clear_sql and _derived_sql_tables:
            try:
                from sqlalchemy import text
                warehouse = _sql_query_mod().get_warehouse()
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
        
        return count
    
    count = 0
    with _registry_lock:
        to_remove = [k for k in _dataset_registry if k.startswith(prefix)]
        for k in to_remove:
            del _dataset_registry[k]
            count += 1
    
    if clear_sql:
        try:
            from sqlalchemy import text
            warehouse = _sql_query_mod().get_warehouse()
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
    
    return count


def get_registered_dataset_info(ref: str) -> Optional[dict]:
    """
    Get info about a registered dataset.

    Prefers in-memory, then Postgres metadata only (fast — no R2 download).
    Falls back to full resolution only when metadata is missing or incomplete.

    Args:
        ref: Reference ID of the dataset

    Returns:
        Dict with 'rows', 'columns', and 'persisted' status, or None if not found
    """
    storage_key: Optional[str] = None

    with _registry_lock:
        if ref in _dataset_registry:
            df = _dataset_registry[ref]
            try:
                from backend.shared.database import get_db_session
                from backend.shared.models import Dataset

                with get_db_session() as session:
                    row = session.query(Dataset).filter(Dataset.name == ref).first()
                    if row and row.properties:
                        storage_key = row.properties.get("storage_key")
            except Exception:
                pass
            return {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "persisted": storage_key is not None,
                "storage_key": storage_key,
            }

    # Postgres metadata only — avoids HeadObject/R2 for every stale ref when listing
    try:
        from backend.shared.database import get_db_session
        from backend.shared.models import Dataset

        with get_db_session() as session:
            row = session.query(Dataset).filter(Dataset.name == ref).first()
            if row and row.properties:
                props = row.properties or {}
                storage_key = props.get("storage_key")
                cols = props.get("columns")
                nrows = props.get("row_count")
                if isinstance(cols, list) and cols and nrows is not None:
                    return {
                        "rows": int(nrows),
                        "columns": len(cols),
                        "column_names": cols,
                        "persisted": storage_key is not None,
                        "storage_key": storage_key,
                    }
    except Exception:
        pass

    df = get_registered_dataset(ref)
    if df is None:
        return None

    try:
        from backend.shared.database import get_db_session
        from backend.shared.models import Dataset

        with get_db_session() as session:
            row = session.query(Dataset).filter(Dataset.name == ref).first()
            if row and row.properties:
                storage_key = row.properties.get("storage_key")
    except Exception:
        pass

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "persisted": storage_key is not None,
        "storage_key": storage_key,
    }


def list_derived_datasets() -> list[dict]:
    """
    List all derived datasets registered in Postgres.
    
    Returns:
        List of dicts with ref, storage_key, and row_count
    """
    datasets: list[dict] = []
    try:
        from backend.shared.database import get_db_session
        from backend.shared.models import Dataset

        with get_db_session() as session:
            for row in session.query(Dataset).all():
                props = row.properties or {}
                datasets.append({
                    "ref": row.name,
                    "storage_key": props.get("storage_key"),
                    "row_count": props.get("row_count"),
                })
    except Exception:
        pass
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
        warehouse = _sql_query_mod().get_warehouse()
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
        refs.extend(_sql_query_mod().list_tables())
    except Exception:
        pass
    
    # Catalog assets
    refs.extend(list_catalog_assets())
    
    return refs

