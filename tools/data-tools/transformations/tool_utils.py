"""Shared utilities for LangChain transformation tools."""

import sys
from pathlib import Path

import pandas as pd

# Add paths for imports
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from ..utils import (clear_dataset_registry, generate_unique_id,
                         get_registered_dataset, register_dataset)
except ImportError:
    from utils import (clear_dataset_registry, generate_unique_id,
                       get_registered_dataset, register_dataset)


def resolve_dataset(dataset_ref: str) -> pd.DataFrame:
    """Resolve a dataset_ref to a DataFrame.
    
    Checks the in-memory registry first, then falls back to the data_loader.
    If loaded from data_loader, the dataset is registered for faster subsequent access.
    """
    df = get_registered_dataset(dataset_ref)
    if df is not None:
        return df
    
    # Try loading from data_loader
    try:
        try:
            from ..data_loader import dataset_get
        except ImportError:
            from data_loader import dataset_get
        result = dataset_get(asset_id=dataset_ref, limit=-1, include_stats=False)
        df = pd.DataFrame(result.data)
        register_dataset(dataset_ref, df)
        return df
    except Exception:
        pass
    
    raise ValueError(f"Dataset not found: '{dataset_ref}'")


def save_result(df: pd.DataFrame, prefix: str = "ds") -> str:
    """Register a DataFrame and return its reference."""
    ref = generate_unique_id(prefix)
    register_dataset(ref, df)
    return ref


def get_dtype_summary(df: pd.DataFrame) -> str:
    """Get a compact dtype summary."""
    type_counts = {}
    for dtype in df.dtypes:
        dtype_str = str(dtype)
        if 'int' in dtype_str:
            key = 'int'
        elif 'float' in dtype_str:
            key = 'float'
        elif 'datetime' in dtype_str:
            key = 'datetime'
        elif 'bool' in dtype_str:
            key = 'bool'
        elif 'category' in dtype_str:
            key = 'category'
        else:
            key = 'str'
        type_counts[key] = type_counts.get(key, 0) + 1
    
    parts = [f"{v}{k[0]}" for k, v in sorted(type_counts.items())]
    return "/".join(parts)


def format_result(ref: str, df: pd.DataFrame, op_name: str, details: str = "", warnings: list[str] = None) -> str:
    """Format a success response for the agent."""
    dtype_summary = get_dtype_summary(df)
    msg = f"✓ {op_name}: {len(df):,} rows × {len(df.columns)} cols [{dtype_summary}] → `{ref}`"
    if details:
        msg += f" ({details})"
    if warnings:
        for w in warnings:
            msg += f" ⚠ {w}"
    return msg


def cleanup_datasets(prefix: str = None) -> int:
    """Clear registered datasets. Returns count of cleared datasets."""
    return clear_dataset_registry(prefix)

