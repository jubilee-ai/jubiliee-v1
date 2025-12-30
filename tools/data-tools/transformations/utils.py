"""Shared utilities for transformation operations."""

import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

# Add types to path
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from jtypes.transformations import SchemaChange, TransformAudit, TransformWarning

from .base import get_dtype_string


@contextmanager
def timed_execution():
    """Context manager that yields elapsed time in ms."""
    start = time.perf_counter()
    result = {"ms": 0.0}
    yield result
    result["ms"] = round((time.perf_counter() - start) * 1000, 2)


def make_audit(
    op_name: str,
    params: dict,
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    schema_changes: list[SchemaChange] = None,
    warnings: list[TransformWarning] = None,
    execution_time_ms: float = None,
) -> TransformAudit:
    """Create a TransformAudit record."""
    return TransformAudit(
        op_name=op_name,
        op_params=params,
        rows_before=len(df_before),
        rows_after=len(df_after),
        columns_before=list(df_before.columns),
        columns_after=list(df_after.columns),
        schema_changes=schema_changes or [],
        warnings=warnings or [],
        success=True,
        execution_time_ms=execution_time_ms,
    )


def column_removed(df: pd.DataFrame, col: str) -> SchemaChange:
    """Create a 'removed' schema change."""
    return SchemaChange(change_type="removed", column=col, old_dtype=get_dtype_string(df[col].dtype))


def column_added(df: pd.DataFrame, col: str) -> SchemaChange:
    """Create an 'added' schema change."""
    return SchemaChange(change_type="added", column=col, new_dtype=get_dtype_string(df[col].dtype))


def column_modified(df_before: pd.DataFrame, df_after: pd.DataFrame, col: str) -> SchemaChange:
    """Create a 'modified' schema change."""
    return SchemaChange(
        change_type="modified", column=col,
        old_dtype=get_dtype_string(df_before[col].dtype),
        new_dtype=get_dtype_string(df_after[col].dtype),
    )


def column_renamed(df: pd.DataFrame, old_name: str, new_name: str) -> SchemaChange:
    """Create a 'renamed' schema change."""
    dtype = get_dtype_string(df[old_name].dtype)
    return SchemaChange(change_type="renamed", column=new_name, renamed_from=old_name, old_dtype=dtype, new_dtype=dtype)


def validate_required(value, name: str, type_name: str = "value"):
    """Raise ValueError if value is falsy."""
    if not value:
        raise ValueError(f"{name} requires a {type_name}")


def check_columns_exist(df: pd.DataFrame, columns: list[str], strict: bool = False):
    """Check which columns exist. Returns (valid_columns, warnings)."""
    existing = set(df.columns)
    valid, warnings = [], []
    
    for col in columns:
        if col in existing:
            valid.append(col)
        elif strict:
            raise ValueError(f"Column not found: '{col}'. Available: {list(df.columns)}")
        else:
            warnings.append(TransformWarning(
                code="COLUMN_NOT_FOUND", message=f"Column '{col}' not found, skipping", details={"column": col}
            ))
    
    return valid, warnings
