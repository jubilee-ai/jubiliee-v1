"""Base utilities for transformation operations."""

import sys
from pathlib import Path

import pandas as pd

# Add types to path for imports
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from jtypes import transformations as _types

# Re-export types for backwards compatibility
DatasetInput = _types.DatasetInput
TransformWarning = _types.TransformWarning
SchemaChange = _types.SchemaChange
TransformAudit = _types.TransformAudit
TransformResult = _types.TransformResult
BaseTransform = _types.BaseTransform


def get_dtype_string(dtype) -> str:
    """Convert pandas dtype to readable string."""
    s = str(dtype).lower()
    for key, val in [("int", "integer"), ("float", "float"), ("bool", "boolean"),
                     ("datetime", "datetime"), ("object", "string"), ("category", "category")]:
        if key in s:
            return val
    return s


def validate_columns_exist(df: pd.DataFrame, columns: list[str], strict: bool = False):
    """Validate columns exist. Returns (valid_columns, warnings)."""
    existing = set(df.columns)
    valid, warnings = [], []
    
    for col in columns:
        if col in existing:
            valid.append(col)
        elif strict:
            raise ValueError(f"Column not found: '{col}'. Available: {list(df.columns)}")
        else:
            warnings.append(TransformWarning(
                code="COLUMN_NOT_FOUND",
                message=f"Column '{col}' not found, skipping",
                details={"column": col, "available": list(df.columns)[:10]}
            ))
    
    return valid, warnings
