"""Cast transformation - convert column data types."""

from typing import Literal

import numpy as np
import pandas as pd

from .base import BaseTransform, TransformResult, TransformWarning
from .utils import (column_modified, make_audit, timed_execution,
                    validate_required)

# Type name mappings
DTYPE_MAP = {
    "integer": "Int64", "int": "Int64", "int32": "Int32", "int64": "Int64",
    "float": "float64", "float32": "float32", "float64": "float64", "double": "float64", "number": "float64",
    "string": "string", "str": "string", "text": "string", "object": "object",
    "boolean": "boolean", "bool": "boolean",
    "category": "category", "categorical": "category",
    "datetime": "datetime64[ns]", "date": "datetime64[ns]",
}


class CastTransform(BaseTransform):
    """Convert a column to a different data type."""
    
    op_name = "cast"
    
    def __init__(self, column: str, dtype: str, errors: Literal["coerce", "raise", "ignore"] = "coerce"):
        validate_required(column, "cast", "column name")
        validate_required(dtype, "cast", "dtype")
        dtype_lower = dtype.lower()
        if dtype_lower not in DTYPE_MAP:
            raise ValueError(f"Unknown dtype: '{dtype}'. Available: {list(DTYPE_MAP.keys())}")
        self.column = column
        self.dtype = dtype_lower
        self.pandas_dtype = DTYPE_MAP[dtype_lower]
        self.errors = errors
    
    def get_params(self) -> dict:
        return {"column": self.column, "dtype": self.dtype, "errors": self.errors}
    
    def validate(self, df: pd.DataFrame):
        if self.column not in df.columns:
            raise ValueError(f"Column not found: '{self.column}'")
        return []
    
    def execute(self, df: pd.DataFrame) -> TransformResult:
        self.validate(df)
        
        with timed_execution() as t:
            result = df.copy()
            series = df[self.column]
            original_nulls = series.isna().sum()
            warnings = []
            
            # Cast based on target type
            if self.pandas_dtype in ("Int64", "Int32"):
                casted = pd.to_numeric(series, errors=self.errors).astype(self.pandas_dtype)
            elif self.pandas_dtype in ("float64", "float32"):
                casted = pd.to_numeric(series, errors=self.errors).astype(self.pandas_dtype)
            elif self.pandas_dtype == "boolean":
                casted = self._to_boolean(series)
            elif "datetime" in self.pandas_dtype:
                casted = pd.to_datetime(series, errors=self.errors)
            else:
                casted = series.astype(self.pandas_dtype)
            
            result[self.column] = casted
            
            # Warn about conversion failures
            new_nulls = result[self.column].isna().sum()
            if new_nulls > original_nulls:
                warnings.append(TransformWarning(
                    "CAST_FAILED", f"{new_nulls - original_nulls} values could not convert, set to null"
                ))
            
            changes = [column_modified(df, result, self.column)]
        
        return TransformResult(result, make_audit(self.op_name, self.get_params(), df, result, changes, warnings, t["ms"]))
    
    def _to_boolean(self, series: pd.Series) -> pd.Series:
        """Convert to boolean with string handling."""
        if series.dtype == object or str(series.dtype) == "string":
            lower = series.str.lower()
            result = pd.array([None] * len(series), dtype="boolean")
            result[lower.isin(['true', 'yes', '1', 't', 'y'])] = True
            result[lower.isin(['false', 'no', '0', 'f', 'n'])] = False
            result[series.isna()] = pd.NA
            return pd.Series(result, index=series.index)
        return series.astype("boolean")


def cast(column: str, dtype: str, errors: Literal["coerce", "raise", "ignore"] = "coerce") -> CastTransform:
    """Create a cast transform. Example: cast('age', 'integer')"""
    return CastTransform(column, dtype, errors)
