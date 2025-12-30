"""Parse datetime transformation - convert strings to datetime."""

from typing import Literal, Optional

import pandas as pd

from .base import (BaseTransform, SchemaChange, TransformResult,
                   TransformWarning, get_dtype_string)
from .utils import (column_added, column_modified, make_audit, timed_execution,
                    validate_required)


class ParseDatetimeTransform(BaseTransform):
    """Parse a column to datetime with optional format and timezone."""
    
    op_name = "parse_datetime"
    
    def __init__(
        self,
        column: str,
        format: Optional[str] = None,
        timezone: Optional[str] = None,
        errors: Literal["coerce", "raise", "ignore"] = "coerce",
        output_column: Optional[str] = None,
    ):
        validate_required(column, "parse_datetime", "column name")
        self.column = column
        self.format = format
        self.timezone = timezone
        self.errors = errors
        self.output_column = output_column or column
    
    def get_params(self) -> dict:
        p = {"column": self.column, "errors": self.errors}
        if self.format: p["format"] = self.format
        if self.timezone: p["timezone"] = self.timezone
        if self.output_column != self.column: p["output_column"] = self.output_column
        return p
    
    def validate(self, df: pd.DataFrame):
        if self.column not in df.columns:
            raise ValueError(f"Column not found: '{self.column}'")
        warnings = []
        if pd.api.types.is_datetime64_any_dtype(df[self.column]):
            warnings.append(TransformWarning("ALREADY_DATETIME", f"Column '{self.column}' already datetime"))
        return warnings
    
    def execute(self, df: pd.DataFrame) -> TransformResult:
        self.validate(df)
        
        with timed_execution() as t:
            result = df.copy()
            original_nulls = df[self.column].isna().sum()
            warnings = []
            
            parsed = pd.to_datetime(df[self.column], format=self.format or "mixed", errors=self.errors)
            
            if self.timezone:
                parsed = parsed.dt.tz_localize(self.timezone) if parsed.dt.tz is None else parsed.dt.tz_convert(self.timezone)
            
            result[self.output_column] = parsed
            
            new_nulls = result[self.output_column].isna().sum()
            if new_nulls > original_nulls:
                warnings.append(TransformWarning("PARSE_FAILED", f"{new_nulls - original_nulls} values could not parse"))
            
            if self.output_column == self.column:
                changes = [column_modified(df, result, self.column)]
            else:
                changes = [column_added(result, self.output_column)]
        
        return TransformResult(result, make_audit(self.op_name, self.get_params(), df, result, changes, warnings, t["ms"]))


def parse_datetime(
    column: str,
    format: Optional[str] = None,
    timezone: Optional[str] = None,
    errors: Literal["coerce", "raise", "ignore"] = "coerce",
    output_column: Optional[str] = None,
) -> ParseDatetimeTransform:
    """Create a parse_datetime transform. Example: parse_datetime('date_str', format='%Y-%m-%d')"""
    return ParseDatetimeTransform(column, format, timezone, errors, output_column)
