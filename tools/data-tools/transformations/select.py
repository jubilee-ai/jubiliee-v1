"""Select transformation - keep specific columns."""

import pandas as pd

from .base import BaseTransform, TransformResult, validate_columns_exist
from .utils import (column_removed, make_audit, timed_execution,
                    validate_required)


class SelectTransform(BaseTransform):
    """Select specific columns, dropping all others."""
    
    op_name = "select"
    
    def __init__(self, columns: list[str], strict: bool = False):
        validate_required(columns, "select", "columns list")
        self.columns = columns
        self.strict = strict
    
    def get_params(self) -> dict:
        return {"columns": self.columns, "strict": self.strict}
    
    def validate(self, df: pd.DataFrame):
        _, warnings = validate_columns_exist(df, self.columns, self.strict)
        return warnings
    
    def execute(self, df: pd.DataFrame) -> TransformResult:
        with timed_execution() as t:
            valid, warnings = validate_columns_exist(df, self.columns, self.strict)
            if not valid:
                raise ValueError(f"No valid columns. Requested: {self.columns}, Available: {list(df.columns)}")
            
            result = df[valid].copy()
            removed = set(df.columns) - set(valid)
            changes = [column_removed(df, c) for c in removed]
        
        return TransformResult(result, make_audit(self.op_name, self.get_params(), df, result, changes, warnings, t["ms"]))


def select(columns: list[str], strict: bool = False) -> SelectTransform:
    """Create a select transform. Example: select(['id', 'name'])"""
    return SelectTransform(columns, strict)
