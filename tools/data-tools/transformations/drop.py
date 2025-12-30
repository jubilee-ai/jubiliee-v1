"""Drop transformation - remove specific columns."""

import pandas as pd

from .base import BaseTransform, TransformResult, validate_columns_exist
from .utils import (column_removed, make_audit, timed_execution,
                    validate_required)


class DropTransform(BaseTransform):
    """Drop specific columns, keeping all others."""
    
    op_name = "drop"
    
    def __init__(self, columns: list[str], strict: bool = False):
        validate_required(columns, "drop", "columns list")
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
            result = df.drop(columns=valid, errors='ignore').copy()
            changes = [column_removed(df, c) for c in valid]
        
        return TransformResult(result, make_audit(self.op_name, self.get_params(), df, result, changes, warnings, t["ms"]))


def drop(columns: list[str], strict: bool = False) -> DropTransform:
    """Create a drop transform. Example: drop(['temp_col'])"""
    return DropTransform(columns, strict)
