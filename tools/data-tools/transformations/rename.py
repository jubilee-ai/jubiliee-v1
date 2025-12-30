"""Rename transformation - rename columns."""

import pandas as pd

from .base import BaseTransform, TransformResult, TransformWarning
from .utils import (column_renamed, make_audit, timed_execution,
                    validate_required)


class RenameTransform(BaseTransform):
    """Rename columns according to a mapping."""
    
    op_name = "rename"
    
    def __init__(self, mapping: dict[str, str], strict: bool = False):
        validate_required(mapping, "rename", "mapping dict")
        if len(set(mapping.values())) != len(mapping.values()):
            raise ValueError("Rename mapping has duplicate target names")
        self.mapping = mapping
        self.strict = strict
    
    def get_params(self) -> dict:
        return {"mapping": self.mapping, "strict": self.strict}
    
    def validate(self, df: pd.DataFrame):
        warnings = []
        existing = set(df.columns)
        for old in self.mapping:
            if old not in existing:
                if self.strict:
                    raise ValueError(f"Column not found: '{old}'")
                warnings.append(TransformWarning("COLUMN_NOT_FOUND", f"Column '{old}' not found, skipping"))
        return warnings
    
    def execute(self, df: pd.DataFrame) -> TransformResult:
        with timed_execution() as t:
            existing = set(df.columns)
            warnings = []
            valid = {}
            
            for old, new in self.mapping.items():
                if old in existing:
                    valid[old] = new
                elif self.strict:
                    raise ValueError(f"Column not found: '{old}'")
                else:
                    warnings.append(TransformWarning("COLUMN_NOT_FOUND", f"Column '{old}' not found, skipping"))
            
            result = df.rename(columns=valid).copy()
            changes = [column_renamed(df, old, new) for old, new in valid.items()]
        
        return TransformResult(result, make_audit(self.op_name, self.get_params(), df, result, changes, warnings, t["ms"]))


def rename(mapping: dict[str, str], strict: bool = False) -> RenameTransform:
    """Create a rename transform. Example: rename({'old_name': 'new_name'})"""
    return RenameTransform(mapping, strict)
