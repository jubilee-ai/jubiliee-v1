"""Add column transformation - add computed columns using safe expressions."""

import ast
import operator

import numpy as np
import pandas as pd

from .base import (BaseTransform, SchemaChange, TransformResult,
                   get_dtype_string)
from .utils import (column_added, column_modified, make_audit, timed_execution,
                    validate_required)

# Safe operators
SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.And: lambda a, b: a & b, ast.Or: lambda a, b: a | b, ast.Not: operator.not_,
}

# Safe functions
SAFE_FUNCS = {
    "abs": np.abs, "round": np.round, "floor": np.floor, "ceil": np.ceil,
    "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "exp": np.exp,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "min": np.minimum, "max": np.maximum,
    "lower": lambda s: s.str.lower(), "upper": lambda s: s.str.upper(),
    "strip": lambda s: s.str.strip(), "len": lambda s: s.str.len(),
    "isnull": pd.isna, "notnull": pd.notna, "fillna": lambda s, v: s.fillna(v),
    "coalesce": lambda *a: pd.concat([pd.Series(x) for x in a], axis=1).bfill(axis=1).iloc[:, 0],
    "int": lambda s: pd.to_numeric(s, errors='coerce').astype('Int64'),
    "float": lambda s: pd.to_numeric(s, errors='coerce'),
    "str": lambda s: s.astype(str), "bool": lambda s: s.astype(bool),
    "where": lambda c, x, y: np.where(c, x, y),
    "clip": lambda s, lo, hi: s.clip(lower=lo, upper=hi),
}


class ExpressionEvaluator:
    """Safely evaluate expressions using AST parsing."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = set(df.columns)
    
    def evaluate(self, expr: str):
        return self._eval(ast.parse(expr, mode='eval').body)
    
    def _eval(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        
        if isinstance(node, ast.Name):
            name = node.id
            if name in self.columns:
                return self.df[name]
            if name in SAFE_FUNCS:
                return SAFE_FUNCS[name]
            if name in ('True', 'False', 'None'):
                return {'True': True, 'False': False, 'None': None}[name]
            raise ValueError(f"Unknown: '{name}'. Columns: {list(self.columns)}")
        
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in SAFE_OPS:
                raise ValueError(f"Unsupported op: {op.__name__}")
            return SAFE_OPS[op](self._eval(node.left), self._eval(node.right))
        
        if isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in SAFE_OPS:
                raise ValueError(f"Unsupported op: {op.__name__}")
            return SAFE_OPS[op](self._eval(node.operand))
        
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                op_type = type(op)
                if op_type not in SAFE_OPS:
                    raise ValueError(f"Unsupported comparison: {op_type.__name__}")
                r = SAFE_OPS[op_type](left, self._eval(comp))
                result = r if result is None else (result & r)
                left = self._eval(comp)
            return result
        
        if isinstance(node, ast.BoolOp):
            op = type(node.op)
            if op not in SAFE_OPS:
                raise ValueError(f"Unsupported op: {op.__name__}")
            vals = [self._eval(v) for v in node.values]
            result = vals[0]
            for v in vals[1:]:
                result = SAFE_OPS[op](result, v)
            return result
        
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCS:
                raise ValueError(f"Unknown function: {getattr(node.func, 'id', '?')}")
            args = [self._eval(a) for a in node.args]
            kwargs = {kw.arg: self._eval(kw.value) for kw in node.keywords}
            return SAFE_FUNCS[node.func.id](*args, **kwargs)
        
        if isinstance(node, ast.IfExp):
            return np.where(self._eval(node.test), self._eval(node.body), self._eval(node.orelse))
        
        raise ValueError(f"Unsupported: {type(node).__name__}")


class AddColumnTransform(BaseTransform):
    """Add a computed column using safe expressions."""
    
    op_name = "add_column"
    
    def __init__(self, name: str, expression: str, overwrite: bool = False):
        validate_required(name, "add_column", "column name")
        validate_required(expression, "add_column", "expression")
        try:
            ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}")
        self.name = name
        self.expression = expression
        self.overwrite = overwrite
    
    def get_params(self) -> dict:
        return {"name": self.name, "expression": self.expression, "overwrite": self.overwrite}
    
    def validate(self, df: pd.DataFrame):
        if self.name in df.columns and not self.overwrite:
            raise ValueError(f"Column '{self.name}' exists. Set overwrite=True to replace.")
        return []
    
    def execute(self, df: pd.DataFrame) -> TransformResult:
        self.validate(df)
        
        with timed_execution() as t:
            evaluator = ExpressionEvaluator(df)
            value = evaluator.evaluate(self.expression)
            
            # Ensure proper Series
            if isinstance(value, np.ndarray):
                value = pd.Series(value, index=df.index)
            elif not isinstance(value, pd.Series):
                value = pd.Series([value] * len(df), index=df.index)
            
            result = df.copy()
            is_overwrite = self.name in df.columns
            result[self.name] = value
            
            if is_overwrite:
                changes = [SchemaChange("modified", self.name, 
                    get_dtype_string(df[self.name].dtype), get_dtype_string(result[self.name].dtype))]
            else:
                changes = [column_added(result, self.name)]
        
        return TransformResult(result, make_audit(self.op_name, self.get_params(), df, result, changes, [], t["ms"]))


def add_column(name: str, expression: str, overwrite: bool = False) -> AddColumnTransform:
    """
    Create an add_column transform.
    
    Expression syntax:
      - Columns: just use column name
      - Operators: +, -, *, /, **, ==, !=, <, >, and, or, not
      - Ternary: 'high' if value >= 100 else 'low'
      - Functions: abs, round, sqrt, log, lower, upper, isnull, fillna, where, clip
    
    Examples:
      add_column('total', 'price * quantity')
      add_column('status', "'active' if score >= 80 else 'inactive'")
    """
    return AddColumnTransform(name, expression, overwrite)
