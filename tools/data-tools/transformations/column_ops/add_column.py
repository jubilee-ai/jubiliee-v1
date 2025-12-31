"""Add column tool - add computed column using safe expressions."""

import ast
import operator

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from ..tool_utils import format_result, resolve_dataset, save_result

# Safe operators
SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.And: lambda a, b: a & b, ast.Or: lambda a, b: a | b,
}

# Safe functions
SAFE_FUNCS = {
    "abs": np.abs, "round": np.round, "sqrt": np.sqrt, "log": np.log,
    "min": np.minimum, "max": np.maximum,
    "lower": lambda s: s.str.lower(), "upper": lambda s: s.str.upper(),
    "len": lambda s: s.str.len(), "strip": lambda s: s.str.strip(),
    "isnull": pd.isna, "notnull": pd.notna,
    "fillna": lambda s, v: s.fillna(v),
}


class ExpressionError(ValueError):
    """Custom error for expression evaluation with helpful messages."""
    pass


class ExpressionEvaluator:
    """Safely evaluate expressions using AST parsing."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = set(df.columns)
    
    def evaluate(self, expr: str):
        try:
            parsed = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ExpressionError(f"Syntax error in expression: {e.msg}")
        return self._eval(parsed.body)
    
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
            # Helpful error with suggestions
            similar = [c for c in self.columns if name.lower() in c.lower() or c.lower() in name.lower()]
            if similar:
                raise ExpressionError(f"Unknown column '{name}'. Did you mean: {similar}?")
            raise ExpressionError(f"Unknown column '{name}'. Available: {list(self.columns)[:5]}...")
        
        if isinstance(node, ast.BinOp):
            if type(node.op) not in SAFE_OPS:
                raise ExpressionError(f"Unsupported operator. Use: +, -, *, /, %, **")
            return SAFE_OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in SAFE_OPS:
                raise ExpressionError(f"Unsupported operator. Use: - (negation)")
            return SAFE_OPS[type(node.op)](self._eval(node.operand))
        
        if isinstance(node, ast.Compare):
            left = self._eval(node.left)
            result = None
            for op, comp in zip(node.ops, node.comparators):
                if type(op) not in SAFE_OPS:
                    raise ExpressionError(f"Unsupported comparison. Use: ==, !=, <, >, <=, >=")
                r = SAFE_OPS[type(op)](left, self._eval(comp))
                result = r if result is None else (result & r)
                left = self._eval(comp)
            return result
        
        if isinstance(node, ast.BoolOp):
            if type(node.op) not in SAFE_OPS:
                raise ExpressionError(f"Unsupported boolean. Use: and, or")
            vals = [self._eval(v) for v in node.values]
            result = vals[0]
            for v in vals[1:]:
                result = SAFE_OPS[type(node.op)](result, v)
            return result
        
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExpressionError(f"Only simple function calls allowed")
            if node.func.id not in SAFE_FUNCS:
                raise ExpressionError(f"Unknown function '{node.func.id}'. Available: {list(SAFE_FUNCS.keys())}")
            args = [self._eval(a) for a in node.args]
            return SAFE_FUNCS[node.func.id](*args)
        
        if isinstance(node, ast.IfExp):
            return np.where(self._eval(node.test), self._eval(node.body), self._eval(node.orelse))
        
        raise ExpressionError(f"Unsupported syntax. Use: columns, operators (+,-,*,/), comparisons, 'x' if cond else 'y'")


class AddColumnInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    name: str = Field(description="Name for the new column")
    expression: str = Field(
        description="Expression using column names. Examples: 'price * quantity', "
                    "'amount > 1000', \"'high' if value > 100 else 'low'\""
    )


@tool(args_schema=AddColumnInput)
def add_column_tool(dataset_ref: str, name: str, expression: str) -> str:
    """Add a computed column using an expression with column names."""
    try:
        df = resolve_dataset(dataset_ref)
        
        evaluator = ExpressionEvaluator(df)
        value = evaluator.evaluate(expression)
        
        # Ensure proper Series
        if isinstance(value, np.ndarray):
            value = pd.Series(value, index=df.index)
        elif not isinstance(value, pd.Series):
            value = pd.Series([value] * len(df), index=df.index)
        
        result = df.copy()
        result[name] = value
        
        ref = save_result(result, "add")
        return format_result(ref, result, "add_column", f"{name} = {expression}")
        
    except ExpressionError as e:
        return f"✗ add_column failed: {e}"
    except Exception as e:
        return f"✗ add_column failed: {type(e).__name__}: {e}"

