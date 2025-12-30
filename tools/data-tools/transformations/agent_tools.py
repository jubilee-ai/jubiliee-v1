"""High-level transformation tools for AI agents."""

import json
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Add paths
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

try:
    from ..utils import build_schema_from_dataframe, get_registered_dataset
except ImportError:
    from utils import get_registered_dataset, build_schema_from_dataframe

from .recipes import RECIPES, get_recipe, list_recipes
from .transform_apply import transform_apply


def _resolve_ops(
    df: pd.DataFrame,
    recipe: Optional[str] = None,
    recipe_config: Optional[dict] = None,
    ops: Optional[list[dict]] = None,
    keep_columns: Optional[list[str]] = None,
    drop_columns: Optional[list[str]] = None,
    rename_columns: Optional[dict[str, str]] = None,
    cast_types: Optional[dict[str, str]] = None,
    parse_dates: Optional[list[str]] = None,
    add_features: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Resolve all inputs into a list of ops."""
    result_ops = []
    
    # Apply recipe first
    if recipe:
        recipe_fn = get_recipe(recipe)
        result_ops.extend(recipe_fn(df, recipe_config or {}))
    
    # Add explicit ops
    if ops:
        result_ops.extend(ops)
    
    # Declarative shortcuts
    if keep_columns:
        result_ops.append({"op": "select", "columns": keep_columns})
    
    if drop_columns:
        result_ops.append({"op": "drop", "columns": drop_columns})
    
    if rename_columns:
        result_ops.append({"op": "rename", "mapping": rename_columns})
    
    if cast_types:
        for col, dtype in cast_types.items():
            result_ops.append({"op": "cast", "column": col, "dtype": dtype})
    
    if parse_dates:
        for col in parse_dates:
            result_ops.append({"op": "parse_datetime", "column": col})
    
    if add_features:
        for name, expr in add_features.items():
            result_ops.append({"op": "add_column", "name": name, "expression": expr})
    
    return result_ops


# =============================================================================
# AGENT-FACING TOOL
# =============================================================================

class TransformDatasetInput(BaseModel):
    """Input for transform_dataset tool."""
    
    dataset_ref: str = Field(
        description="Dataset to transform. Use a dataset_ref from a previous operation (e.g., 'ds_abc123') or a catalog asset_id."
    )
    
    recipe: Optional[str] = Field(
        default=None,
        description="Predefined transformation recipe. Options: "
                    "'clean_for_ml' (drop nulls, parse dates, cast types), "
                    "'clean_basic' (parse dates), "
                    "'prepare_for_join' (normalize keys), "
                    "'normalize_names' (snake_case columns). "
                    "Can combine with other options."
    )
    
    keep_columns: Optional[list[str]] = Field(
        default=None,
        description="Keep only these columns (drop all others). Example: ['id', 'name', 'amount']"
    )
    
    drop_columns: Optional[list[str]] = Field(
        default=None,
        description="Drop these columns. Example: ['temp_col', 'debug_info']"
    )
    
    rename_columns: Optional[dict[str, str]] = Field(
        default=None,
        description="Rename columns. Example: {'old_name': 'new_name', 'user_id': 'id'}"
    )
    
    cast_types: Optional[dict[str, str]] = Field(
        default=None,
        description="Cast column types. Types: integer, float, string, boolean, category. "
                    "Example: {'age': 'integer', 'price': 'float'}"
    )
    
    parse_dates: Optional[list[str]] = Field(
        default=None,
        description="Parse these columns as datetime. Example: ['created_at', 'updated_at']"
    )
    
    add_features: Optional[dict[str, str]] = Field(
        default=None,
        description="Add computed columns. Use column names in expressions. "
                    "Supports: +, -, *, /, comparisons, 'x' if cond else 'y'. "
                    "Example: {'total': 'price * quantity', 'is_large': 'amount > 1000'}"
    )
    
    dry_run: Optional[bool] = Field(
        default=False,
        description="If true, validate without executing. Use to check if transformations will work."
    )


@tool(args_schema=TransformDatasetInput)
def transform_dataset(
    dataset_ref: str,
    recipe: Optional[str] = None,
    keep_columns: Optional[list[str]] = None,
    drop_columns: Optional[list[str]] = None,
    rename_columns: Optional[dict[str, str]] = None,
    cast_types: Optional[dict[str, str]] = None,
    parse_dates: Optional[list[str]] = None,
    add_features: Optional[dict[str, str]] = None,
    dry_run: bool = False,
) -> str:
    """
    Transform a dataset with cleaning, type casting, and feature engineering.
    
    Use this tool to prepare data for analysis or modeling. You can:
    - Apply a predefined recipe (clean_for_ml, clean_basic, etc.)
    - Select/drop columns
    - Rename columns  
    - Cast data types
    - Parse date strings
    - Add computed features
    
    Returns a new dataset_ref that you can use in subsequent operations.
    
    Examples:
    - Clean for ML: transform_dataset(dataset_ref="data", recipe="clean_for_ml")
    - Select columns: transform_dataset(dataset_ref="data", keep_columns=["id", "amount"])
    - Add feature: transform_dataset(dataset_ref="data", add_features={"is_high": "value > 100"})
    """
    try:
        # Get DataFrame to inspect for recipe
        df = get_registered_dataset(dataset_ref)
        if df is None:
            # Try to load it
            try:
                from ..data_loader import dataset_get
            except ImportError:
                from data_loader import dataset_get
            result = dataset_get(asset_id=dataset_ref, limit=-1, include_stats=False)
            df = pd.DataFrame(result.data)
        
        # Resolve all inputs to ops
        ops = _resolve_ops(
            df=df,
            recipe=recipe,
            keep_columns=keep_columns,
            drop_columns=drop_columns,
            rename_columns=rename_columns,
            cast_types=cast_types,
            parse_dates=parse_dates,
            add_features=add_features,
        )
        
        if not ops:
            return "No transformations specified. Provide a recipe or transformation options."
        
        # Execute
        result = transform_apply(dataset_ref, ops, dry_run=dry_run)
        
        # Format response
        lines = []
        status = "✓ SUCCESS" if result.success else "✗ FAILED"
        mode = " [DRY RUN]" if dry_run else ""
        lines.append(f"## Transform {status}{mode}")
        lines.append("")
        
        if result.success and not dry_run:
            lines.append(f"**Result:** `{result.dataset_ref}` (use this in next operation)")
            lines.append("")
        
        lines.append(f"- Rows: {result.final_rows:,}")
        lines.append(f"- Columns: {len(result.final_schema)}")
        lines.append(f"- Operations: {result.ops_applied}")
        lines.append(f"- Time: {result.execution_time_ms:.1f}ms")
        
        if result.warnings_total > 0:
            lines.append(f"- Warnings: {result.warnings_total}")
        
        if result.error:
            lines.append(f"\n**Error:** {result.error}")
        
        # Schema summary
        lines.append("\n**Schema:**")
        for col in result.final_schema[:10]:
            lines.append(f"- {col['name']}: {col['type']}")
        if len(result.final_schema) > 10:
            lines.append(f"- ... +{len(result.final_schema) - 10} more")
        
        # Operations applied
        if result.audits:
            lines.append("\n**Operations applied:**")
            for i, a in enumerate(result.audits, 1):
                status_icon = "✓" if a.success else "✗"
                lines.append(f"{i}. {status_icon} {a.op_name}")
                for w in a.warnings:
                    lines.append(f"   ⚠ {w.message}")
                if a.error:
                    lines.append(f"   ✗ {a.error}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


# Single tool list for agent
transform_agent_tools = [transform_dataset]

