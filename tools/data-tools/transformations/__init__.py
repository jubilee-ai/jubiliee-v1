"""
Transformations module.

For AI agents, use:
    from transformations import transform_agent_tools
    
For internal automation, use:
    from transformations import transform_apply, select, drop, rename, cast, ...
"""

import sys
from pathlib import Path
from typing import Type

# Add jtypes to path
_root = Path(__file__).parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Types from central location
from jtypes.transformations import (BaseTransform, DatasetInput, SchemaChange,
                                    TransformApplyResult, TransformAudit,
                                    TransformLineage, TransformResult,
                                    TransformWarning)

# Internal transforms (not agent-facing)
from .add_column import SAFE_FUNCS, AddColumnTransform, add_column
# Agent-facing tools (this is what agents should use)
from .agent_tools import transform_agent_tools, transform_dataset
from .base import get_dtype_string, validate_columns_exist
from .cast import DTYPE_MAP, CastTransform, cast
from .drop import DropTransform, drop
from .parse_datetime import ParseDatetimeTransform, parse_datetime
# Recipes
from .recipes import RECIPES, get_recipe, list_recipes
from .rename import RenameTransform, rename
from .select import SelectTransform, select
# Internal orchestrator
from .transform_apply import transform_apply

# Registry (internal)
TRANSFORM_REGISTRY: dict[str, Type[BaseTransform]] = {
    "select": SelectTransform,
    "drop": DropTransform,
    "rename": RenameTransform,
    "cast": CastTransform,
    "parse_datetime": ParseDatetimeTransform,
    "add_column": AddColumnTransform,
}


def get_transform(op_def: dict) -> BaseTransform:
    """Create transform from op definition dict. Internal use."""
    if not isinstance(op_def, dict) or "op" not in op_def:
        raise ValueError("Op definition must be dict with 'op' key")
    
    op_name = op_def["op"]
    if op_name not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown op: '{op_name}'. Available: {list(TRANSFORM_REGISTRY.keys())}")
    
    params = {k: v for k, v in op_def.items() if k != "op"}
    try:
        return TRANSFORM_REGISTRY[op_name](**params)
    except TypeError as e:
        raise ValueError(f"Invalid params for '{op_name}': {e}")


def list_transforms() -> list[str]:
    """List available transforms. Internal use."""
    return list(TRANSFORM_REGISTRY.keys())


__all__ = [
    # === AGENT-FACING (use these with LangChain agents) ===
    "transform_dataset",
    "transform_agent_tools",
    
    # === TYPES ===
    "BaseTransform", "DatasetInput", "TransformResult", "TransformAudit",
    "TransformWarning", "SchemaChange", "TransformApplyResult", "TransformLineage",
    
    # === INTERNAL: Orchestrator ===
    "transform_apply",
    
    # === INTERNAL: Individual transforms ===
    "SelectTransform", "DropTransform", "RenameTransform", "CastTransform",
    "ParseDatetimeTransform", "AddColumnTransform",
    "select", "drop", "rename", "cast", "parse_datetime", "add_column",
    
    # === INTERNAL: Utilities ===
    "get_dtype_string", "validate_columns_exist",
    "TRANSFORM_REGISTRY", "get_transform", "list_transforms",
    "DTYPE_MAP", "SAFE_FUNCS",
    
    # === INTERNAL: Recipes ===
    "RECIPES", "get_recipe", "list_recipes",
]
