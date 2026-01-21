"""
Apply multiple transformations in sequence by dynamically calling tools.

Takes a list of tools and their params, executes them in order,
parses the output to extract the new dataset_ref, and chains them.
"""

import re
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from .tool_utils import resolve_dataset, save_result

# =============================================================================
# TOOL REGISTRY - Dynamically populated from submodules
# =============================================================================

_TOOL_REGISTRY: dict[str, BaseTool] = {}


def _populate_tool_registry():
    """Dynamically populate registry with all tools from submodules."""
    from .agg_ops import agg_tools
    from .clean_ops import clean_tools
    from .column_ops import column_tools
    from .feature_ops import feature_tools
    from .reshape_ops import reshape_tools
    from .row_ops import row_tools
    from .time_ops import time_tools
    
    all_tools = clean_tools + column_tools + row_tools + feature_tools + time_tools + agg_tools + reshape_tools
    
    for t in all_tools:
        name = getattr(t, "name", None) or getattr(t, "__name__", str(t))
        _TOOL_REGISTRY[name] = t
        # Also register without _tool suffix
        if name.endswith("_tool"):
            _TOOL_REGISTRY[name[:-5]] = t


def get_tool_descriptions() -> str:
    """Get formatted descriptions of all available tools."""
    lines = []
    seen = set()
    for name, t in _TOOL_REGISTRY.items():
        if name.endswith("_tool") or name in seen:
            continue
        seen.add(name)
        # Get first line of docstring
        doc = getattr(t, "description", "") or getattr(t.func, "__doc__", "") or ""
        first_line = doc.split("\n")[0].strip() if doc else ""
        lines.append(f"- {name}: {first_line}")
    return "\n".join(sorted(lines))


def get_registered_tools() -> list[str]:
    """Get list of registered tool names (without _tool suffix)."""
    return sorted([n for n in _TOOL_REGISTRY.keys() if not n.endswith("_tool")])


# Populate on module load
_populate_tool_registry()


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def _extract_dataset_ref(output: str) -> str | None:
    """Extract dataset_ref from tool output."""
    # Pattern: → `ref`
    match = re.search(r'→\s*[`\'"]?(\w+)[`\'"]?', output)
    if match:
        return match.group(1)
    # Pattern: dataset: `ref`
    match = re.search(r'dataset:\s*[`\'"]?(\w+)[`\'"]?', output, re.IGNORECASE)
    if match:
        return match.group(1)
    # Pattern: ref-like (prefix_hash_hash)
    match = re.search(r'\b([a-z]+_[a-f0-9]+_[a-f0-9]+)\b', output)
    if match:
        return match.group(1)
    return None


def apply_transformations(
    dataset_ref: str,
    transformations: list[dict[str, Any]],
    output_prefix: str = "transformed",
) -> dict[str, Any]:
    """
    Apply multiple transformations in sequence.
    
    Args:
        dataset_ref: Initial dataset reference
        transformations: List of {"tool": <tool>, "params": {...}}
        output_prefix: Prefix for final dataset ref
    
    Returns:
        {"dataset_ref": str, "outputs": list, "errors": list|None}
    """
    current_ref = dataset_ref
    outputs = []
    errors = []
    
    for i, transform in enumerate(transformations):
        tool_fn = transform.get("tool")
        params = transform.get("params", {}).copy()
        params["dataset_ref"] = current_ref
        
        if tool_fn is None:
            errors.append(f"Step {i+1}: Missing 'tool'")
            continue
        
        try:
            if isinstance(tool_fn, BaseTool):
                output = tool_fn.invoke(params)
            elif callable(tool_fn):
                output = tool_fn(**params)
            else:
                errors.append(f"Step {i+1}: Invalid tool type")
                continue
            
            outputs.append({"step": i + 1, "tool": getattr(tool_fn, "name", str(tool_fn)), "output": output})
            
            if isinstance(output, str) and output.startswith("✗"):
                errors.append(f"Step {i+1}: {output}")
                continue
            
            new_ref = _extract_dataset_ref(str(output))
            if new_ref:
                current_ref = new_ref
                
        except Exception as e:
            errors.append(f"Step {i+1}: {e}")
    
    # Re-register final result
    try:
        df = resolve_dataset(current_ref)
        final_ref = save_result(df, output_prefix)
    except Exception:
        final_ref = current_ref
    
    return {"dataset_ref": final_ref, "outputs": outputs, "errors": errors if errors else None}


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class ApplyTransformationsInput(BaseModel):
    """Input schema."""
    dataset_ref: str = Field(description="Dataset reference to transform")
    transformations: list[dict] = Field(
        default=[],
        description="List of transformations. Each dict has 'tool_name' and tool parameters. "
                    "Example: {'tool_name': 'impute', 'column': 'age', 'strategy': 'median'}"
    )
    output_prefix: str = Field(default="transformed", description="Prefix for output dataset ref")


# Build dynamic description
_TOOL_DESCRIPTION = f"""Apply one or more data transformations in sequence.

Pass a list of transformations. Each transformation is a dict with 'tool_name' and the tool's parameters.

AVAILABLE TOOLS:
{get_tool_descriptions()}

FORMAT: {{"tool_name": "<name>", "<param1>": <value1>, "<param2>": <value2>}}

EXAMPLES:
- Impute nulls: {{"tool_name": "impute", "column": "age", "strategy": "median"}}
- Cap outliers: {{"tool_name": "clip", "column": "income", "min_val": 0, "max_val": 500000}}
- Drop nulls: {{"tool_name": "drop_nulls", "subset": ["income"]}}
- Drop columns: {{"tool_name": "drop_columns", "columns": ["id"]}}
- Fill null: {{"tool_name": "fill_null", "column": "status", "value": "unknown"}}
- Dedupe: {{"tool_name": "dedupe"}}
"""


@tool(args_schema=ApplyTransformationsInput, description=_TOOL_DESCRIPTION)
def apply_transformations_tool(
    dataset_ref: str,
    transformations: list[dict] = None,
    output_prefix: str = "transformed",
) -> str:
    # Docstring set dynamically below
    if not transformations:
        return (
            "✗ No transformations provided. Pass a list of transformations like:\n"
            "  [{'tool_name': 'impute', 'column': 'age', 'strategy': 'median'}]\n"
            f"Available tools: {get_registered_tools()[:10]}..."
        )
    
    tool_transforms = []
    for spec in transformations:
        # Extract tool_name and treat everything else as params
        name = spec.get("tool_name")
        if not name:
            return f"✗ Missing 'tool_name' in transformation: {spec}"
        
        # Get params - either from nested 'params' key or from flat structure
        if "params" in spec:
            params = spec["params"]
        else:
            # Flat structure: everything except tool_name is a param
            params = {k: v for k, v in spec.items() if k != "tool_name"}
        
        if name not in _TOOL_REGISTRY:
            if name + "_tool" in _TOOL_REGISTRY:
                name = name + "_tool"
            else:
                return f"✗ Unknown tool '{name}'. Available: {get_registered_tools()}"
        tool_transforms.append({"tool": _TOOL_REGISTRY[name], "params": params})
    
    result = apply_transformations(dataset_ref, tool_transforms, output_prefix)
    
    # Format output
    lines = [f"✓ Applied {len(result['outputs'])}/{len(transformations)} → `{result['dataset_ref']}`"]
    for out in result["outputs"]:
        status = "✓" if not any(str(out["step"]) in e for e in (result.get("errors") or [])) else "✗"
        lines.append(f"  {out['step']}. {status} {out['tool']}")
    if result.get("errors"):
        for err in result["errors"]:
            lines.append(f"  ⚠ {err}")
    return "\n".join(lines)


__all__ = [
    "apply_transformations",
    "apply_transformations_tool", 
    "get_registered_tools",
    "get_tool_descriptions",
]
