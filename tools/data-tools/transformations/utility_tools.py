"""Utility tools for dataset management."""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .tool_utils import cleanup_datasets

try:
    from ..utils import get_registered_dataset_info, list_registered_datasets
except ImportError:
    from utils import get_registered_dataset_info, list_registered_datasets


class CleanupInput(BaseModel):
    prefix: Optional[str] = Field(
        default=None, 
        description="Only cleanup datasets starting with this prefix (e.g., 'flt_', 'srt_'). "
                    "If not provided, clears ALL datasets."
    )


@tool(args_schema=CleanupInput)
def cleanup_datasets_tool(prefix: Optional[str] = None) -> str:
    """
    Clear registered datasets from memory to free up space.
    Use after completing a pipeline or when starting fresh.
    """
    try:
        count = cleanup_datasets(prefix)
        if prefix:
            return f"✓ Cleared {count} datasets with prefix '{prefix}'"
        return f"✓ Cleared all {count} datasets from memory"
    except Exception as e:
        return f"✗ cleanup failed: {e}"


@tool
def list_datasets_tool() -> str:
    """List all currently registered datasets and their sizes."""
    try:
        refs = list_registered_datasets()
        if not refs:
            return "No datasets currently registered"
        
        lines = [f"Registered datasets ({len(refs)}):"]
        for ref in refs[:20]:  # Limit to 20
            info = get_registered_dataset_info(ref)
            if info:
                lines.append(f"  • `{ref}`: {info['rows']:,} rows × {info['columns']} cols")
            else:
                lines.append(f"  • `{ref}`: (info unavailable)")
        
        if len(refs) > 20:
            lines.append(f"  ... and {len(refs) - 20} more")
        
        return "\n".join(lines)
    except Exception as e:
        return f"✗ list failed: {e}"


utility_tools = [cleanup_datasets_tool, list_datasets_tool]

__all__ = ["cleanup_datasets_tool", "list_datasets_tool", "utility_tools"]

