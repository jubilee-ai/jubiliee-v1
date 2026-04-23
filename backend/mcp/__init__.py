"""Jubilee MCP integration: delegate tasks to the same chat stack as the UI."""

from backend.mcp.jubilee_adapter import run_jubilee_task_sync

__all__ = ["run_jubilee_task_sync"]
