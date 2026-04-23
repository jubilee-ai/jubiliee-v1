"""Stdio MCP server exposing a single ``assign_task`` tool for Jubilee.

Run from the repository root (requires ``mcp`` package)::

    python -m backend.mcp.server

Use stderr for logging; stdout is reserved for the MCP JSON-RPC stream.
"""

from __future__ import annotations

import json
import os
import sys


def _bootstrap_runtime() -> None:
    """Match backend startup: paths, settings, MLflow, DB (optional skip for tests)."""
    from backend.shared.mlflow_setup import bootstrap_mlflow
    from backend.shared.settings import bootstrap_paths, get_settings

    bootstrap_paths()
    get_settings()
    bootstrap_mlflow()
    if os.getenv("JUBILEE_MCP_SKIP_DB", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    try:
        from backend.shared.database import init_db

        init_db()
    except Exception as exc:  # pragma: no cover - startup diagnostic
        print(f"[jubilee-mcp] init_db skipped or failed: {exc}", file=sys.stderr)


def main() -> None:
    _bootstrap_runtime()

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover
        print(
            "Missing dependency: install the official MCP SDK, e.g. `pip install mcp` "
            "(see requirements.txt).",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    from backend.mcp.jubilee_adapter import AssignTaskParams

    mcp = FastMCP("Jubilee")

    @mcp.tool()
    def assign_task(message: str = "", options_json: str = "{}") -> str:
        """Assign a task to Jubilee (same orchestrator + tools as in-app chat).

        ``options_json`` is a JSON object with optional keys: ``experiment_id``,
        ``linked_datasets``, ``model_preference``, ``mode``, ``conversation``,
        ``chat_thread_id``, ``org_id``, ``background_intake``, ``resume_training``,
        ``resume``, ``force_orchestrator``. See :class:`backend.chat.schemas.ChatRequest`.
        """
        payload = AssignTaskParams(message=message, options_json=options_json).run()
        return json.dumps(payload, ensure_ascii=False, default=str)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
