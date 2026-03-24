"""Optional LangGraph custom stream events (live progress / "thinking" lines).

Safe to call from any node: no-ops when not inside a streamed graph run.
"""

from __future__ import annotations

from typing import Any, Mapping


def emit_graph_stream(payload: Mapping[str, Any]) -> None:
    """Send a custom stream chunk if ``stream_mode`` includes ``\"custom\"``."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        if writer is not None:
            writer(dict(payload))
    except RuntimeError:
        pass
