"""Optional LangGraph custom stream events (live progress / "thinking" lines).

Safe to call from any node: no-ops when not inside a streamed graph run.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Mapping

from langchain_core.callbacks import BaseCallbackHandler


def emit_graph_stream(payload: Mapping[str, Any]) -> None:
    """Send a custom stream chunk if ``stream_mode`` includes ``"custom"``."""
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        if writer is not None:
            writer(dict(payload))
    except RuntimeError:
        pass


# ---------------------------------------------------------------------------
# Step timing context — tracks wall-clock per graph node
# ---------------------------------------------------------------------------

_step_timing: threading.local = threading.local()


def step_timer_start(node: str) -> None:
    """Mark the start of a pipeline step for duration tracking."""
    _step_timing.current_node = node
    _step_timing.started_at = datetime.now(timezone.utc)
    _step_timing.start_mono = time.monotonic()


def step_timer_finish() -> dict[str, Any]:
    """Return timing data for the current step and reset."""
    started = getattr(_step_timing, "started_at", None)
    start_mono = getattr(_step_timing, "start_mono", None)
    node = getattr(_step_timing, "current_node", None)
    if started is None or start_mono is None:
        return {}
    elapsed_ms = int((time.monotonic() - start_mono) * 1000)
    result = {
        "node": node,
        "started_at": started.isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": elapsed_ms,
    }
    _step_timing.started_at = None
    _step_timing.start_mono = None
    _step_timing.current_node = None
    return result


class GraphTokenStreamHandler(BaseCallbackHandler):
    """LangChain callback that relays LLM tokens as custom graph stream events.

    Attach to any ``structured_llm.invoke(prompt, config={"callbacks": [handler]})``
    call inside a graph node.  Each token is emitted via ``emit_graph_stream``
    so that ``agent.stream(stream_mode=["updates","custom"])`` picks it up and
    the SSE layer can forward it as a ``token`` event.
    """

    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            emit_graph_stream({
                "type": "token",
                "content": token,
                "phase": self.phase,
            })
