"""Optional LangGraph custom stream events (live progress / "thinking" lines).

Safe to call from any node: no-ops when not inside a streamed graph run.
"""

from __future__ import annotations

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
