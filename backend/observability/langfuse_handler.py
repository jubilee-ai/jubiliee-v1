"""Langfuse LLM telemetry — shared callback handler factory.

All LLM calls in the agent graph should use `get_langfuse_handler()` so that
token counts, latency, cost, and prompt info are tracked automatically.  Traces
are tagged with experiment_id + node so the proxy endpoint can query them.

If Langfuse env vars are not configured, returns None (callers should skip).
"""

from __future__ import annotations

from typing import Optional

from backend.shared.settings import get_settings


def get_langfuse_handler(
    experiment_id: str | None = None,
    node: str | None = None,
    user_id: str | None = None,
) -> Optional[object]:
    """Return a Langfuse CallbackHandler tagged with metadata, or None if unconfigured."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None

    try:
        from langfuse.callback import CallbackHandler

        metadata = {}
        if experiment_id:
            metadata["experiment_id"] = experiment_id
        if node:
            metadata["node"] = node

        return CallbackHandler(
            secret_key=settings.langfuse_secret_key,
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            host=settings.LANGFUSE_HOST,
            session_id=experiment_id,
            user_id=user_id,
            metadata=metadata,
            tags=[node] if node else [],
        )
    except Exception:
        return None
