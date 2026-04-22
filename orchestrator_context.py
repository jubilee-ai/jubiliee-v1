"""Per-request context for the main Jubilee orchestrator (chat SSE scope).

Uses ``threading.local`` instead of ``ContextVar`` because Starlette streams
sync generators in a worker thread — ``ContextVar.reset(token)`` fails across
the thread/context boundary.
"""

import threading

_local = threading.local()


def attached_datasets_active() -> bool:
    return getattr(_local, "attached", False)


def set_attached_datasets_active(val: bool) -> None:
    _local.attached = bool(val)


def reset_attached_datasets_active() -> None:
    _local.attached = False
