"""In-memory overrides for demo PATCH/POST (single-process; resets on restart)."""

from __future__ import annotations

from typing import Any

_profile_overrides: dict[str, dict[str, Any]] = {}
_approval_edits: dict[str, list[dict[str, Any]]] = {}


def get_profile_overrides(model_name: str) -> dict[str, Any]:
    return dict(_profile_overrides.get(model_name, {}))


def set_profile_overrides(model_name: str, patch: dict[str, Any]) -> None:
    cur = dict(_profile_overrides.get(model_name, {}))
    for k, v in patch.items():
        if v is None:
            cur.pop(k, None)
        else:
            cur[k] = v
    _profile_overrides[model_name] = cur


def get_approval_edits(model_name: str) -> list[dict[str, Any]] | None:
    if model_name not in _approval_edits:
        return None
    return [dict(x) for x in _approval_edits[model_name]]


def set_approval_edits(model_name: str, approvals: list[dict[str, Any]]) -> None:
    _approval_edits[model_name] = [dict(x) for x in approvals]
