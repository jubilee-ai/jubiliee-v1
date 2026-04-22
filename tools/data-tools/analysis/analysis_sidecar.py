"""Attach structured JSON sidecars to analysis tool markdown output for SSE/UI."""

from __future__ import annotations

import json
import re
from typing import Any

_ANALYSIS_JSON_RE = re.compile(
    r"<ANALYSIS_JSON>(.*?)</ANALYSIS_JSON>",
    re.DOTALL,
)


def attach_analysis_sidecar(
    markdown_body: str,
    *,
    kind: str,
    tool: str,
    summary: str,
    payload: dict[str, Any],
) -> str:
    """Append a parseable JSON blob after the human-readable markdown."""
    envelope = {
        "kind": kind,
        "tool": tool,
        "summary": summary,
        "payload": payload,
    }
    raw = json.dumps(envelope, ensure_ascii=False, default=str)
    text = (markdown_body or "").rstrip()
    return f"{text}\n\n<ANALYSIS_JSON>{raw}</ANALYSIS_JSON>"


def extract_analysis_json_sidecars(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Strip all sidecar blocks from *text* and return cleaned markdown + parsed envelopes."""
    if not text or "<ANALYSIS_JSON>" not in text:
        return text, []

    envelopes: list[dict[str, Any]] = []
    cleaned = text

    for m in _ANALYSIS_JSON_RE.finditer(text):
        raw_json = m.group(1).strip()
        try:
            data = json.loads(raw_json)
            if isinstance(data, dict):
                envelopes.append(data)
        except json.JSONDecodeError:
            continue
        cleaned = cleaned.replace(m.group(0), "").strip()

    return cleaned, envelopes
