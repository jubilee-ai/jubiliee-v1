"""Normalize multi-turn chat for the planner and format it for prompts."""

from __future__ import annotations

from typing import Any, Optional

MAX_PLANNER_TRANSCRIPT_CHARS = 14_000


def normalize_conversation_turns(
    conversation: Optional[list[dict[str, Any]]],
    *,
    triggering_message: str,
) -> list[dict[str, str]]:
    """Build an ordered user/agent transcript for planning.

    ``triggering_message`` is the latest user instruction (``ChatRequest.message``).
    It is appended only if it is not already the last user turn (avoids duplicates when
    the client sends a full transcript that includes the current message).
    """
    out: list[dict[str, str]] = []
    if conversation:
        for row in conversation:
            role_raw = str(row.get("role") or "").strip().lower()
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            if role_raw in ("assistant", "agent"):
                out.append({"role": "agent", "content": content})
            elif role_raw == "user":
                out.append({"role": "user", "content": content})

    trig = (triggering_message or "").strip()
    if not out:
        return [{"role": "user", "content": trig or "Training run"}]

    last = out[-1]
    if trig and (last["role"] != "user" or last["content"] != trig):
        out.append({"role": "user", "content": trig})
    return out


def transcript_for_planner_prompt(
    turns: list[dict[str, str]],
    *,
    max_chars: int = MAX_PLANNER_TRANSCRIPT_CHARS,
) -> str:
    """Render turns for the planner; keeps the tail if the log exceeds ``max_chars``."""
    lines: list[str] = []
    for t in turns:
        label = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{label}: {t['content']}")
    text = "\n\n".join(lines)
    if len(text) <= max_chars:
        return text
    ellipsis = "\n\n[... earlier dialogue omitted for length ...]\n\n"
    room = max_chars - len(ellipsis)
    if room < 48:
        return text[-max_chars:]
    return ellipsis + text[-room:]
