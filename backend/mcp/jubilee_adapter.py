"""In-process Jubilee delegation for MCP and other headless clients.

Reuses :func:`backend.chat.service.chat` so routing (resume, train fast-path,
orchestrator + training pivot) stays identical to ``POST /api/chat``.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator, Optional

from pydantic import BaseModel, Field

from backend.chat.schemas import ChatRequest


def _iter_sse_json_payloads(sse_lines: Iterator[str]) -> Iterator[dict[str, Any]]:
    """Parse ``data: {...}\\n\\n`` lines yielded by chat generators."""
    for block in sse_lines:
        if not isinstance(block, str):
            continue
        for line in block.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            raw = line[5:].strip()
            if raw in ("", "[DONE]"):
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


class JubileeTaskOptions(BaseModel):
    """Optional fields mirroring :class:`backend.chat.schemas.ChatRequest` (minus ``message``)."""

    experiment_id: Optional[str] = None
    linked_datasets: Optional[list[str]] = None
    model_preference: Optional[str] = None
    mode: Optional[str] = None
    conversation: Optional[list[dict[str, Any]]] = None
    chat_thread_id: Optional[str] = None
    org_id: Optional[str] = None
    background_intake: bool = False
    resume_training: Optional[dict[str, Any]] = None
    resume: Optional[dict[str, Any]] = None
    force_orchestrator: bool = False


def _summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact MCP-friendly view from parsed SSE payloads."""
    tokens: list[str] = []
    tools: list[dict[str, Any]] = []
    errors: list[str] = []
    training_steps: list[str] = []
    other_types: list[str] = []

    for ev in events:
        et = str(ev.get("type") or "")
        if et == "token":
            c = ev.get("content")
            if isinstance(c, str) and c:
                tokens.append(c)
        elif et == "tool.start":
            tools.append(
                {
                    "phase": "start",
                    "tool": ev.get("tool"),
                    "args": ev.get("args"),
                    "headline": ev.get("headline"),
                }
            )
        elif et == "tool.end":
            res = ev.get("result")
            snippet = res if isinstance(res, str) else json.dumps(res, default=str)[:2000]
            tools.append({"phase": "end", "tool": ev.get("tool"), "result_snippet": snippet})
        elif et == "error":
            msg = ev.get("error") or ev.get("message") or str(ev)
            errors.append(str(msg))
        elif et in ("step.complete", "step.skipped", "stream.start", "stream.end"):
            if et == "step.complete":
                node = ev.get("node")
                headline = ev.get("headline")
                training_steps.append(f"{node}: {headline}" if headline else str(node))
            elif et == "stream.start" and ev.get("training_graph"):
                training_steps.append("training_graph: started")
            elif et == "stream.end":
                training_steps.append(
                    f"stream.end pipeline_completed={ev.get('pipeline_completed')!r}"
                )
        else:
            if et and et not in {"dataset.resolved", "predict.start", "predict.complete"}:
                other_types.append(et)

    assistant_text = "".join(tokens).strip()
    return {
        "assistant_text": assistant_text,
        "tool_trace": tools,
        "errors": errors,
        "training_log": training_steps,
        "other_event_types": sorted(set(other_types)),
    }


def run_jubilee_task_sync(
    *,
    message: str,
    options: Optional[JubileeTaskOptions | dict[str, Any]] = None,
    chat_fn: Optional[Callable[..., Any]] = None,
) -> dict[str, Any]:
    """Run one Jubilee turn via the same stack as ``POST /api/chat``.

    Parameters
    ----------
    message:
        User message (or empty when ``ChatRequest`` validation allows, e.g.
        train fast-path with linked datasets).
    options:
        Optional structured fields; may be a dict or :class:`JubileeTaskOptions`.
    """
    opts = (
        options
        if isinstance(options, JubileeTaskOptions)
        else JubileeTaskOptions.model_validate(options or {})
    )
    org_id = opts.org_id

    req_dict: dict[str, Any] = {
        "message": message,
        "experiment_id": opts.experiment_id,
        "linked_datasets": opts.linked_datasets,
        "model_preference": opts.model_preference,
        "mode": opts.mode,
        "conversation": opts.conversation,
        "chat_thread_id": opts.chat_thread_id,
        "background_intake": opts.background_intake,
        "force_orchestrator": opts.force_orchestrator,
    }
    if opts.resume_training is not None:
        req_dict["resume_training"] = opts.resume_training
    if opts.resume is not None:
        req_dict["resume"] = opts.resume

    request = ChatRequest.model_validate(req_dict)
    fn = chat_fn
    if fn is None:
        # Lazy import so unit tests can inject ``chat_fn`` without importing sklearn.
        from backend.chat.service import chat as _default_chat

        fn = _default_chat

    thread_id, gen = fn(request, org_id=org_id)
    events = list(_iter_sse_json_payloads(gen))
    summary = _summarize_events(events)
    out: dict[str, Any] = {
        "ok": not summary["errors"],
        "thread_id": thread_id,
        "summary": summary,
        "raw_event_count": len(events),
    }
    if summary["errors"]:
        out["error"] = "; ".join(summary["errors"])
    return out


class AssignTaskParams(BaseModel):
    """MCP ``assign_task`` body: primary message plus JSON ``options_json``."""

    message: str = Field(default="", description="User task / question for Jubilee.")
    options_json: str = Field(
        default="{}",
        description=(
            "JSON object with optional keys: experiment_id, linked_datasets, "
            "model_preference, mode, conversation, chat_thread_id, org_id, "
            "background_intake, resume_training, resume, force_orchestrator."
        ),
    )

    def run(self, chat_fn: Optional[Callable[..., Any]] = None) -> dict[str, Any]:
        try:
            raw_opts = json.loads(self.options_json or "{}")
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"Invalid options_json: {e}"}
        if not isinstance(raw_opts, dict):
            return {"ok": False, "error": "options_json must be a JSON object"}
        return run_jubilee_task_sync(message=self.message, options=raw_opts, chat_fn=chat_fn)
