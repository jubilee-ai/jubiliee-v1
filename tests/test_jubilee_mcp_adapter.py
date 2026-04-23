"""Tests for Jubilee MCP adapter (delegates to ``backend.chat.service.chat``)."""

from __future__ import annotations

from backend.mcp.jubilee_adapter import AssignTaskParams, run_jubilee_task_sync


def _fake_sse(*lines: str):
    for ln in lines:
        yield ln


def test_run_jubilee_task_sync_parses_tokens_and_tool_events():
    def fake_chat(request, org_id=None):
        assert request.message == "hello"
        gen = _fake_sse(
            'data: {"type":"token","content":"Hello "}\n\n',
            'data: {"type":"token","content":"world"}\n\n',
            'data: {"type":"tool.start","tool":"chart_tool","args":{},"headline":"…"}\n\n',
            'data: {"type":"tool.end","tool":"chart_tool","result":"done"}\n\n',
            'data: {"type":"stream.end","experiment_id":null}\n\n',
        )
        return ("thread-abc", gen)

    out = run_jubilee_task_sync(message="hello", options={"org_id": None}, chat_fn=fake_chat)

    assert out["ok"] is True
    assert out["thread_id"] == "thread-abc"
    assert out["summary"]["assistant_text"] == "Hello world"
    assert any(t.get("tool") == "chart_tool" for t in out["summary"]["tool_trace"])


def test_assign_task_params_options_json_with_injected_chat():
    def fake_chat(request, org_id=None):
        assert org_id == "org_test"
        return ("t1", _fake_sse('data: {"type":"stream.end"}\n\n'))

    p = AssignTaskParams(
        message="ping",
        options_json='{"org_id": "org_test"}',
    )
    out = p.run(chat_fn=fake_chat)

    assert out["ok"] is True
    assert out["thread_id"] == "t1"


def test_assign_task_params_invalid_options_json_no_chat_import():
    p = AssignTaskParams(message="x", options_json="not-json")
    out = p.run()
    assert out["ok"] is False
    assert "Invalid options_json" in out["error"]


def test_effective_thread_id_on_request():
    captured = {}

    def fake_chat(request, org_id=None):
        captured["experiment_id"] = request.experiment_id
        captured["chat_thread_id"] = request.chat_thread_id
        return ("from-chat-fn", _fake_sse('data: {"type":"stream.end"}\n\n'))

    run_jubilee_task_sync(
        message="m",
        options={"experiment_id": "exp-1", "chat_thread_id": "custom-thread-9"},
        chat_fn=fake_chat,
    )

    assert captured["chat_thread_id"] == "custom-thread-9"
    assert captured["experiment_id"] == "exp-1"
