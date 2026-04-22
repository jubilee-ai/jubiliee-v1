"""Unit tests for unified training SSE generators (mocked agent.stream, no LLM)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from backend.training import service


def _collect_sse_payloads(gen) -> list[dict]:
    payloads = []
    for raw in gen:
        if isinstance(raw, str) and raw.startswith("data: "):
            payloads.append(json.loads(raw[6:].strip()))
    return payloads


@pytest.fixture
def in_memory_repo():
    """Isolate repository store for one test."""
    stores: dict[str, dict] = {}

    def put_store(tid: str, val: dict) -> None:
        stores[tid] = val

    def get_store(tid: str):
        return stores.get(tid)

    with (
        patch.object(service.repository, "put_simple_agent_store", side_effect=put_store),
        patch.object(service.repository, "get_simple_agent_store", side_effect=get_store),
        patch.object(service.repository, "save_training_context", MagicMock()),
        patch.object(service.repository, "save_run_dataset_links", MagicMock()),
        patch.object(service.repository, "save_interrupt_ids", MagicMock()),
    ):
        yield stores


def _make_tool_update(tool_name: str, content: str = "ok"):
    return {
        "tools": {
            "messages": [
                SimpleNamespace(name=tool_name, content=content),
            ]
        }
    }


def test_graph_sse_tool_emits_step_complete(in_memory_repo):
    thread = "graph-test-thread"
    shared = {"goal": "test goal", "collected_dataset_ref": "ds_ref"}

    def fake_stream(_input, config=None, stream_mode=None, subgraphs=None):
        yield (None, _make_tool_update("tool_data_collection"))

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(
        values={"goal": "test goal", "collected_dataset_ref": "ds_ref"}
    )

    with (
        patch(
            "agents.training.agent_simple.create_simple_training_agent",
            return_value=(mock_agent, shared),
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch.object(service, "load_and_register_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events(
                "test goal", None, None, thread_id=thread
            )
        )

    assert not any(p.get("type") == "node_update" for p in payloads)
    completes = [p for p in payloads if p.get("type") == "step.complete"]
    dc = next((p for p in completes if p.get("node") == "data_collection"), None)
    assert dc is not None
    assert dc.get("thread_id") == thread
    assert dc.get("stream_step_key") == "1:data_collection"

    service.repository.save_training_context.assert_called_once()
    service.repository.save_run_dataset_links.assert_called_once_with(thread, shared)


def test_graph_sse_interrupt_includes_state_snapshot(in_memory_repo):
    thread = "graph-interrupt-thread"

    class MockInterrupt:
        id = "int-1"
        value = {
            "node": "data_collection",
            "summary": "Review data",
            "message": "Approve?",
        }

    def fake_stream(_input, config=None, stream_mode=None, subgraphs=None):
        yield (None, {"__interrupt__": [MockInterrupt()]})

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(
        values={"goal": "g2", "collected_dataset_ref": "r1"}
    )

    shared = {"goal": "g2"}

    with (
        patch(
            "agents.training.agent_simple.create_simple_training_agent",
            return_value=(mock_agent, shared),
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch.object(service, "load_and_register_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events("g2", None, None, thread_id=thread)
        )

    intr = next(p for p in payloads if p.get("type") == "review.required")
    assert "state_snapshot" in intr
    assert intr["state_snapshot"].get("goal") == "g2"

    service.repository.save_training_context.assert_not_called()
    service.repository.save_run_dataset_links.assert_not_called()

    assert not any(p.get("type") == "completed" for p in payloads)


def test_graph_sse_same_step_name_twice_distinct_stream_keys(in_memory_repo):
    """Two tool calls for the same logical step emit distinct stream_step_key values."""
    thread = "graph-dup-thread"
    shared = {"goal": "g-dup"}

    def fake_stream(_input, config=None, stream_mode=None, subgraphs=None):
        yield (None, _make_tool_update("tool_data_collection", "ok a"))
        yield (None, _make_tool_update("tool_data_collection", "ok b"))

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(values={"goal": "g-dup"})

    with (
        patch(
            "agents.training.agent_simple.create_simple_training_agent",
            return_value=(mock_agent, shared),
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch.object(service, "load_and_register_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events("g-dup", None, None, thread_id=thread)
        )

    completes = [
        p
        for p in payloads
        if p.get("type") == "step.complete" and p.get("node") == "data_collection"
    ]
    assert len(completes) == 2
    keys = {p.get("stream_step_key") for p in completes}
    assert keys == {"1:data_collection", "2:data_collection"}
