"""Unit tests for graph-mode SSE generators (mocked agent.stream, no LLM)."""

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


def test_graph_sse_step_node_emits_node_complete_not_node_update(in_memory_repo):
    thread = "graph-test-thread"

    def fake_stream(_input, config=None, stream_mode=None):
        yield (
            "updates",
            {
                "data_collection": {
                    "collected_dataset_ref": "ds_ref",
                    "plan_history": [],
                    "plan_index": 0,
                    "audit_trace": [{"step": "data_collection", "rows": 10, "columns": ["a"]}],
                }
            },
        )

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(
        values={"goal": "test goal", "collected_dataset_ref": "ds_ref"}
    )

    with (
        patch(
            "agents.training.core.graph.create_training_agent",
            return_value=mock_agent,
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch(
            "agents.training.core.state.create_initial_state",
            return_value={"goal": "test goal"},
        ),
        patch.object(service, "load_and_register_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events(
                "test goal", None, None, thread_id=thread
            )
        )

    assert not any(p.get("type") == "node_update" for p in payloads)
    completes = [p for p in payloads if p.get("type") == "node_complete"]
    dc = next((p for p in completes if p.get("node") == "data_collection"), None)
    assert dc is not None
    assert dc.get("thread_id") == thread
    assert dc.get("stream_step_key") == "0:0:data_collection"

    service.repository.save_training_context.assert_called_once()
    service.repository.save_run_dataset_links.assert_called_once_with(
        thread,
        {"goal": "test goal", "collected_dataset_ref": "ds_ref"},
    )


def test_graph_sse_interrupt_includes_state_snapshot(in_memory_repo):
    thread = "graph-interrupt-thread"

    class MockInterrupt:
        id = "int-1"
        value = {
            "node": "planner",
            "summary": "Review plan",
            "message": "Approve?",
        }

    def fake_stream(_input, config=None, stream_mode=None):
        yield ("updates", {"__interrupt__": [MockInterrupt()]})

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(
        values={"goal": "g2", "plan": [], "plan_index": 0}
    )

    with (
        patch(
            "agents.training.core.graph.create_training_agent",
            return_value=mock_agent,
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch(
            "agents.training.core.state.create_initial_state",
            return_value={"goal": "g2"},
        ),
        patch.object(service, "load_and_register_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events("g2", None, None, thread_id=thread)
        )

    intr = next(p for p in payloads if p.get("type") == "interrupt")
    assert "state_snapshot" in intr
    assert intr["state_snapshot"].get("goal") == "g2"

    service.repository.save_training_context.assert_not_called()
    service.repository.save_run_dataset_links.assert_not_called()

    assert not any(p.get("type") == "completed" for p in payloads)


def test_graph_sse_planner_emits_node_skipped_deduped(in_memory_repo):
    thread = "graph-skip-thread"

    planner_out = {
        "plan": [{"step": "data_collection", "rationale": "r"}],
        "skipped_steps": ["cleaning", "label_split_definition"],
        "skip_rationale": "already clean",
        "audit_trace": [],
        "current_step": "planner",
    }

    def fake_stream(_input, config=None, stream_mode=None):
        yield ("updates", {"planner": planner_out})
        yield ("updates", {"planner": planner_out})

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(values={"goal": "g3"})

    with (
        patch(
            "agents.training.core.graph.create_training_agent",
            return_value=mock_agent,
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch(
            "agents.training.core.state.create_initial_state",
            return_value={"goal": "g3"},
        ),
        patch.object(service, "load_and_register_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events("g3", None, None, thread_id=thread)
        )

    skipped = [p for p in payloads if p.get("type") == "node_skipped"]
    assert len(skipped) == 2
    nodes = {p["node"] for p in skipped}
    assert nodes == {"cleaning", "label_split_definition"}
    assert all(p.get("reason") == "already clean" for p in skipped)
    assert all(p.get("thread_id") == thread for p in skipped)


def test_graph_sse_same_step_name_twice_distinct_stream_keys(in_memory_repo):
    """After replan, data_collection may run again — must emit a second node_complete."""
    thread = "graph-dup-thread"

    dc_state_a = {
        "collected_dataset_ref": "ds_a",
        "plan_history": [],
        "plan_index": 0,
        "audit_trace": [{"step": "data_collection", "rows": 1, "columns": ["x"]}],
    }
    dc_state_b = {
        "collected_dataset_ref": "ds_b",
        "plan_history": [{"steps": []}],
        "plan_index": 0,
        "audit_trace": [{"step": "data_collection", "rows": 2, "columns": ["y"]}],
    }

    def fake_stream(_input, config=None, stream_mode=None):
        yield ("updates", {"data_collection": dc_state_a})
        yield ("updates", {"data_collection": dc_state_b})

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(values={"goal": "g-dup"})

    with (
        patch(
            "agents.training.core.graph.create_training_agent",
            return_value=mock_agent,
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch(
            "agents.training.core.state.create_initial_state",
            return_value={"goal": "g-dup"},
        ),
        patch.object(service, "load_and_register_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events("g-dup", None, None, thread_id=thread)
        )

    completes = [p for p in payloads if p.get("type") == "node_complete" and p.get("node") == "data_collection"]
    assert len(completes) == 2
    keys = {p.get("stream_step_key") for p in completes}
    assert keys == {"0:0:data_collection", "1:0:data_collection"}
