"""Tests for linked-dataset resolution (resolve_linked_dataset + SSE integration)."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.training import service


def _collect_sse_payloads(gen) -> list[dict]:
    payloads = []
    for raw in gen:
        if isinstance(raw, str) and raw.startswith("data: "):
            payloads.append(json.loads(raw[6:].strip()))
    return payloads


# ---------------------------------------------------------------------------
# resolve_linked_dataset unit tests
# ---------------------------------------------------------------------------


@patch.object(service, "get_registered_dataset")
def test_resolve_already_registered(mock_get):
    mock_get.return_value = pd.DataFrame({"a": [1]})
    assert service.resolve_linked_dataset("csv_insurance") == "csv_insurance"
    mock_get.assert_called_once_with("csv_insurance")


@patch.object(service, "get_registered_dataset", return_value=None)
@patch.object(service, "load_and_register_dataset", return_value="csv_foo")
def test_resolve_via_filesystem(mock_load, mock_get):
    assert service.resolve_linked_dataset("csv/foo.csv") == "csv_foo"
    mock_load.assert_called_once_with("csv/foo.csv")


@patch.object(service, "get_registered_dataset", return_value=None)
@patch.object(service, "load_and_register_dataset")
def test_resolve_via_db_name(mock_load, mock_get):
    mock_load.side_effect = lambda path: "csv_insurance" if path == "csv/insurance.csv" else None

    fake_row = SimpleNamespace(
        id=uuid.uuid4(), name="Health Insurance Charges Dataset",
        properties={"file": "csv/insurance.csv"},
    )
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.query.return_value.filter.return_value.first.return_value = fake_row

    with patch("backend.training.service.get_db_session", return_value=mock_session):
        result = service.resolve_linked_dataset("Health Insurance Charges Dataset")

    assert result == "csv_insurance"


@patch.object(service, "get_registered_dataset", return_value=None)
@patch.object(service, "load_and_register_dataset")
def test_resolve_via_db_uuid(mock_load, mock_get):
    ds_id = uuid.uuid4()
    mock_load.side_effect = lambda path: "csv_ins" if path == "csv/ins.csv" else None

    fake_row = SimpleNamespace(
        id=ds_id, name="Insurance", properties={"file": "csv/ins.csv"},
    )
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    mock_session.query.return_value.filter.return_value.first.return_value = fake_row

    with patch("backend.training.service.get_db_session", return_value=mock_session):
        result = service.resolve_linked_dataset(str(ds_id))

    assert result == "csv_ins"


@patch.object(service, "get_registered_dataset", return_value=None)
@patch.object(service, "load_and_register_dataset", return_value=None)
def test_resolve_returns_none_on_all_failures(mock_load, mock_get):
    with patch("backend.training.service.get_db_session", side_effect=Exception("no db")):
        assert service.resolve_linked_dataset("nonexistent") is None


# ---------------------------------------------------------------------------
# SSE integration: dataset_error emitted when resolution fails
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_repo():
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


def test_graph_sse_emits_dataset_error_when_resolve_fails(in_memory_repo):
    thread = "graph-err-thread"

    with (
        patch.object(service, "resolve_linked_dataset", return_value=None),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events(
                "test", ["Bad Dataset Name"], None, thread_id=thread
            )
        )

    errors = [p for p in payloads if p.get("type") == "dataset.error"]
    assert len(errors) == 1
    assert errors[0]["ref"] == "Bad Dataset Name"
    assert "Bad Dataset Name" in errors[0]["error"]

    loaded = [p for p in payloads if p.get("type") == "dataset.resolved"]
    assert len(loaded) == 0


def test_graph_sse_emits_dataset_loaded_when_resolve_succeeds(in_memory_repo):
    thread = "graph-ok-thread"

    def fake_stream(_input, config=None, stream_mode=None):
        yield ("updates", {"__interrupt__": []})

    mock_agent = MagicMock()
    mock_agent.stream = fake_stream
    mock_agent.get_state.return_value = SimpleNamespace(values={"goal": "test"})

    with (
        patch(
            "agents.training.agent_simple.create_simple_training_agent",
            return_value=(mock_agent, {"goal": "test"}),
        ),
        patch("langgraph.checkpoint.memory.MemorySaver", return_value=MagicMock()),
        patch.object(service, "resolve_linked_dataset", return_value="csv_insurance"),
    ):
        payloads = _collect_sse_payloads(
            service.generate_graph_sse_events(
                "test", ["csv/insurance.csv"], None, thread_id=thread
            )
        )

    loaded = [p for p in payloads if p.get("type") == "dataset.resolved"]
    assert len(loaded) == 1
    assert loaded[0]["ref"] == "csv_insurance"
    assert loaded[0].get("dataset_info", {}).get("dataset") == "csv/insurance.csv"

    errors = [p for p in payloads if p.get("type") == "dataset.error"]
    assert len(errors) == 0
