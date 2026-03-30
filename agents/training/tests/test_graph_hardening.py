"""Tests for graph/HITL hardening (run_with_hitl error handling)."""

from unittest.mock import patch

import pytest

from agents.training.core.dispatcher import dispatcher_node
from agents.training.core.hitl import run_with_hitl
from agents.training.core.state import create_initial_state


@patch("agents.training.core.hitl.interrupt")
def test_run_with_hitl_work_fn_exception_returns_error_state(mock_interrupt):
    mock_interrupt.return_value = True

    state = create_initial_state("test goal")

    def work_fn(s, feedback):
        raise ValueError("work failed")

    def get_summary(r):
        return "ok"

    out = run_with_hitl("test_node", state, work_fn, get_summary)

    assert out.get("error") == "work failed"
    assert out.get("hitl_error_node") == "test_node"
    mock_interrupt.assert_called_once()


@patch("agents.training.core.hitl.interrupt")
def test_run_with_hitl_summary_fn_exception_uses_fallback(mock_interrupt):
    mock_interrupt.return_value = True

    state = create_initial_state("test goal")

    def work_fn(s, feedback):
        return {**s, "error": None, "some_key": "done"}

    def get_summary(r):
        raise RuntimeError("summary boom")

    out = run_with_hitl("summary_node", state, work_fn, get_summary)

    assert out.get("some_key") == "done"
    call = mock_interrupt.call_args[0][0]
    assert "could not build summary" in call["summary"]
    assert "summary boom" in call["summary"]
    assert call["node"] == "summary_node"


@patch("agents.training.core.hitl.interrupt")
def test_run_with_hitl_auto_approve_skips_interrupt(mock_interrupt):
    state = {**create_initial_state("g"), "hitl_auto_approve": True}

    def work_fn(s, feedback):
        return {**s, "done": True}

    out = run_with_hitl("auto_node", state, work_fn, lambda r: "ok")

    assert out.get("done") is True
    mock_interrupt.assert_not_called()


@patch("agents.training.core.dispatcher.emit_graph_stream")
def test_dispatcher_swaps_data_collection_before_cleaning(mock_emit):
    state = {
        "plan": [
            {"step": "select_model", "rationale": ""},
            {"step": "cleaning", "rationale": ""},
            {"step": "data_collection", "rationale": ""},
        ],
        "plan_index": 1,
        "collected_dataset_ref": None,
    }
    out = dispatcher_node(state)
    assert out["current_step"] == "data_collection"
    assert out["plan"][1]["step"] == "data_collection"
    assert out["plan"][2]["step"] == "cleaning"


@patch("agents.training.core.dispatcher.emit_graph_stream")
def test_dispatcher_inserts_data_collection_when_cleaning_first(mock_emit):
    state = {
        "plan": [{"step": "cleaning", "rationale": ""}],
        "plan_index": 0,
        "collected_dataset_ref": None,
    }
    out = dispatcher_node(state)
    assert out["current_step"] == "data_collection"
    assert len(out["plan"]) == 2
    assert out["plan"][0]["step"] == "data_collection"
    assert out["plan"][1]["step"] == "cleaning"


@patch("agents.training.core.dispatcher.emit_graph_stream")
def test_dispatcher_does_not_mutate_when_dataset_already_collected(mock_emit):
    state = {
        "plan": [{"step": "cleaning"}],
        "plan_index": 0,
        "collected_dataset_ref": "my_ds",
    }
    out = dispatcher_node(state)
    assert out["current_step"] == "cleaning"
    assert out["plan"] == state["plan"]
