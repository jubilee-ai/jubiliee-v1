"""Tests for graph/HITL hardening and downstream failure handling."""

from unittest.mock import patch

import pandas as pd
import pytest

from agents.training.core import hitl as hitl_mod
from agents.training.core.dispatcher import dispatcher_node
from agents.training.core.evaluator import evaluator_node
from agents.training.core.hitl import make_serializable, run_with_hitl
from agents.training.core.state import create_initial_state
from agents.training.steps.label_and_split import normalize_label_definition_for_df
from agents.training.steps.orchestrator import (generate_report,
                                                label_split_definition,
                                                training, training_approval)


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


@patch("agents.training.core.hitl.interrupt")
def test_run_with_hitl_replay_uses_cache_skips_work_fn(mock_interrupt):
    """Second graph entry with pending cache must not re-run work_fn (HITL replay)."""
    mock_interrupt.return_value = True
    key = "thread-a::replay_node"
    state = create_initial_state("goal")
    cached = {k: make_serializable(v) for k, v in {**state, "marker": 42}.items()}
    hitl_mod._HITL_WORK_CACHE[key] = cached

    calls = {"n": 0}

    def work_fn(s, feedback):
        calls["n"] += 1
        return {**s, "marker": 99}

    with patch.object(hitl_mod, "_hitl_memory_key", return_value=key):
        out = run_with_hitl("replay_node", state, work_fn, lambda r: "summary")

    assert calls["n"] == 0
    assert out.get("marker") == 42
    assert key not in hitl_mod._HITL_WORK_CACHE


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


def test_evaluator_stops_interactive_run_on_error():
    state = {
        "plan": [
            {"step": "label_split_definition"},
            {"step": "feature_selection_specification"},
        ],
        "plan_index": 0,
        "current_step": "label_split_definition",
        "error": "bad target",
        "hitl_auto_approve": False,
        "audit_trace": [],
    }

    out = evaluator_node(state)

    assert out["evaluator_decision"] == "done"
    assert "bad target" in out["audit_trace"][-1]["reasoning"]


@patch("agents.training.steps.orchestrator.register_dataset")
@patch("agents.training.steps.orchestrator.get_registered_dataset")
@patch("agents.training.steps.orchestrator.run_label_split_definition")
def test_label_split_guard_auto_corrects_classification_to_regression(
    mock_run_label_split_definition, mock_get_registered_dataset, mock_register_dataset,
):
    df = pd.DataFrame(
        {
            "Financial Distress": list(range(120)),
            "x1": list(range(120)),
        }
    )
    mock_get_registered_dataset.return_value = df
    mock_register_dataset.return_value = "ref"
    mock_run_label_split_definition.return_value = {
        "target_column": "Financial Distress",
        "prediction_horizon": None,
        "grain": "one row",
        "as_of_cutoff": None,
        "split_strategy": "random",
        "forbidden_columns": [],
    }

    state = {
        **create_initial_state("predict distress"),
        "hitl_auto_approve": True,
        "selected_model": "supervised",
        "task_type": "classification",
        "cleaned_dataset_ref": "cleaned_ds",
        "train_dataset_ref": "stale_train",
        "val_dataset_ref": "stale_val",
        "test_dataset_ref": "stale_test",
        "transformed_train_ref": "stale_train_features",
        "experiment_result": {"best_variant_name": "old"},
    }

    out = label_split_definition(state)

    assert out["task_type"] == "regression"
    assert out.get("error") is None
    assert out["train_dataset_ref"] is not None


@patch("agents.training.steps.orchestrator._run_training")
def test_training_surfaces_failed_result_as_top_level_error(mock_run_training):
    mock_run_training.return_value = {
        "success": False,
        "error": "task type mismatch",
        "model_name": None,
        "model_type": "supervised",
    }

    state = {
        **create_initial_state("predict distress"),
        "hitl_auto_approve": True,
        "selected_model": "supervised",
        "task_type": "classification",
        "transformed_train_ref": "train_features",
        "transformed_val_ref": "val_features",
        "transformed_test_ref": "test_features",
        "label_definition": {"target_column": "target"},
    }

    out = training(state)

    assert out["training_metrics"]["success"] is False
    assert out["error"] == "task type mismatch"


@patch("agents.training.steps.orchestrator.init_chat_model")
@patch("agents.training.steps.orchestrator.get_registered_dataset")
def test_training_approval_uses_resolved_state_task_type(
    mock_get_registered_dataset, mock_init_chat_model
):
    class _Response:
        content = "{}"

    class _LLM:
        def invoke(self, _messages):
            return _Response()

    mock_init_chat_model.return_value = _LLM()
    mock_get_registered_dataset.return_value = pd.DataFrame(
        {"target": [0.1, 0.2, 0.3], "x1": [1, 2, 3]}
    )

    state = {
        **create_initial_state("predict distress"),
        "hitl_auto_approve": True,
        "selected_model": "supervised",
        "task_type": "regression",
        "transformed_train_ref": "train_features",
        "transformed_val_ref": "val_features",
        "label_definition": {"target_column": "target"},
        "feature_spec": {"features": [{"name": "x1"}]},
    }

    out = training_approval(state)

    assert out["training_plan"]["task_type"] == "regression"


def test_normalize_label_definition_falls_back_entity_when_unresolvable():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]})
    ld = {
        "split_strategy": "entity_based",
        "grain": "opaque grain with no id columns",
        "entity_column": None,
        "target_column": "y",
    }
    out = normalize_label_definition_for_df(df, ld)
    assert out["split_strategy"] == "random"


def test_normalize_label_definition_falls_back_time_when_column_missing():
    df = pd.DataFrame({"y": [1, 2, 3]})
    ld = {
        "split_strategy": "time_based",
        "as_of_cutoff": "no_such_col",
        "target_column": "y",
    }
    out = normalize_label_definition_for_df(df, ld)
    assert out["split_strategy"] == "random"
    assert out.get("as_of_cutoff") is None


def test_generate_report_refuses_failed_training_state():
    state = {
        **create_initial_state("predict distress"),
        "hitl_auto_approve": True,
        "training_metrics": {"success": False, "model_name": None},
    }

    out = generate_report(state)

    assert "Cannot generate report" in out["error"]
    assert out.get("report_path") is None
