"""Unit tests for model ↔ experiment linking helpers (no DB)."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.training import repository as tr


def test_extract_model_name_prefers_training_metrics():
    assert (
        tr.extract_model_name_from_training_state(
            {"training_metrics": {"model_name": "  my_model  "}}
        )
        == "my_model"
    )


def test_extract_model_name_from_weights_path_plain_string():
    assert (
        tr.extract_model_name_from_training_state(
            {"model_weights_path": "ridge_baseline_v1"}
        )
        == "ridge_baseline_v1"
    )


def test_extract_model_name_from_weights_path_file_like():
    assert (
        tr.extract_model_name_from_training_state(
            {"model_weights_path": "/tmp/trained_models/foo_bar.joblib"}
        )
        == "foo_bar"
    )


def test_link_model_to_experiment_updates_properties(monkeypatch):
    mock_model = SimpleNamespace(name="m1", properties={})

    class _Sess:
        def query(self, model_cls):
            assert model_cls.__name__ == "Model"
            q = MagicMock()

            def filter(*_args, **_kwargs):
                fq = MagicMock()
                fq.first = lambda: mock_model
                return fq

            q.filter = filter
            return q

        def get(self, _entity, _eid):
            return SimpleNamespace(name="My experiment")

    class _Ctx:
        def __enter__(self):
            return _Sess()

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(tr, "get_db_session", lambda: _Ctx())

    tr.link_model_to_experiment("m1", "exp-abc123")

    assert mock_model.properties["experiment_id"] == "exp-abc123"
    assert mock_model.properties["experiment_name"] == "My experiment"


def test_link_model_to_experiment_noop_when_missing():
    # Should not raise
    tr.link_model_to_experiment(None, "exp-1")
    tr.link_model_to_experiment("x", None)
