"""Unit tests for deterministic H2O AutoML path (``run_h2o_training``)."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "data-tools"))


@pytest.fixture
def clf_data():
    return pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            "f2": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _fake_h2o_frame(df):
    class _Fr:
        def __init__(self, columns):
            self.columns = columns
            self._ymock = MagicMock()
            self._ymock.asfactor.return_value = self._ymock

        def __getitem__(self, key):
            return self._ymock if key == "y" else MagicMock()

        def __setitem__(self, key, val):
            return None

    return _Fr(list(df.columns))


def test_java_missing_returns_fix(clf_data):
    from utils import register_dataset

    register_dataset("java_train", clf_data.iloc[:6], register_sql=False)
    register_dataset("java_val", clf_data.iloc[6:], register_sql=False)

    from agents.training.steps import h2o_training as ht

    with patch.object(ht, "_h2o_available", return_value=(True, "")), patch.object(
        ht, "_java_runtime_check", return_value=(False, "/usr/bin/java", "stub failed")
    ):
        out = ht.run_h2o_training(
            train_ref="java_train",
            val_ref="java_val",
            test_ref=None,
            target_column="y",
            goal="classify",
            selected_model="supervised",
            explicit_task_type="classification",
        )
    assert out["success"] is False
    assert "fix" in out
    assert "H2O-3" in out["fix"] or "JDK" in out["fix"]


def test_run_h2o_training_calls_automl_max_models_20(clf_data):
    from utils import register_dataset

    register_dataset("ht_train", clf_data.iloc[:6], register_sql=False)
    register_dataset("ht_val", clf_data.iloc[6:], register_sql=False)

    captured: dict = {}

    class FakeLeader:
        model_id = "GBM_1"
        algo = "gbm"

        def model_performance(self, *args, **kwargs):
            m = MagicMock()
            m.auc.return_value = 0.82
            m.accuracy.return_value = [(0.5, 0.79)]
            m.rmse.return_value = 0.1
            m.mae.return_value = 0.05
            m.r2.return_value = 0.5
            return m

        def varimp(self, use_pandas=True):
            return pd.DataFrame({"variable": ["f1", "f2"], "relative_importance": [0.6, 0.4]})

    class FakeAML:
        leader = FakeLeader()

        def train(self, **kwargs):
            captured["train_kw"] = kwargs

        @property
        def leaderboard(self):
            lb = pd.DataFrame(
                {
                    "model_id": ["GBM_1", "DRF_1"],
                    "auc": [0.82, 0.70],
                    "mean_per_class_error": [0.21, 0.30],
                }
            )
            m = MagicMock()
            m.as_data_frame = lambda: lb
            return m

    mock_automl_cls = MagicMock(return_value=FakeAML())

    fake_h2o = types.ModuleType("h2o")
    fake_h2o.__path__ = []  # noqa: SLF001 — package stub for submodule imports
    fake_h2o.H2OFrame = lambda df: _fake_h2o_frame(df)
    fake_h2o.init = MagicMock()

    def _cluster():
        return types.SimpleNamespace(shutdown=MagicMock())

    fake_h2o.cluster = _cluster

    def fake_get_model(_mid):
        m = MagicMock()
        m.download_mojo = lambda _d: "/tmp/fake.mojo"
        return m

    fake_h2o.get_model = fake_get_model

    fake_automl_mod = types.ModuleType("h2o.automl")
    fake_automl_mod.H2OAutoML = mock_automl_cls

    from agents.training.steps import h2o_training as ht

    saved = {k: sys.modules[k] for k in ("h2o", "h2o.automl") if k in sys.modules}
    for k in list(saved):
        del sys.modules[k]
    try:
        with patch.object(ht, "_h2o_available", return_value=(True, "")), patch.object(
            ht, "_java_runtime_check", return_value=(True, "/opt/java/bin/java", "")
        ), patch.object(
            ht, "register_h2o_mojo_model", return_value={"ok": True, "registered": {"model_name": "m1"}}
        ):
            with patch.dict(sys.modules, {"h2o": fake_h2o, "h2o.automl": fake_automl_mod}, clear=False):
                out = ht.run_h2o_training(
                    train_ref="ht_train",
                    val_ref="ht_val",
                    test_ref=None,
                    target_column="y",
                    goal="binary classification",
                    selected_model="supervised",
                    model_name="automl_test_model",
                    explicit_task_type="classification",
                )
    finally:
        sys.modules.update(saved)

    assert out["success"] is True
    assert out["model_name"] == "automl_test_model"
    assert mock_automl_cls.called
    aml_kwargs = mock_automl_cls.call_args[1]
    assert aml_kwargs.get("max_models") == 20
    assert captured["train_kw"]["y"] == "y"
    assert set(captured["train_kw"]["x"]) == {"f1", "f2"}
    assert "validation_frame" in captured["train_kw"]


def test_unsupervised_delegates_to_run_training_agent():
    from utils import register_dataset

    register_dataset("us_train", pd.DataFrame({"a": [1, 2], "b": [3, 4]}), register_sql=False)

    from agents.training.steps import h2o_training as ht

    with patch("agents.training.steps.training.run_training_agent", return_value={"success": True, "delegated": True}):
        out = ht.run_h2o_training(
            train_ref="us_train",
            val_ref=None,
            test_ref=None,
            target_column="",
            goal="cluster segments",
            selected_model="unsupervised",
            explicit_task_type="unsupervised",
        )
    assert out.get("delegated") is True
