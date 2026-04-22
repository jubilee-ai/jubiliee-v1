"""
Verify parallel execution: batch_train_with_skill (threads), evaluate_models (threads),
and feature experiment grid (process pool).

Run: python3 -m pytest tests/test_parallel_training_paths.py -v
"""

import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "data-tools"))


def test_batch_train_runs_multiple_workers_concurrently():
    """Peak concurrent _run_skill calls should be >1 when submitting 3 jobs."""
    from agents.training.steps import training as training_mod

    concurrent = {"n": 0, "peak": 0}
    lock = threading.Lock()

    def slow_run_skill(skill_name: str, params: dict) -> str:
        with lock:
            concurrent["n"] += 1
            concurrent["peak"] = max(concurrent["peak"], concurrent["n"])
        time.sleep(0.2)
        with lock:
            concurrent["n"] -= 1
        return f"trained {params.get('model_name')}"

    configs = [
        training_mod.BatchTrainConfig(estimator="A", model_name="m1", hyperparams={}),
        training_mod.BatchTrainConfig(estimator="B", model_name="m2", hyperparams={}),
        training_mod.BatchTrainConfig(estimator="C", model_name="m3", hyperparams={}),
    ]

    _mock_settings = MagicMock()
    _mock_settings.MAX_PARALLEL_BATCH_TRAIN = 3

    with patch.object(training_mod, "_run_skill", side_effect=slow_run_skill), patch(
        "backend.shared.settings.get_settings", return_value=_mock_settings
    ):
        out = training_mod.batch_train_with_skill_tool.invoke(
            {
                "skill_name": "supervised",
                "configs": configs,
                "train_dataset_ref": "dummy_train",
                "val_dataset_ref": "dummy_val",
                "target_column": "y",
            }
        )

    assert concurrent["peak"] >= 2, (
        f"Expected overlapping runs (peak concurrent >= 2), got peak={concurrent['peak']}. "
        "ThreadPoolExecutor is not running jobs in parallel."
    )
    assert "m1" in out and "m2" in out and "m3" in out, out[:500]


def test_tool_training_forwards_experiment_context_to_run_training():
    """Regression: outer agent passes experiment_result / feature_rankings into run_training_agent."""
    from agents.training.agent_simple import create_simple_training_agent

    _tools = {}

    def capture_create_agent(**kwargs):
        for fn in kwargs.get("tools", []):
            _tools[fn.__name__] = fn
        return MagicMock()

    received = {}

    def mock_run_training(**kwargs):
        received.clear()
        received.update(
            {
                "experiment_result": kwargs.get("experiment_result"),
                "feature_rankings": kwargs.get("feature_rankings"),
            }
        )
        return {
            "success": True,
            "model_name": "x",
            "model_type": "supervised",
            "feature_redo_requested": False,
            "iterations": [],
            "num_iterations": 0,
            "best_iteration": None,
        }

    with patch("agents.training.agent_simple.create_deep_agent", side_effect=capture_create_agent), patch(
        "agents.training.agent_simple._run_training", side_effect=mock_run_training
    ), patch("agents.training.agent_simple.init_chat_model", return_value=MagicMock()):
        _agent, state = create_simple_training_agent(goal="test", hitl=False, model=MagicMock())
        state.update({
            "label_definition": {"target_column": "target"},
            "transformed_train_ref": "ttr",
            "transformed_val_ref": "tvr",
            "transformed_test_ref": "tst",
            "selected_model": "supervised",
            "training_plan": {"max_iterations": 2},
            "experiment_result": {"total_scouts": 4, "best_variant_name": "mi_top_k"},
            "feature_rankings": {"f1": 0.9, "f2": 0.1},
        })
        _tools["tool_training"]()

    assert received.get("experiment_result", {}).get("best_variant_name") == "mi_top_k"
    assert received.get("feature_rankings") == {"f1": 0.9, "f2": 0.1}


def test_evaluate_models_runs_three_fits_with_peak_concurrency():
    """tool_evaluate_models uses ThreadPoolExecutor; slow fits should overlap."""
    from utils import clear_registry, register_dataset
    from agents.training.agent_simple import create_simple_training_agent

    _cols = {f"c{i}": np.random.randn(80) for i in range(4)}
    _cols["target"] = np.random.randint(0, 2, 80)
    train_df = pd.DataFrame(_cols)
    val_df = train_df.copy()
    clear_registry()
    register_dataset("ev_train", train_df, persist=False, register_sql=False)
    register_dataset("ev_val", val_df, persist=False, register_sql=False)

    _tools = {}
    _cap = {"peak": 0, "active": 0}
    _lock = threading.Lock()

    def wrap(cls):
        class Slow(cls):
            def fit(self, X, y):
                with _lock:
                    _cap["active"] += 1
                    _cap["peak"] = max(_cap["peak"], _cap["active"])
                time.sleep(0.12)
                try:
                    return super().fit(X, y)
                finally:
                    with _lock:
                        _cap["active"] -= 1
        return Slow

    import sklearn.ensemble as ensemble_mod
    import sklearn.linear_model as linear_mod

    _mock_settings = MagicMock()
    _mock_settings.MAX_PARALLEL_MODEL_EVAL = 3

    with (
        patch.object(
            ensemble_mod,
            "HistGradientBoostingClassifier",
            wrap(ensemble_mod.HistGradientBoostingClassifier),
        ),
        patch.object(
            ensemble_mod,
            "RandomForestClassifier",
            wrap(ensemble_mod.RandomForestClassifier),
        ),
        patch.object(
            linear_mod,
            "LogisticRegression",
            wrap(linear_mod.LogisticRegression),
        ),
        patch("backend.shared.settings.get_settings", return_value=_mock_settings),
    ):

        def cap_create(**kw):
            for fn in kw.get("tools", []):
                _tools[fn.__name__] = fn
            return MagicMock()

        with patch("agents.training.agent_simple.create_deep_agent", side_effect=cap_create), patch(
            "agents.training.agent_simple.init_chat_model", return_value=MagicMock()
        ):
            _a, state = create_simple_training_agent(goal="classification", hitl=False, model=MagicMock())

        state.update({
            "transformed_train_ref": "ev_train",
            "transformed_val_ref": "ev_val",
            "label_definition": {"target_column": "target"},
            "selected_model": "supervised",
            "goal": "predict loan default",
        })
        out = _tools["tool_evaluate_models"](
            "HistGradientBoostingClassifier, RandomForestClassifier, LogisticRegression"
        )

    assert _cap["peak"] >= 2, (
        f"evaluate_models should overlap fits (peak active >= 2), got {_cap['peak']}"
    )
    assert "Model Comparison" in out or "## Model Comparison Results" in out, out[:400]


@pytest.fixture
def registered_classification_split():
    from utils import clear_registry, register_dataset

    clear_registry()
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "f3": np.random.randn(n),
        "f4": np.random.randn(n),
        "f5": np.random.randn(n),
        "f6": np.random.randn(n),
        "target": np.random.randint(0, 2, n),
    })
    train, val = df.iloc[:140].copy(), df.iloc[140:].copy()
    register_dataset("exp_raw_train", train, persist=False, register_sql=False)
    register_dataset("exp_raw_val", val, persist=False, register_sql=False)
    yield
    clear_registry()


def test_run_experiment_grid_runs_multiple_scouts(registered_classification_split):
    """Feature experiment grid: variants × 2 families, parallel process pool."""
    from agents.training.steps.feature_experiment_runner import generate_feature_variants, run_experiment_grid
    from utils import get_registered_dataset

    features = [
        {"name": f"f{i}", "formula": {"op": "passthrough", "column": f"f{i}"}}
        for i in range(1, 7)
    ]
    spec = {"features": features}
    train = get_registered_dataset("exp_raw_train")
    variants = generate_feature_variants(spec, train, "target", "classification")
    assert len(variants) >= 2, "Expected multiple variants for 6 features"

    t0 = time.perf_counter()
    result = run_experiment_grid(
        feature_spec=spec,
        train_ref="exp_raw_train",
        val_ref="exp_raw_val",
        test_ref=None,
        target_column="target",
        task_type="classification",
        max_workers=4,
    )
    elapsed = time.perf_counter() - t0

    assert result.total_scouts > 0, "No scout jobs ran"
    assert result.total_variants >= 1
    print(
        f"\n  [experiment_grid] variants={result.total_variants} scouts={result.total_scouts} "
        f"wall_time_reported={result.wall_time_seconds}s perf_counter={elapsed:.2f}s "
        f"best={result.best_variant_name} metric={result.best_metric:.4f}"
    )


def test_aggregate_transformed_importances_merges_ohe_columns():
    """Preprocessor feature names (num__/cat__) aggregate onto raw DataFrame columns."""
    from agents.training.steps import training as training_mod

    names = ["num__Age", "cat__Education_Bachelor", "cat__Education_PhD"]
    imp = np.array([0.5, 0.3, 0.2])
    raw = ["Age", "Education"]
    out = training_mod._aggregate_transformed_importances_to_raw(names, imp, raw)
    assert out["Age"] == 0.5
    assert out["Education"] == pytest.approx(0.5)


def test_select_scoring_prefers_binary_ranking_metrics():
    from agents.training.skills.supervised.train import _select_scoring

    y_balanced = pd.Series([0, 1] * 20)
    y_rare = pd.Series([0] * 95 + [1] * 5)

    assert _select_scoring(True, y_balanced) == "roc_auc"
    assert _select_scoring(True, y_rare) == "average_precision"


def test_supervised_skill_reports_validation_metrics():
    from agents.training.skills.supervised.train import run
    from utils import clear_registry, register_dataset

    clear_registry()
    train_df = pd.DataFrame({
        "x1": [0, 0, 1, 1, 0, 1, 0, 1],
        "x2": [0, 1, 0, 1, 0, 1, 1, 0],
        "target": [0, 0, 0, 1, 0, 1, 1, 1],
    })
    val_df = pd.DataFrame({
        "x1": [0, 1, 0, 1],
        "x2": [1, 0, 0, 1],
        "target": [0, 1, 0, 1],
    })
    register_dataset("skill_train", train_df, persist=False, register_sql=False)
    register_dataset("skill_val", val_df, persist=False, register_sql=False)

    out = run({
        "estimator": "LogisticRegression",
        "train_dataset_ref": "skill_train",
        "val_dataset_ref": "skill_val",
        "target_column": "target",
        "model_name": "skill_val_metrics_model",
        "auto_tune": False,
    })

    assert "VALIDATION METRICS" in out
    assert "Val Accuracy:" in out


def test_generate_feature_variants_mi_uses_source_columns():
    from agents.training.steps.feature_experiment_runner import generate_feature_variants

    train = pd.DataFrame({
        "raw_a": [0, 1, 0, 1, 0, 1],
        "raw_b": [1, 0, 1, 0, 1, 0],
        "raw_c": [0, 0, 1, 1, 0, 1],
        "raw_d": [1, 1, 0, 0, 1, 0],
        "target": [0, 1, 0, 1, 0, 1],
    })
    spec = {
        "features": [
            {"name": "expr_ratio", "formula": {"op": "expression", "source_columns": ["raw_a", "raw_b"]}},
            {"name": "raw_c_feat", "formula": {"op": "passthrough", "column": "raw_c"}},
            {"name": "raw_d_feat", "formula": {"op": "passthrough", "column": "raw_d"}},
            {"name": "raw_a_feat", "formula": {"op": "passthrough", "column": "raw_a"}},
            {"name": "raw_b_feat", "formula": {"op": "passthrough", "column": "raw_b"}},
        ]
    }

    fake_scores = {"raw_a": 0.9, "raw_b": 0.8, "raw_c": 0.7, "raw_d": 0.1}
    with patch("agents.training.steps.feature_experiment_runner._compute_mutual_info", return_value=fake_scores):
        variants = generate_feature_variants(spec, train, "target", "classification")

    mi_variant = next(v for v in variants if v.name == "mi_top_k")
    kept_names = {f["name"] for f in mi_variant.feature_spec["features"]}
    assert "expr_ratio" in kept_names

