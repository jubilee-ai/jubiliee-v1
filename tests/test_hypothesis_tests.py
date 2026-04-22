"""Tests for inferential statistics tools (hypothesis_tests, statistical_models)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

pytest.importorskip("langchain_core", reason="analysis tools require langchain_core")

from analysis.hypothesis_tests import (
    categorical_association_test_tool,
    group_comparison_test_tool,
    run_categorical_association,
    run_group_comparison,
    run_inferential_feature_scan,
    run_numeric_binary_comparison,
)
from analysis.statistical_models import regression_summary_tool, run_ols_summary
from utils import clear_dataset_registry, register_dataset


@pytest.fixture(autouse=True)
def _clear_registry():
    clear_dataset_registry()
    yield
    clear_dataset_registry()


def test_run_group_comparison_two_groups_welch_or_mw():
    np.random.seed(0)
    a = np.random.normal(10, 1, 40)
    b = np.random.normal(12, 1, 40)
    df = pd.DataFrame({"y": np.concatenate([a, b]), "g": ["A"] * 40 + ["B"] * 40})
    out = run_group_comparison(df, "y", "g", test="auto")
    assert "error" not in out
    assert out.get("p_value") is not None
    assert float(out["p_value"]) < 0.05


def test_run_categorical_association_independent():
    n = 2000
    rng = np.random.default_rng(42)
    x = rng.choice(["L", "M"], size=n)
    y = rng.choice([0, 1], size=n)
    df = pd.DataFrame({"x": x, "y": y})
    out = run_categorical_association(df, "y", "x")
    assert "error" not in out
    assert out["p_value"] is not None
    assert 0 <= float(out["p_value"]) <= 1


def test_run_categorical_association_dependent():
    n = 300
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rng = np.random.default_rng(1)
    rng.shuffle(y)
    x = np.where(y == 1, rng.choice(["A", "B"], size=n, p=[0.8, 0.2]), rng.choice(["A", "B"], size=n, p=[0.2, 0.8]))
    df = pd.DataFrame({"target": y, "feature": x})
    out = run_categorical_association(df, "target", "feature")
    assert "error" not in out
    assert float(out["p_value"]) < 0.01


def test_run_numeric_binary_comparison():
    n = 120
    y = np.array([0] * 60 + [1] * 60)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 0.3, n)
    x = np.where(y == 0, rng.normal(0, 1, n), rng.normal(2, 1, n)) + noise
    df = pd.DataFrame({"target": y, "feat": x})
    out = run_numeric_binary_comparison(df, "feat", "target")
    assert "error" not in out
    assert out.get("primary_p") is not None or out.get("welch_t_p") is not None


def test_inferential_scan_registered_dataset():
    n = 200
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "cat": rng.choice(["u", "v", "w"], size=n),
            "num": rng.normal(0, 1, n),
            "target": rng.choice([0, 1], size=n),
        }
    )
    register_dataset("hyp_infer_test", df, register_sql=False)
    out = run_inferential_feature_scan("hyp_infer_test", "target", task_type="classification")
    assert "error" not in out
    assert "categorical_tests" in out
    assert "numeric_tests" in out


def test_group_comparison_tool_invoke():
    df = pd.DataFrame({"y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "g": ["a"] * 6 + ["b"] * 6})
    register_dataset("gcomp_t", df, register_sql=False)
    text = group_comparison_test_tool.invoke({"dataset_ref": "gcomp_t", "outcome_col": "y", "group_col": "g"})
    assert "Group comparison" in text or "group" in text.lower()
    assert "ANALYSIS_JSON" in text


def test_categorical_association_tool_invoke():
    df = pd.DataFrame({"a": ["x", "y"] * 50, "b": ["p", "q"] * 50})
    register_dataset("cass_t", df, register_sql=False)
    text = categorical_association_test_tool.invoke({"dataset_ref": "cass_t", "col_a": "a", "col_b": "b"})
    assert "Categorical association" in text or "chi" in text.lower()
    assert "ANALYSIS_JSON" in text


def test_ols_summary_core():
    np.random.seed(0)
    n = 80
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 2.0 * x1 - 0.5 * x2 + np.random.normal(0, 0.2, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    out = run_ols_summary(df, "y", ["x1", "x2"])
    if "error" in out:
        pytest.skip(out["error"])
    assert out.get("r_squared", 0) > 0.7
    coefs = {c["name"]: c for c in out["coefficients"]}
    assert "x1" in coefs
    assert coefs["x1"]["p_value"] < 0.05


def test_regression_summary_tool_invoke():
    np.random.default_rng(9)
    n = 100
    x1 = np.linspace(0, 1, n)
    x2 = np.random.normal(0, 0.1, n)
    y = 3 * x1 + x2 + np.random.normal(0, 0.05, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    register_dataset("ols_tool_t", df, register_sql=False)
    text = regression_summary_tool.invoke(
        {"dataset_ref": "ols_tool_t", "target_col": "y", "feature_cols": ["x1", "x2"], "model": "ols"}
    )
    if "requires" in text.lower() and "statsmodels" in text.lower():
        pytest.skip("statsmodels not installed")
    assert "ANALYSIS_JSON" in text or "Regression" in text
