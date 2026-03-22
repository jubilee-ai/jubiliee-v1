"""Tests for the Jubilee MCP server tools.

Uses mocking to isolate each tool from heavy dependencies (DB, model files,
LLM agents) so tests run fast and without infrastructure.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tools", "data-tools"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tools", "models-tools", "training"))

from mcp_server.server import (
    _align_features,
    _jsonable,
    assign_task,
    list_datasets,
    list_models,
    predict,
    train_model,
)


# ============================================================================
# Fixtures
# ============================================================================

SAMPLE_MODEL_FAMILIES = [
    {"id": "logistic_regression", "name": "Logistic Regression", "description": "Binary/multiclass classification"},
    {"id": "random_forest", "name": "Random Forest", "description": "Classification/regression"},
]

SAMPLE_TRAINED_MODELS = [
    {
        "model_name": "loan_default_v1",
        "model_type": "sklearn_random_forest",
        "description": "Loan default prediction",
        "metrics": {"test_accuracy": 0.91, "test_roc_auc": 0.87},
        "feature_names": ["age", "income", "employed"],
        "target_column": "default",
        "classes": [0, 1],
    },
]

SAMPLE_CATALOG_DATASETS = [
    {
        "id": "ds-001",
        "name": "Loan_default",
        "source_type": "csv",
        "format": "CSV",
        "rows": 1000,
        "columns": ["age", "income", "employed", "default"],
        "description": "Loan default dataset",
    },
]

SAMPLE_REGISTERED_DATASETS = [
    {"ref": "csv_Loan_default", "rows": 1000, "columns": 4},
]


# ============================================================================
# list_models
# ============================================================================

class TestListModels:

    @patch("backend.catalog.repository.get_models", return_value=SAMPLE_MODEL_FAMILIES)
    def test_returns_families_and_trained(self, mock_get_models):
        mock_storage = MagicMock()
        mock_storage.list_models.return_value = SAMPLE_TRAINED_MODELS
        with patch.dict("sys.modules", {"model_storage": mock_storage}):
            result = list_models(include_trained=True)

        assert "model_families" in result
        assert len(result["model_families"]) == 2
        assert result["model_families"][0]["id"] == "logistic_regression"
        assert "trained_models" in result
        assert len(result["trained_models"]) == 1

    @patch("backend.catalog.repository.get_models", return_value=SAMPLE_MODEL_FAMILIES)
    def test_exclude_trained(self, mock_get_models):
        result = list_models(include_trained=False)
        assert "model_families" in result
        assert len(result["model_families"]) == 2
        assert "trained_models" not in result

    @patch("backend.catalog.repository.get_models", return_value=SAMPLE_MODEL_FAMILIES)
    @patch("backend.catalog.repository.get_trained_models", return_value={})
    def test_trained_fallback_empty(self, mock_trained, mock_families):
        mock_storage = MagicMock()
        mock_storage.list_models.side_effect = Exception("DB unavailable")
        with patch.dict("sys.modules", {"model_storage": mock_storage}):
            result = list_models(include_trained=True)
        assert isinstance(result.get("trained_models", []), list)


# ============================================================================
# list_datasets
# ============================================================================

class TestListDatasets:

    @patch("backend.catalog.repository.get_datasets", return_value=SAMPLE_CATALOG_DATASETS)
    @patch("utils.get_all_available_datasets", return_value={
        "registered": SAMPLE_REGISTERED_DATASETS, "sql_tables": [], "catalog": [],
    })
    def test_returns_all_sources(self, mock_all_ds, mock_catalog):
        result = list_datasets()
        assert len(result["catalog"]) == 1
        assert result["catalog"][0]["name"] == "Loan_default"
        assert len(result["registered"]) == 1

    @patch("backend.catalog.repository.get_datasets", return_value=SAMPLE_CATALOG_DATASETS)
    @patch("utils.get_all_available_datasets", return_value={
        "registered": SAMPLE_REGISTERED_DATASETS, "sql_tables": [], "catalog": [],
    })
    def test_query_matches_catalog_and_registered(self, mock_all_ds, mock_catalog):
        result = list_datasets(query="loan")
        assert len(result["catalog"]) == 1
        assert len(result["registered"]) == 1  # csv_Loan_default contains "loan"

    @patch("backend.catalog.repository.get_datasets", return_value=SAMPLE_CATALOG_DATASETS)
    @patch("utils.get_all_available_datasets", return_value={
        "registered": SAMPLE_REGISTERED_DATASETS, "sql_tables": [], "catalog": [],
    })
    def test_query_no_match(self, mock_all_ds, mock_catalog):
        result = list_datasets(query="nonexistent_xyz")
        assert len(result["catalog"]) == 0
        assert len(result["registered"]) == 0

    @patch("backend.catalog.repository.get_datasets", side_effect=Exception("DB down"))
    def test_catalog_failure_graceful(self, mock_catalog):
        with patch("utils.get_all_available_datasets", return_value={
            "registered": [], "sql_tables": [], "catalog": [],
        }):
            result = list_datasets()
        assert result["catalog"] == []

    @patch("backend.catalog.repository.get_datasets", return_value=[])
    def test_no_query_returns_all(self, mock_catalog):
        with patch("utils.get_all_available_datasets", return_value={
            "registered": SAMPLE_REGISTERED_DATASETS, "sql_tables": [], "catalog": [],
        }):
            result = list_datasets(query=None)
        assert len(result["registered"]) == 1


# ============================================================================
# predict
# ============================================================================

class TestPredict:

    def _make_mock_classifier(self):
        model = MagicMock()
        model.predict.return_value = np.array([0, 1])
        model.predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        model.classes_ = np.array([0, 1])
        return model

    def _make_mock_regressor(self):
        model = MagicMock(spec=["predict", "fit"])
        model.predict.return_value = np.array([45000.0, 72000.0])
        return model

    def test_classifier_prediction(self):
        mock_model = self._make_mock_classifier()
        mock_info = {
            "model_name": "loan_v1",
            "model_type": "sklearn_logistic_regression",
            "target_column": "default",
            "feature_names": ["age", "income"],
        }
        with patch.dict("sys.modules", {"model_storage": MagicMock()}):
            import model_storage
            model_storage.get_model_info = MagicMock(return_value=mock_info)
            model_storage.load_model = MagicMock(return_value=mock_model)
            model_storage.list_models = MagicMock(return_value=[])

            result = predict(
                model_name="loan_v1",
                input_data=[
                    {"age": 30, "income": 50000},
                    {"age": 22, "income": 25000},
                ],
            )

        assert "error" not in result
        assert result["model_name"] == "loan_v1"
        assert result["num_records"] == 2
        assert len(result["predictions"]) == 2
        assert result["predictions"][0]["prediction"] == 0
        assert "probabilities" in result["predictions"][0]
        assert "0" in result["predictions"][0]["probabilities"]

    def test_regressor_prediction(self):
        mock_model = self._make_mock_regressor()
        mock_info = {
            "model_name": "income_v1",
            "model_type": "sklearn_random_forest",
            "target_column": "income",
            "feature_names": [],
        }
        with patch.dict("sys.modules", {"model_storage": MagicMock()}):
            import model_storage
            model_storage.get_model_info = MagicMock(return_value=mock_info)
            model_storage.load_model = MagicMock(return_value=mock_model)

            result = predict(
                model_name="income_v1",
                input_data=[{"age": 30}, {"age": 45}],
            )

        assert "error" not in result
        assert len(result["predictions"]) == 2
        assert result["predictions"][0]["prediction"] == 45000.0
        assert "probabilities" not in result["predictions"][0]

    def test_model_not_found(self):
        with patch.dict("sys.modules", {"model_storage": MagicMock()}):
            import model_storage
            model_storage.get_model_info = MagicMock(return_value=None)
            model_storage.list_models = MagicMock(return_value=[{"model_name": "other_model"}])

            result = predict(model_name="missing_model", input_data=[{"x": 1}])

        assert "error" in result
        assert "missing_model" in result["error"]
        assert "other_model" in result["available_models"]

    def test_model_load_failure(self):
        with patch.dict("sys.modules", {"model_storage": MagicMock()}):
            import model_storage
            model_storage.get_model_info = MagicMock(return_value={"model_name": "broken"})
            model_storage.load_model = MagicMock(side_effect=FileNotFoundError("not on disk"))

            result = predict(model_name="broken", input_data=[{"x": 1}])

        assert "error" in result
        assert "not on disk" in result["error"]

    def test_feature_alignment_fallback(self):
        """If direct predict fails, the tool attempts to align features."""
        expected_features = ["num__age", "num__income", "cat__employed_yes", "cat__employed_no"]

        model = MagicMock()
        # First call with raw columns fails, second call with aligned columns works
        model.predict.side_effect = [
            ValueError("column mismatch"),
            np.array([1]),
        ]
        model.predict_proba.return_value = np.array([[0.2, 0.8]])
        model.classes_ = np.array([0, 1])

        mock_info = {
            "model_name": "test_model",
            "model_type": "sklearn",
            "target_column": "default",
            "feature_names": expected_features,
        }

        with patch.dict("sys.modules", {"model_storage": MagicMock()}):
            import model_storage
            model_storage.get_model_info = MagicMock(return_value=mock_info)
            model_storage.load_model = MagicMock(return_value=model)

            result = predict(
                model_name="test_model",
                input_data=[{"age": 30, "income": 50000, "employed": "yes"}],
            )

        assert "error" not in result
        assert result["predictions"][0]["prediction"] == 1


# ============================================================================
# _align_features helper
# ============================================================================

class TestAlignFeatures:

    def test_numeric_and_categorical(self):
        df = pd.DataFrame([{"age": 30, "income": 50000, "employed": "yes"}])
        expected = ["num__age", "num__income", "cat__employed_yes", "cat__employed_no"]
        aligned = _align_features(df, expected)

        assert "num__age" in aligned.columns
        assert "num__income" in aligned.columns
        assert "cat__employed_yes" in aligned.columns
        assert aligned["cat__employed_yes"].iloc[0] == 1.0
        assert aligned["cat__employed_no"].iloc[0] == 0.0

    def test_missing_column_returns_partial(self):
        df = pd.DataFrame([{"age": 30}])
        expected = ["num__age", "num__missing_col"]
        aligned = _align_features(df, expected)
        assert "num__age" in aligned.columns


# ============================================================================
# _jsonable helper
# ============================================================================

class TestJsonable:

    def test_numpy_int(self):
        assert _jsonable(np.int64(42)) == 42
        assert isinstance(_jsonable(np.int64(42)), int)

    def test_numpy_float(self):
        assert _jsonable(np.float64(3.14)) == 3.14
        assert isinstance(_jsonable(np.float64(3.14)), float)

    def test_numpy_array(self):
        assert _jsonable(np.array([1, 2, 3])) == [1, 2, 3]

    def test_plain_python(self):
        assert _jsonable("hello") == "hello"
        assert _jsonable(42) == 42


# ============================================================================
# train_model
# ============================================================================

class TestTrainModel:

    @patch("agent._run_training_to_completion")
    def test_successful_training(self, mock_train):
        mock_train.return_value = {
            "selected_model": "random_forest",
            "model_explanation": "Best fit for tabular classification",
            "training_metrics": {
                "model_name": "loan_default_v2",
                "model_type": "sklearn_random_forest",
                "test_accuracy": 0.92,
                "test_roc_auc": 0.88,
                "summary": "Good performance",
                "recommendations": "Try feature engineering",
            },
            "report_path": "/reports/loan_default_v2.html",
            "model_weights_path": "/models/loan_default_v2.joblib",
        }

        result = train_model(
            goal="Predict loan default",
            dataset_ref="csv_Loan_default",
            model_preference="supervised",
        )

        assert result["status"] == "completed"
        assert result["selected_model"] == "random_forest"
        assert result["model_name"] == "loan_default_v2"
        assert result["metrics"]["test_accuracy"] == 0.92
        assert result["report_path"] == "/reports/loan_default_v2.html"

        mock_train.assert_called_once_with(
            goal="Predict loan default",
            linked_datasets=["csv_Loan_default"],
            model_preference="supervised",
            use_external_sources=False,
        )

    @patch("agent._run_training_to_completion")
    def test_training_no_metrics(self, mock_train):
        mock_train.return_value = {
            "selected_model": "logistic_regression",
            "training_metrics": None,
        }
        result = train_model(goal="Train something")
        assert result["status"] == "incomplete"
        assert result["selected_model"] == "logistic_regression"

    @patch("agent._run_training_to_completion", side_effect=RuntimeError("OOM"))
    def test_training_exception(self, mock_train):
        result = train_model(goal="Train something big")
        assert "error" in result
        assert "OOM" in result["error"]
        assert "traceback" in result

    @patch("agent._run_training_to_completion")
    def test_no_dataset_ref_passes_none(self, mock_train):
        mock_train.return_value = {"training_metrics": {}}
        train_model(goal="Auto train")
        mock_train.assert_called_once_with(
            goal="Auto train",
            linked_datasets=None,
            model_preference=None,
            use_external_sources=False,
        )

    @patch("agent._run_training_to_completion")
    def test_external_sources_flag(self, mock_train):
        mock_train.return_value = {"training_metrics": {}}
        train_model(goal="Find data and train", use_external_sources=True)
        mock_train.assert_called_once_with(
            goal="Find data and train",
            linked_datasets=None,
            model_preference=None,
            use_external_sources=True,
        )


# ============================================================================
# assign_task
# ============================================================================

class TestAssignTask:

    @patch("agent.agent")
    def test_successful_task(self, mock_orchestrator):
        mock_message = MagicMock()
        mock_message.content = "The default risk is HIGH (probability 0.82)."
        mock_orchestrator.invoke.return_value = {"messages": [mock_message]}

        result = assign_task(task="What is the default risk for a 35-year-old?")

        assert "error" not in result
        assert "default risk" in result["answer"].lower()
        assert result["thread_id"].startswith("mcp-")
        assert result["data_source_ref"] is None

    @patch("agent.agent")
    def test_task_with_data_source(self, mock_orchestrator):
        mock_message = MagicMock()
        mock_message.content = "Analysis complete."
        mock_orchestrator.invoke.return_value = {"messages": [mock_message]}

        result = assign_task(task="Analyze trends", data_source_ref="csv_Loan_default")

        call_args = mock_orchestrator.invoke.call_args[0][0]
        assert "csv_Loan_default" in call_args["messages"][0]["content"]
        assert result["data_source_ref"] == "csv_Loan_default"

    @patch("agent.agent")
    def test_task_exception(self, mock_orchestrator):
        mock_orchestrator.invoke.side_effect = Exception("LLM timeout")

        result = assign_task(task="Do something")

        assert "error" in result
        assert "LLM timeout" in result["error"]
        assert "traceback" in result

    @patch("agent.agent")
    def test_empty_response(self, mock_orchestrator):
        mock_orchestrator.invoke.return_value = {"messages": []}
        result = assign_task(task="Hello")
        assert result["answer"] == "No response generated."


# ============================================================================
# MCP registration
# ============================================================================

class TestMCPRegistration:

    def test_server_has_expected_tools(self):
        from mcp_server.server import mcp as server

        tool_names = {t.name for t in server._tool_manager._tools.values()}
        expected = {"list_models", "list_datasets", "predict", "train_model", "assign_task"}
        assert expected.issubset(tool_names), f"Missing tools: {expected - tool_names}"

    def test_server_name(self):
        from mcp_server.server import mcp as server
        assert server.name == "Jubilee ML Platform"

    def test_server_instructions_set(self):
        from mcp_server.server import mcp as server
        assert "list_models" in server.instructions
