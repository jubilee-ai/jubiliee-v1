"""Tests for the sklearn Random Forest training tool."""

import os
import sys
import uuid

import pytest

# Add training module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models-tools', 'training'))


def get_classification_data():
    """Generate sample binary classification data for testing."""
    return [
        {'claim_amount': 2500, 'age': 45, 'prior_claims': 0, 'policy_age': 5, 'fraud': 0},
        {'claim_amount': 3200, 'age': 52, 'prior_claims': 1, 'policy_age': 8, 'fraud': 0},
        {'claim_amount': 1800, 'age': 38, 'prior_claims': 0, 'policy_age': 3, 'fraud': 0},
        {'claim_amount': 4500, 'age': 61, 'prior_claims': 2, 'policy_age': 12, 'fraud': 0},
        {'claim_amount': 2100, 'age': 29, 'prior_claims': 0, 'policy_age': 2, 'fraud': 0},
        {'claim_amount': 3800, 'age': 55, 'prior_claims': 1, 'policy_age': 7, 'fraud': 0},
        {'claim_amount': 45000, 'age': 28, 'prior_claims': 4, 'policy_age': 1, 'fraud': 1},
        {'claim_amount': 52000, 'age': 31, 'prior_claims': 3, 'policy_age': 0, 'fraud': 1},
        {'claim_amount': 38000, 'age': 25, 'prior_claims': 5, 'policy_age': 1, 'fraud': 1},
        {'claim_amount': 61000, 'age': 27, 'prior_claims': 4, 'policy_age': 0, 'fraud': 1},
        {'claim_amount': 48000, 'age': 33, 'prior_claims': 3, 'policy_age': 1, 'fraud': 1},
        {'claim_amount': 55000, 'age': 29, 'prior_claims': 6, 'policy_age': 0, 'fraud': 1},
    ]


def get_regression_data():
    """Generate sample regression data for testing."""
    return [
        {'age': 25, 'vehicle_age': 1, 'policy_type': 'basic', 'claim_amount': 2500},
        {'age': 35, 'vehicle_age': 3, 'policy_type': 'comprehensive', 'claim_amount': 4500},
        {'age': 45, 'vehicle_age': 5, 'policy_type': 'basic', 'claim_amount': 3200},
        {'age': 55, 'vehicle_age': 2, 'policy_type': 'premium', 'claim_amount': 8500},
        {'age': 30, 'vehicle_age': 4, 'policy_type': 'comprehensive', 'claim_amount': 5100},
        {'age': 40, 'vehicle_age': 6, 'policy_type': 'basic', 'claim_amount': 2800},
        {'age': 50, 'vehicle_age': 1, 'policy_type': 'premium', 'claim_amount': 9200},
        {'age': 28, 'vehicle_age': 2, 'policy_type': 'basic', 'claim_amount': 2100},
        {'age': 38, 'vehicle_age': 3, 'policy_type': 'comprehensive', 'claim_amount': 5800},
        {'age': 48, 'vehicle_age': 7, 'policy_type': 'basic', 'claim_amount': 3500},
        {'age': 60, 'vehicle_age': 1, 'policy_type': 'premium', 'claim_amount': 10500},
        {'age': 33, 'vehicle_age': 4, 'policy_type': 'comprehensive', 'claim_amount': 4200},
    ]


def unique_model_name(prefix="test_rf"):
    """Generate unique model name for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestRandomForestClassification:
    """Test suite for Random Forest classification."""

    def test_basic_classification(self):
        """Test basic classification training."""
        from random_forest import train_random_forest, RandomForestTrainingInput
        from model_storage import delete_model

        data = get_classification_data()
        model_name = unique_model_name("test_clf")

        try:
            result = train_random_forest(RandomForestTrainingInput(
                model_name=model_name,
                data=data,
                target_column="fraud",
                task_type="classification",
                n_estimators=10,
            ))

            assert result.success is True
            assert result.task_type == "classification"
            assert result.n_samples == 12
            assert result.n_classes == 2
            assert result.test_score >= 0.0 and result.test_score <= 1.0
            assert len(result.feature_importances) > 0
            assert result.classification_report is not None
            assert result.confusion_matrix is not None
        finally:
            delete_model(model_name)

    def test_classification_with_balanced_weights(self):
        """Test classification with balanced class weights."""
        from random_forest import train_random_forest, RandomForestTrainingInput
        from model_storage import delete_model

        data = get_classification_data()
        model_name = unique_model_name("test_balanced")

        try:
            result = train_random_forest(RandomForestTrainingInput(
                model_name=model_name,
                data=data,
                target_column="fraud",
                task_type="classification",
                n_estimators=10,
                class_weight="balanced",
            ))

            assert result.success is True
        finally:
            delete_model(model_name)

    def test_classification_with_depth_limit(self):
        """Test classification with max_depth constraint."""
        from random_forest import train_random_forest, RandomForestTrainingInput
        from model_storage import delete_model

        data = get_classification_data()
        model_name = unique_model_name("test_depth")

        try:
            result = train_random_forest(RandomForestTrainingInput(
                model_name=model_name,
                data=data,
                target_column="fraud",
                task_type="classification",
                n_estimators=10,
                max_depth=3,
            ))

            assert result.success is True
        finally:
            delete_model(model_name)


class TestRandomForestRegression:
    """Test suite for Random Forest regression."""

    def test_basic_regression(self):
        """Test basic regression training."""
        from random_forest import train_random_forest, RandomForestTrainingInput
        from model_storage import delete_model

        data = get_regression_data()
        model_name = unique_model_name("test_reg")

        try:
            result = train_random_forest(RandomForestTrainingInput(
                model_name=model_name,
                data=data,
                target_column="claim_amount",
                task_type="regression",
                categorical_columns=["policy_type"],
                n_estimators=10,
            ))

            assert result.success is True
            assert result.task_type == "regression"
            assert result.n_samples == 12
            assert result.n_classes is None
            assert result.classification_report is None
            assert "mae" in result.additional_metrics
            assert "rmse" in result.additional_metrics
            assert len(result.feature_importances) > 0
        finally:
            delete_model(model_name)

    def test_regression_oob_score(self):
        """Test regression with OOB score enabled."""
        from random_forest import train_random_forest, RandomForestTrainingInput
        from model_storage import delete_model

        data = get_regression_data()
        model_name = unique_model_name("test_oob")

        try:
            result = train_random_forest(RandomForestTrainingInput(
                model_name=model_name,
                data=data,
                target_column="claim_amount",
                task_type="regression",
                categorical_columns=["policy_type"],
                n_estimators=20,
                bootstrap=True,
            ))

            assert result.success is True
            assert result.oob_score is not None
        finally:
            delete_model(model_name)


class TestRandomForestTool:
    """Test suite for the LangChain tool wrapper."""

    def test_classification_tool(self):
        """Test the classification tool returns formatted string."""
        from random_forest import sklearn_random_forest_tool
        from model_storage import delete_model

        data = get_classification_data()
        model_name = unique_model_name("test_tool_clf")

        try:
            result = sklearn_random_forest_tool.invoke({
                "model_name": model_name,
                "data": data,
                "target_column": "fraud",
                "task_type": "classification",
                "n_estimators": 10,
            })

            assert isinstance(result, str)
            assert "RANDOM FOREST CLASSIFICATION TRAINING COMPLETE" in result
            assert "FEATURE IMPORTANCES" in result
            assert "MODEL REGISTERED" in result
        finally:
            delete_model(model_name)

    def test_regression_tool(self):
        """Test the regression tool returns formatted string."""
        from random_forest import sklearn_random_forest_tool
        from model_storage import delete_model

        data = get_regression_data()
        model_name = unique_model_name("test_tool_reg")

        try:
            result = sklearn_random_forest_tool.invoke({
                "model_name": model_name,
                "data": data,
                "target_column": "claim_amount",
                "task_type": "regression",
                "categorical_columns": ["policy_type"],
                "n_estimators": 10,
            })

            assert isinstance(result, str)
            assert "RANDOM FOREST REGRESSION TRAINING COMPLETE" in result
            assert "MAE" in result
            assert "RMSE" in result
        finally:
            delete_model(model_name)

    def test_tool_error_handling(self):
        """Test the tool handles errors gracefully."""
        from random_forest import sklearn_random_forest_tool

        result = sklearn_random_forest_tool.invoke({
            "model_name": unique_model_name("test_error"),
            "data": [{"a": 1}],
            "target_column": "nonexistent",
            "task_type": "classification",
        })

        assert isinstance(result, str)
        assert "TRAINING FAILED" in result


class TestPredictionWithRandomForest:
    """Test predictions with trained Random Forest models."""

    def test_classification_prediction(self):
        """Test classification predictions."""
        from random_forest import train_random_forest, RandomForestTrainingInput
        from model_storage import predict_with_model_tool, delete_model

        data = get_classification_data()
        model_name = unique_model_name("test_pred_clf")

        try:
            train_random_forest(RandomForestTrainingInput(
                model_name=model_name,
                data=data,
                target_column="fraud",
                task_type="classification",
                n_estimators=10,
            ))

            result = predict_with_model_tool.invoke({
                "model_name": model_name,
                "data": [
                    {"claim_amount": 3000, "age": 50, "prior_claims": 1, "policy_age": 5},
                    {"claim_amount": 55000, "age": 26, "prior_claims": 5, "policy_age": 0},
                ],
            })

            assert isinstance(result, str)
            assert "PREDICTIONS" in result
            assert "Probabilities" in result
        finally:
            delete_model(model_name)

    def test_regression_prediction(self):
        """Test regression predictions."""
        from random_forest import train_random_forest, RandomForestTrainingInput
        from model_storage import predict_with_model_tool, delete_model

        data = get_regression_data()
        model_name = unique_model_name("test_pred_reg")

        try:
            train_random_forest(RandomForestTrainingInput(
                model_name=model_name,
                data=data,
                target_column="claim_amount",
                task_type="regression",
                categorical_columns=["policy_type"],
                n_estimators=10,
            ))

            result = predict_with_model_tool.invoke({
                "model_name": model_name,
                "data": [
                    {"age": 42, "vehicle_age": 2, "policy_type": "premium"},
                    {"age": 32, "vehicle_age": 5, "policy_type": "basic"},
                ],
            })

            assert isinstance(result, str)
            assert "PREDICTIONS" in result
            assert "Predicted Value" in result
            assert "Regression" in result
        finally:
            delete_model(model_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

