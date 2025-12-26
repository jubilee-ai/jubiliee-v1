"""Tests for the sklearn Logistic Regression training tool."""

import os
import sys
import tempfile
import uuid

import pytest

# Add training module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models-tools', 'training'))


def get_sample_classification_data():
    """Generate sample binary classification data for testing."""
    return [
        {"age": 25, "income": 30000, "employed": "yes", "default": 0},
        {"age": 35, "income": 75000, "employed": "yes", "default": 0},
        {"age": 45, "income": 120000, "employed": "yes", "default": 0},
        {"age": 28, "income": 55000, "employed": "yes", "default": 0},
        {"age": 52, "income": 90000, "employed": "yes", "default": 0},
        {"age": 31, "income": 42000, "employed": "yes", "default": 0},
        {"age": 29, "income": 38000, "employed": "no", "default": 1},
        {"age": 22, "income": 25000, "employed": "no", "default": 1},
        {"age": 40, "income": 28000, "employed": "no", "default": 1},
        {"age": 33, "income": 32000, "employed": "no", "default": 1},
        {"age": 27, "income": 29000, "employed": "yes", "default": 1},
        {"age": 38, "income": 35000, "employed": "no", "default": 1},
        {"age": 55, "income": 150000, "employed": "yes", "default": 0},
        {"age": 48, "income": 95000, "employed": "yes", "default": 0},
        {"age": 23, "income": 22000, "employed": "no", "default": 1},
        {"age": 36, "income": 68000, "employed": "yes", "default": 0},
        {"age": 42, "income": 85000, "employed": "yes", "default": 0},
        {"age": 30, "income": 45000, "employed": "yes", "default": 0},
        {"age": 26, "income": 27000, "employed": "no", "default": 1},
        {"age": 44, "income": 78000, "employed": "yes", "default": 0},
        {"age": 50, "income": 110000, "employed": "yes", "default": 0},
        {"age": 34, "income": 31000, "employed": "no", "default": 1},
        {"age": 39, "income": 72000, "employed": "yes", "default": 0},
        {"age": 24, "income": 26000, "employed": "no", "default": 1},
        {"age": 47, "income": 88000, "employed": "yes", "default": 0},
    ]


def get_multiclass_data():
    """Generate sample multiclass data for testing."""
    return [
        {"feature1": 1.0, "feature2": 2.0, "category": "A"},
        {"feature1": 1.5, "feature2": 1.8, "category": "A"},
        {"feature1": 1.2, "feature2": 2.1, "category": "A"},
        {"feature1": 5.0, "feature2": 5.5, "category": "B"},
        {"feature1": 5.2, "feature2": 5.0, "category": "B"},
        {"feature1": 4.8, "feature2": 5.3, "category": "B"},
        {"feature1": 9.0, "feature2": 9.5, "category": "C"},
        {"feature1": 8.8, "feature2": 9.2, "category": "C"},
        {"feature1": 9.2, "feature2": 9.0, "category": "C"},
        {"feature1": 1.1, "feature2": 2.2, "category": "A"},
        {"feature1": 5.1, "feature2": 5.1, "category": "B"},
        {"feature1": 8.9, "feature2": 9.1, "category": "C"},
    ]


def unique_model_name(prefix="test_model"):
    """Generate unique model name for testing."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class TestLogisticRegressionTraining:
    """Test suite for logistic regression training function."""
    
    def test_basic_training(self):
        """Test basic binary classification training."""
        from logistic_regression import (
            train_logistic_regression,
            LogisticRegressionTrainingInput,
        )
        from model_storage import delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_basic")
        
        try:
            result = train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                data=data,
                target_column="default",
                categorical_columns=["employed"],
            ))
            
            assert result.success is True
            assert result.n_samples == 25
            assert result.n_classes == 2
            assert result.test_accuracy >= 0.0 and result.test_accuracy <= 1.0
            assert result.train_accuracy >= 0.0 and result.train_accuracy <= 1.0
            assert len(result.feature_names) > 0
            assert result.n_iterations > 0
            assert result.model_name == model_name
        finally:
            delete_model(model_name)
    
    def test_training_with_regularization(self):
        """Test training with L1 regularization."""
        from logistic_regression import (
            train_logistic_regression,
            LogisticRegressionTrainingInput,
        )
        from model_storage import delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_l1")
        
        try:
            result = train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                data=data,
                target_column="default",
                categorical_columns=["employed"],
                l1_ratio=1.0,  # Pure L1
                solver="saga",
                C=0.1,
            ))
            
            assert result.success is True
        finally:
            delete_model(model_name)
    
    def test_training_with_balanced_weights(self):
        """Test training with balanced class weights."""
        from logistic_regression import (
            train_logistic_regression,
            LogisticRegressionTrainingInput,
        )
        from model_storage import delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_balanced")
        
        try:
            result = train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                data=data,
                target_column="default",
                categorical_columns=["employed"],
                class_weight="balanced",
            ))
            
            assert result.success is True
        finally:
            delete_model(model_name)
    
    def test_multiclass_training(self):
        """Test multiclass classification."""
        from logistic_regression import (
            train_logistic_regression,
            LogisticRegressionTrainingInput,
        )
        from model_storage import delete_model
        
        data = get_multiclass_data()
        model_name = unique_model_name("test_multiclass")
        
        try:
            result = train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                data=data,
                target_column="category",
                test_size=0.25,
            ))
            
            assert result.success is True
            assert result.n_classes == 3
        finally:
            delete_model(model_name)
    
    def test_model_saved_and_registered(self):
        """Test that models are saved and registered correctly."""
        from logistic_regression import (
            train_logistic_regression,
            LogisticRegressionTrainingInput,
        )
        from model_storage import get_model_info, load_model, delete_model
        import pandas as pd
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_save")
        
        try:
            result = train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                description="Test model for unit tests",
                data=data,
                target_column="default",
                categorical_columns=["employed"],
            ))
            
            # Check model was saved
            assert os.path.exists(result.saved_path)
            
            # Check model is in registry
            info = get_model_info(model_name)
            assert info is not None
            assert info["model_name"] == model_name
            assert info["description"] == "Test model for unit tests"
            
            # Check model can be loaded and used
            model = load_model(model_name)
            test_sample = pd.DataFrame([{"age": 30, "income": 50000, "employed": "yes"}])
            prediction = model.predict(test_sample)
            assert prediction[0] in [0, 1]
        finally:
            delete_model(model_name)
    
    def test_invalid_target_column(self):
        """Test error handling for invalid target column."""
        from logistic_regression import (
            train_logistic_regression,
            LogisticRegressionTrainingInput,
        )
        
        data = get_sample_classification_data()
        
        with pytest.raises(ValueError, match="Target column"):
            train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=unique_model_name("test_invalid"),
                data=data,
                target_column="nonexistent",
            ))
    
    def test_auto_detect_categorical(self):
        """Test automatic detection of categorical columns."""
        from logistic_regression import (
            train_logistic_regression,
            LogisticRegressionTrainingInput,
        )
        from model_storage import delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_autodetect")
        
        try:
            # Don't specify categorical_columns - should auto-detect 'employed'
            result = train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                data=data,
                target_column="default",
            ))
            
            assert result.success is True
            # Should have encoded 'employed' column
            assert any("employed" in f.lower() for f in result.feature_names)
        finally:
            delete_model(model_name)


class TestLogisticRegressionTool:
    """Test suite for the LangChain tool wrapper."""
    
    def test_tool_execution(self):
        """Test the LangChain tool returns formatted string."""
        from logistic_regression import sklearn_logistic_regression_tool
        from model_storage import delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_tool")
        
        try:
            result = sklearn_logistic_regression_tool.invoke({
                "model_name": model_name,
                "data": data,
                "target_column": "default",
                "categorical_columns": ["employed"],
            })
            
            assert isinstance(result, str)
            assert "LOGISTIC REGRESSION TRAINING COMPLETE" in result
            assert "Train Accuracy" in result
            assert "Test Accuracy" in result
            assert "MODEL REGISTERED" in result
        finally:
            delete_model(model_name)
    
    def test_tool_error_handling(self):
        """Test the tool handles errors gracefully."""
        from logistic_regression import sklearn_logistic_regression_tool
        
        result = sklearn_logistic_regression_tool.invoke({
            "model_name": unique_model_name("test_error"),
            "data": [{"a": 1}],  # Invalid: no target column
            "target_column": "nonexistent",
        })
        
        assert isinstance(result, str)
        assert "TRAINING FAILED" in result


class TestModelStorageTools:
    """Test suite for model storage tools."""
    
    def test_list_models(self):
        """Test listing trained models."""
        from logistic_regression import train_logistic_regression, LogisticRegressionTrainingInput
        from model_storage import list_trained_models_tool, delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_list")
        
        try:
            # Train a model
            train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                data=data,
                target_column="default",
            ))
            
            # List models
            result = list_trained_models_tool.invoke({})
            
            assert isinstance(result, str)
            assert model_name in result
        finally:
            delete_model(model_name)
    
    def test_predict_with_model(self):
        """Test making predictions with a saved model."""
        from logistic_regression import train_logistic_regression, LogisticRegressionTrainingInput
        from model_storage import predict_with_model_tool, delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_predict")
        
        try:
            # Train a model
            train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                data=data,
                target_column="default",
                categorical_columns=["employed"],
            ))
            
            # Make predictions
            result = predict_with_model_tool.invoke({
                "model_name": model_name,
                "data": [
                    {"age": 40, "income": 80000, "employed": "yes"},
                    {"age": 25, "income": 25000, "employed": "no"},
                ],
            })
            
            assert isinstance(result, str)
            assert "PREDICTIONS" in result
            assert "Row 1" in result
            assert "Row 2" in result
        finally:
            delete_model(model_name)
    
    def test_get_model_info(self):
        """Test getting model info."""
        from logistic_regression import train_logistic_regression, LogisticRegressionTrainingInput
        from model_storage import get_model_info_tool, delete_model
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_info")
        
        try:
            # Train a model
            train_logistic_regression(LogisticRegressionTrainingInput(
                model_name=model_name,
                description="Test model description",
                data=data,
                target_column="default",
            ))
            
            # Get info
            result = get_model_info_tool.invoke({"model_name": model_name})
            
            assert isinstance(result, str)
            assert model_name in result
            assert "Test model description" in result
            assert "HYPERPARAMETERS" in result
        finally:
            delete_model(model_name)
    
    def test_delete_model(self):
        """Test deleting a model."""
        from logistic_regression import train_logistic_regression, LogisticRegressionTrainingInput
        from model_storage import delete_trained_model_tool, get_model_info
        
        data = get_sample_classification_data()
        model_name = unique_model_name("test_delete")
        
        # Train a model
        train_logistic_regression(LogisticRegressionTrainingInput(
            model_name=model_name,
            data=data,
            target_column="default",
        ))
        
        # Verify it exists
        assert get_model_info(model_name) is not None
        
        # Delete it
        result = delete_trained_model_tool.invoke({"model_name": model_name})
        
        assert "deleted successfully" in result
        assert get_model_info(model_name) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
