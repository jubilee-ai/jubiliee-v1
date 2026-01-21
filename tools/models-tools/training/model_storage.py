"""
Model Storage and Registry for Trained Models

Provides utilities for saving, loading, listing, and managing trained models.
All models are stored in the trained_models/ directory with a JSON registry
that tracks metadata for each model.
"""

import json
import os
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import joblib
import pandas as pd
from langchain.tools import tool
from pydantic import BaseModel, Field

# Default storage location
TRAINED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "trained_models"
)
REGISTRY_FILE = os.path.join(TRAINED_MODELS_DIR, "registry.json")


def _ensure_storage_dir():
    """Ensure the trained_models directory exists."""
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


def _load_registry() -> dict:
    """Load the model registry from disk."""
    _ensure_storage_dir()
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"models": {}}


def _save_registry(registry: dict):
    """Save the model registry to disk."""
    _ensure_storage_dir()
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def generate_model_path(model_name: str) -> str:
    """Generate a file path for a model in the trained_models directory."""
    _ensure_storage_dir()
    # Sanitize model name for filesystem
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
    return os.path.join(TRAINED_MODELS_DIR, f"{safe_name}.joblib")


def register_model(
    model_name: str,
    model_path: str,
    model_type: str,
    description: str,
    metrics: dict,
    feature_names: list[str],
    target_column: str,
    hyperparameters: dict,
    training_samples: int,
    classes: list[str],
) -> dict:
    """
    Register a trained model in the registry.
    
    Returns the registry entry for the model.
    """
    registry = _load_registry()
    
    entry = {
        "model_name": model_name,
        "model_path": model_path,
        "model_type": model_type,
        "description": description,
        "metrics": metrics,
        "feature_names": feature_names,
        "target_column": target_column,
        "hyperparameters": hyperparameters,
        "training_samples": training_samples,
        "classes": classes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # If model already exists, preserve created_at
    if model_name in registry["models"]:
        entry["created_at"] = registry["models"][model_name]["created_at"]
    
    registry["models"][model_name] = entry
    _save_registry(registry)
    
    return entry


def get_model_info(model_name: str) -> Optional[dict]:
    """Get registry info for a model by name."""
    registry = _load_registry()
    return registry["models"].get(model_name)


def list_models() -> list[dict]:
    """List all registered models."""
    registry = _load_registry()
    return list(registry["models"].values())


def load_model(model_name: str) -> Any:
    """Load a trained model by name."""
    info = get_model_info(model_name)
    if info is None:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    model_path = info["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)


def delete_model(model_name: str) -> bool:
    """Delete a model from registry and disk."""
    registry = _load_registry()
    
    if model_name not in registry["models"]:
        return False
    
    # Delete file if exists
    model_path = registry["models"][model_name]["model_path"]
    if os.path.exists(model_path):
        os.remove(model_path)
    
    # Remove from registry
    del registry["models"][model_name]
    _save_registry(registry)
    
    return True


# =============================================================================
# LangChain Tool Implementations
# =============================================================================


class ListModelsInput(BaseModel):
    """Input for listing trained models."""
    model_type: Optional[str] = Field(
        default=None,
        description="Filter by model type (e.g., 'sklearn_logistic_regression'). If None, lists all models."
    )


@tool("list_trained_models", args_schema=ListModelsInput)
def list_trained_models_tool(
    model_type: Optional[str] = None,
) -> str:
    """List all trained models available in the model registry.
    
    WHEN TO USE:
    - Before training: Check if a model already exists for your use case
    - Before inference: Find the right model to use for predictions
    - Model management: Review what models have been trained
    
    OUTPUT:
    - Model name, type, and description
    - Performance metrics (accuracy, ROC-AUC)
    - Training details (samples, features, classes)
    - Creation timestamp
    
    Args:
        model_type: Optional filter by model type (e.g., 'sklearn_logistic_regression')
    """
    models = list_models()
    
    if model_type:
        models = [m for m in models if m["model_type"] == model_type]
    
    if not models:
        if model_type:
            return f"No trained models found with type '{model_type}'.\n\nUse a training tool to create one."
        return "No trained models found.\n\nUse a training tool (e.g., sklearn_logistic_regression) to train and save a model."
    
    # Sort by updated_at descending
    models.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    
    lines = [
        "=" * 60,
        f"TRAINED MODELS ({len(models)} found)",
        "=" * 60,
        "",
    ]
    
    for m in models:
        metrics = m.get("metrics", {})
        hyperparams = m.get("hyperparameters", {})
        task_type = hyperparams.get("task_type", "classification")
        
        lines.extend([
            f"📦 {m['model_name']}",
            f"   Type: {m['model_type']}",
            f"   Description: {m.get('description', 'N/A')}",
            f"   Target: {m.get('target_column', 'N/A')}",
        ])
        
        # Show classes only for classification
        if m.get('classes'):
            lines.append(f"   Classes: {m.get('classes', [])}")
        
        lines.append(f"   Samples: {m.get('training_samples', 'N/A')} | Features: {len(m.get('feature_names', []))}")
        
        # Show appropriate metrics based on task type
        if task_type == "regression":
            test_score = metrics.get('test_score')
            lines.append(f"   Test R²: {test_score:.4f}" if isinstance(test_score, (int, float)) else "   Test R²: N/A")
            mae = metrics.get('mae')
            if mae is not None:
                lines.append(f"   Test MAE: {mae:.4f}")
        else:
            # Classification metrics
            test_acc = metrics.get('test_accuracy') or metrics.get('test_score')
            lines.append(f"   Accuracy: {test_acc:.4f}" if isinstance(test_acc, (int, float)) else "   Accuracy: N/A")
            roc_auc = metrics.get('test_roc_auc') or metrics.get('roc_auc')
            lines.append(f"   ROC-AUC: {roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else "   ROC-AUC: N/A")
        
        lines.extend([
            f"   Updated: {m.get('updated_at', 'N/A')}",
            "",
        ])
    
    lines.append("=" * 60)
    return "\n".join(lines)


class LoadModelInput(BaseModel):
    """Input for loading a trained model."""
    model_name: str = Field(
        description="Name of the model to load (as shown in list_trained_models)"
    )


class PredictInput(BaseModel):
    """Input for making predictions with a trained model."""
    model_name: str = Field(
        description="Name of the trained model to use for predictions"
    )
    dataset_ref: str = Field(
        description="Reference name of the registered dataset to predict on. "
                    "The dataset must be registered via register_dataset(). "
                    "Must have the same feature columns as training data."
    )


def _load_dataset_from_ref(dataset_ref: str) -> pd.DataFrame:
    """Load a dataset from the registry by reference name."""
    import sys
    from pathlib import Path
    
    data_tools_path = str(Path(__file__).parent.parent.parent / "data-tools")
    if data_tools_path not in sys.path:
        sys.path.insert(0, data_tools_path)
    
    from utils import get_registered_dataset
    
    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry.")
    return df


@tool("predict_with_model", args_schema=PredictInput)
def predict_with_model_tool(
    model_name: str,
    dataset_ref: str,
) -> str:
    """Make predictions using a trained model from the registry.
    
    WHEN TO USE:
    - Inference: Apply a trained model to new data
    - Batch scoring: Predict on multiple samples at once
    - Model testing: Validate model on held-out data
    
    REQUIREMENTS:
    - Model must exist in registry (use list_trained_models to check)
    - Input data must have same feature columns as training data
    - Categorical values must match training categories
    
    OUTPUT:
    - Prediction for each row
    - Class probabilities (classification) or predicted values (regression)
    - Model info used for predictions
    
    Args:
        model_name: Name of the model to use
        dataset_ref: Reference name of registered dataset to predict on
    """
    # Get model info
    info = get_model_info(model_name)
    if info is None:
        available = [m["model_name"] for m in list_models()]
        return (
            f"❌ Model '{model_name}' not found.\n\n"
            f"Available models: {available if available else 'None - train a model first'}"
        )
    
    try:
        # Load model
        model = load_model(model_name)
        
        # Load dataset from registry
        df = _load_dataset_from_ref(dataset_ref)
        data = df.to_dict(orient="records")
        
        # Make predictions
        predictions = model.predict(df)
        
        # Check if this is a classification or regression model
        hyperparams = info.get("hyperparameters", {})
        task_type = hyperparams.get("task_type", "classification")
        is_classifier = hasattr(model, "predict_proba") and hasattr(model, "classes_")
        
        # Format output based on model type
        lines = [
            "=" * 60,
            f"PREDICTIONS: {model_name}",
            "=" * 60,
            f"Model Type: {info['model_type']}",
            f"Target: {info['target_column']}",
        ]
        
        if is_classifier:
            # Classification model
            probabilities = model.predict_proba(df)
            classes = model.classes_
            lines.append(f"Classes: {list(classes)}")
            lines.extend(["", "-" * 60])
            
            for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
                prob_str = ", ".join(f"{c}: {p:.1%}" for c, p in zip(classes, probs))
                lines.extend([
                    f"Row {i + 1}:",
                    f"  Input: {data[i]}",
                    f"  Prediction: {pred}",
                    f"  Probabilities: {prob_str}",
                    "",
                ])
        else:
            # Regression model
            lines.append(f"Task: Regression")
            lines.extend(["", "-" * 60])
            
            for i, pred in enumerate(predictions):
                lines.extend([
                    f"Row {i + 1}:",
                    f"  Input: {data[i]}",
                    f"  Predicted Value: {pred:.4f}" if isinstance(pred, float) else f"  Predicted Value: {pred}",
                    "",
                ])
        
        lines.append("=" * 60)
        return "\n".join(lines)
        
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"


class GetModelInfoInput(BaseModel):
    """Input for getting detailed model information."""
    model_name: str = Field(
        description="Name of the model to get info for"
    )


@tool("get_model_info", args_schema=GetModelInfoInput)
def get_model_info_tool(
    model_name: str,
) -> str:
    """Get detailed information about a trained model.
    
    WHEN TO USE:
    - Understanding a model before using it
    - Checking hyperparameters and features
    - Reviewing model performance
    
    OUTPUT:
    - Full model metadata
    - Hyperparameters used
    - Feature names
    - Performance metrics
    
    Args:
        model_name: Name of the model
    """
    info = get_model_info(model_name)
    
    if info is None:
        available = [m["model_name"] for m in list_models()]
        return (
            f"❌ Model '{model_name}' not found.\n\n"
            f"Available models: {available if available else 'None'}"
        )
    
    metrics = info.get("metrics", {})
    hyperparams = info.get("hyperparameters", {})
    
    lines = [
        "=" * 60,
        f"MODEL INFO: {model_name}",
        "=" * 60,
        "",
        "📋 BASIC INFO",
        f"  Type: {info['model_type']}",
        f"  Description: {info.get('description', 'N/A')}",
        f"  Path: {info['model_path']}",
        f"  Created: {info.get('created_at', 'N/A')}",
        f"  Updated: {info.get('updated_at', 'N/A')}",
        "",
        "🎯 TARGET",
        f"  Column: {info.get('target_column', 'N/A')}",
        f"  Classes: {info.get('classes', [])}",
        "",
        "📊 TRAINING DATA",
        f"  Samples: {info.get('training_samples', 'N/A')}",
        f"  Features: {len(info.get('feature_names', []))}",
        "",
        "📈 PERFORMANCE",
        f"  Train Accuracy: {metrics.get('train_accuracy', 'N/A')}",
        f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A')}",
        f"  Test ROC-AUC: {metrics.get('test_roc_auc', 'N/A')}",
        "",
        "⚙️ HYPERPARAMETERS",
    ]
    
    for k, v in hyperparams.items():
        lines.append(f"  {k}: {v}")
    
    lines.extend([
        "",
        "🔢 FEATURES",
        "  " + ", ".join(info.get("feature_names", [])[:10]),
    ])
    
    if len(info.get("feature_names", [])) > 10:
        lines.append(f"  ... and {len(info['feature_names']) - 10} more")
    
    lines.extend(["", "=" * 60])
    return "\n".join(lines)


class DeleteModelInput(BaseModel):
    """Input for deleting a trained model."""
    model_name: str = Field(
        description="Name of the model to delete"
    )


@tool("delete_trained_model", args_schema=DeleteModelInput)
def delete_trained_model_tool(
    model_name: str,
) -> str:
    """Delete a trained model from the registry and disk.
    
    WHEN TO USE:
    - Cleaning up old/unused models
    - Removing failed experiments
    - Freeing disk space
    
    WARNING: This permanently deletes the model file and registry entry.
    
    Args:
        model_name: Name of the model to delete
    """
    info = get_model_info(model_name)
    
    if info is None:
        return f"❌ Model '{model_name}' not found in registry."
    
    success = delete_model(model_name)
    
    if success:
        return f"✅ Model '{model_name}' deleted successfully."
    else:
        return f"❌ Failed to delete model '{model_name}'."


# =============================================================================
# EVALUATE MODEL TOOL
# =============================================================================

class EvaluateModelInput(BaseModel):
    """Input for evaluating a model on a dataset."""
    model_name: str = Field(
        description="Name of the trained model to evaluate"
    )
    dataset_ref: str = Field(
        description="Reference name of the registered dataset to evaluate on (e.g., 'val_data', 'test_data')"
    )
    target_column: str = Field(
        description="Name of the target column in the dataset"
    )


@tool("evaluate_model", args_schema=EvaluateModelInput)
def evaluate_model_tool(
    model_name: str,
    dataset_ref: str,
    target_column: str,
) -> str:
    """Evaluate a trained model on a registered dataset and return metrics.
    
    WHAT IT DOES:
    Loads a trained model and a registered dataset, makes predictions,
    and computes comprehensive classification/regression metrics.
    
    WHEN TO USE:
    - After training, evaluate on validation data to check performance
    - Evaluate on test data for final metrics
    - Compare different models on the same dataset
    
    METRICS COMPUTED:
    - For classification: accuracy, ROC-AUC, precision, recall, F1, confusion matrix
    - For regression: MSE, RMSE, MAE, R²
    
    Args:
        model_name: Name of the trained model in the registry
        dataset_ref: Reference name of the registered dataset (must be registered via register_dataset)
        target_column: Name of the target column to evaluate against
    
    Returns:
        Formatted string with evaluation metrics
    """
    import sys
    from pathlib import Path
    
    # Add data-tools path
    data_tools_path = str(Path(__file__).parent.parent.parent / "data-tools")
    if data_tools_path not in sys.path:
        sys.path.insert(0, data_tools_path)
    
    from utils import get_registered_dataset
    
    # Load model info
    info = get_model_info(model_name)
    if info is None:
        return f"❌ Model '{model_name}' not found in registry."
    
    # Load model
    model = load_model(model_name)
    if model is None:
        return f"❌ Failed to load model '{model_name}'."
    
    # Load dataset
    df = get_registered_dataset(dataset_ref)
    if df is None:
        return f"❌ Dataset '{dataset_ref}' not found in registry."
    
    # Check target column exists
    if target_column not in df.columns:
        return f"❌ Target column '{target_column}' not found in dataset. Available: {list(df.columns)}"
    
    # Prepare data
    y_true = df[target_column]
    feature_cols = [c for c in df.columns if c != target_column]
    X = df[feature_cols]
    
    # Make predictions
    try:
        y_pred = model.predict(X)
        
        # Check if classification or regression
        is_classification = hasattr(model, 'predict_proba') or hasattr(model, 'classes_')
        
        if is_classification:
            # Classification metrics
            from sklearn.metrics import (
                accuracy_score, roc_auc_score, precision_score, recall_score,
                f1_score, confusion_matrix, classification_report
            )
            
            accuracy = accuracy_score(y_true, y_pred)
            
            # Get probabilities if available
            roc_auc = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X)
                    if y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except Exception:
                    pass
            
            # Precision, recall, F1
            try:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            except Exception:
                precision = recall = f1 = None
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Classification report
            report = classification_report(y_true, y_pred, zero_division=0)
            
            lines = [
                "=" * 60,
                f"EVALUATION: {model_name} on {dataset_ref}",
                "=" * 60,
                "",
                "📊 DATASET INFO",
                f"  Samples: {len(df)}",
                f"  Features: {len(feature_cols)}",
                f"  Target: {target_column}",
                "",
                "📈 CLASSIFICATION METRICS",
                f"  Accuracy: {accuracy:.4f}",
                f"  ROC-AUC: {roc_auc:.4f}" if roc_auc else "  ROC-AUC: N/A",
                f"  Precision (weighted): {precision:.4f}" if precision else "  Precision: N/A",
                f"  Recall (weighted): {recall:.4f}" if recall else "  Recall: N/A",
                f"  F1 Score (weighted): {f1:.4f}" if f1 else "  F1: N/A",
                "",
                "🔢 CONFUSION MATRIX",
                str(cm),
                "",
                "📋 CLASSIFICATION REPORT",
                report,
                "",
                "=" * 60,
            ]
            
            return "\n".join(lines)
            
        else:
            # Regression metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            import numpy as np
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            lines = [
                "=" * 60,
                f"EVALUATION: {model_name} on {dataset_ref}",
                "=" * 60,
                "",
                "📊 DATASET INFO",
                f"  Samples: {len(df)}",
                f"  Features: {len(feature_cols)}",
                f"  Target: {target_column}",
                "",
                "📈 REGRESSION METRICS",
                f"  MSE: {mse:.4f}",
                f"  RMSE: {rmse:.4f}",
                f"  MAE: {mae:.4f}",
                f"  R²: {r2:.4f}",
                "",
                "=" * 60,
            ]
            
            return "\n".join(lines)
            
    except Exception as e:
        return f"❌ Evaluation failed: {str(e)}"

