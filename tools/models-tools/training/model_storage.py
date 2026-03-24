"""
Model Storage and Registry for Trained Models

Provides utilities for saving, loading, listing, and managing trained models.
Models are registered in Postgres (models + model_versions tables) and weights
are stored in Cloudflare R2 via the ArtifactStore. Falls back to local
filesystem + JSON registry when Postgres is unavailable.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

import joblib
import pandas as pd
from langchain.tools import tool
from pydantic import BaseModel, ConfigDict, Field

# Local staging / cache directory (also used as fallback when DB is down)
TRAINED_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "trained_models"
)
REGISTRY_FILE = os.path.join(TRAINED_MODELS_DIR, "registry.json")


def _ensure_storage_dir():
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)


def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _db_available() -> bool:
    try:
        from backend.shared.database import get_db_session
        return True
    except Exception:
        return False


# ── Postgres-backed helpers ──────────────────────────────────────────────────

def _version_to_dict(
    model: "Model", version: "ModelVersion",  # noqa: F821 – forward refs
) -> dict:
    """Convert DB rows to the legacy dict shape that LangChain tools expect."""
    props = model.properties or {}
    v_props = version.properties or {}
    local_path = os.path.join(TRAINED_MODELS_DIR, f"{_sanitize(model.name)}.joblib")
    return {
        "model_name": model.name,
        "model_path": local_path,
        "model_type": props.get("model_type", ""),
        "description": props.get("description", ""),
        "metrics": version.metrics or {},
        "feature_names": props.get("feature_names", []),
        "target_column": props.get("target_column", ""),
        "hyperparameters": v_props.get("hyperparameters", {}),
        "training_samples": v_props.get("training_samples", 0),
        "classes": v_props.get("classes", []),
        "created_at": model.created_at.isoformat() if model.created_at else "",
        "updated_at": model.updated_at.isoformat() if model.updated_at else "",
        "version": version.version,
        "storage_key": version.storage_key,
    }


# ── JSON file fallback (local dev without Postgres) ─────────────────────────

def _load_registry() -> dict:
    _ensure_storage_dir()
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {"models": {}}


def _save_registry(registry: dict):
    _ensure_storage_dir()
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


# ── Public API (used by training skills + LangChain tools) ──────────────────

def generate_model_path(model_name: str) -> str:
    """Local staging path where the training skill writes the .joblib file."""
    _ensure_storage_dir()
    return os.path.join(TRAINED_MODELS_DIR, f"{_sanitize(model_name)}.joblib")


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
    training_run_id: Optional[str] = None,
) -> dict:
    """Register a trained model, upload weights to R2, and persist to Postgres."""
    if not _db_available():
        return _register_model_fallback(
            model_name, model_path, model_type, description, metrics,
            feature_names, target_column, hyperparameters, training_samples, classes,
        )

    from backend.shared.artifact_store import get_artifact_store
    from backend.shared.database import get_db_session
    from backend.shared.models import Model, ModelVersion

    store = get_artifact_store()

    with get_db_session() as session:
        # UPSERT model identity
        model = session.query(Model).filter(Model.name == model_name).first()
        if model is None:
            model = Model(
                name=model_name,
                properties={
                    "model_type": model_type,
                    "description": description,
                    "feature_names": feature_names,
                    "target_column": target_column,
                },
            )
            session.add(model)
            session.flush()  # get model.id
        else:
            props = model.properties or {}
            props.update({
                "model_type": model_type,
                "description": description,
                "feature_names": feature_names,
                "target_column": target_column,
            })
            model.properties = props

        # Compute next version number
        latest = (
            session.query(ModelVersion)
            .filter(ModelVersion.model_id == model.id)
            .order_by(ModelVersion.version.desc())
            .first()
        )
        next_version = (latest.version + 1) if latest else 1

        # Upload to R2
        storage_key = f"models/{_sanitize(model_name)}/v{next_version}/artifact.joblib"
        if os.path.exists(model_path):
            store.upload(Path(model_path), storage_key)

        # Mark all prior versions as not current
        session.query(ModelVersion).filter(
            ModelVersion.model_id == model.id,
            ModelVersion.is_current.is_(True),
        ).update({"is_current": False})

        # Create new version
        mv = ModelVersion(
            model_id=model.id,
            version=next_version,
            training_run_id=training_run_id,
            storage_key=storage_key,
            metrics=metrics,
            is_current=True,
            properties={
                "hyperparameters": hyperparameters,
                "training_samples": training_samples,
                "classes": classes,
            },
        )
        session.add(mv)
        session.flush()

        return _version_to_dict(model, mv)


def _register_model_fallback(
    model_name, model_path, model_type, description, metrics,
    feature_names, target_column, hyperparameters, training_samples, classes,
) -> dict:
    """JSON-file fallback for local dev without Postgres."""
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
    if model_name in registry["models"]:
        entry["created_at"] = registry["models"][model_name]["created_at"]
    registry["models"][model_name] = entry
    _save_registry(registry)
    return entry


def get_model_info(model_name: str) -> Optional[dict]:
    """Get registry info for a model by name (current version)."""
    if not _db_available():
        registry = _load_registry()
        return registry["models"].get(model_name)

    from backend.shared.database import get_db_session
    from backend.shared.models import Model, ModelVersion

    with get_db_session() as session:
        model = session.query(Model).filter(Model.name == model_name).first()
        if model is None:
            return None
        version = (
            session.query(ModelVersion)
            .filter(ModelVersion.model_id == model.id, ModelVersion.is_current.is_(True))
            .first()
        )
        if version is None:
            version = (
                session.query(ModelVersion)
                .filter(ModelVersion.model_id == model.id)
                .order_by(ModelVersion.version.desc())
                .first()
            )
        if version is None:
            return None
        return _version_to_dict(model, version)


def list_models() -> list[dict]:
    """List all registered models (current versions only)."""
    if not _db_available():
        registry = _load_registry()
        return list(registry["models"].values())

    from backend.shared.database import get_db_session
    from backend.shared.models import Model, ModelVersion

    results = []
    with get_db_session() as session:
        models = session.query(Model).all()
        for model in models:
            version = (
                session.query(ModelVersion)
                .filter(
                    ModelVersion.model_id == model.id,
                    ModelVersion.is_current.is_(True),
                )
                .first()
            )
            if version is None:
                continue
            results.append(_version_to_dict(model, version))
    return results


def load_model(model_name: str) -> Any:
    """Load a trained model by name. Downloads from R2 if not cached locally."""
    info = get_model_info(model_name)
    if info is None:
        raise ValueError(f"Model '{model_name}' not found in registry")

    local_path = info["model_path"]

    # If file doesn't exist locally, try downloading from R2
    if not os.path.exists(local_path) and info.get("storage_key"):
        from backend.shared.artifact_store import get_artifact_store
        store = get_artifact_store()
        store.download(info["storage_key"], Path(local_path))

    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"Model file not found locally or in R2: {local_path}"
        )

    if info.get("model_type") == "pytorch_nn":
        import pickle
        with open(local_path, "rb") as f:
            return pickle.load(f)

    return joblib.load(local_path)


def delete_model(model_name: str) -> bool:
    """Delete a model from registry, R2, and local cache."""
    if not _db_available():
        registry = _load_registry()
        if model_name not in registry["models"]:
            return False
        model_path = registry["models"][model_name]["model_path"]
        if os.path.exists(model_path):
            os.remove(model_path)
        del registry["models"][model_name]
        _save_registry(registry)
        return True

    from backend.shared.artifact_store import get_artifact_store
    from backend.shared.database import get_db_session
    from backend.shared.models import Model, ModelVersion

    store = get_artifact_store()

    with get_db_session() as session:
        model = session.query(Model).filter(Model.name == model_name).first()
        if model is None:
            return False

        # Delete all version artifacts from R2
        versions = (
            session.query(ModelVersion)
            .filter(ModelVersion.model_id == model.id)
            .all()
        )
        for v in versions:
            if v.storage_key:
                try:
                    store.delete(v.storage_key)
                except Exception:
                    pass

        # Delete from Postgres (CASCADE deletes versions)
        session.delete(model)

    # Delete local cache
    local_path = os.path.join(TRAINED_MODELS_DIR, f"{_sanitize(model_name)}.joblib")
    if os.path.exists(local_path):
        os.remove(local_path)

    return True


# =============================================================================
# LangChain Tool Implementations
# =============================================================================


class ListModelsInput(BaseModel):
    """Input for listing trained models."""
    model_config = ConfigDict(extra="forbid")
    
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
    model_config = ConfigDict(extra="forbid")
    
    model_name: str = Field(
        description="Name of the model to load (as shown in list_trained_models)"
    )


class PredictInput(BaseModel):
    """Input for making predictions with a trained model."""
    model_config = ConfigDict(extra="forbid")
    
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
    model_config = ConfigDict(extra="forbid")
    
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
    model_config = ConfigDict(extra="forbid")
    
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
    model_config = ConfigDict(extra="forbid")
    
    model_name: str = Field(
        description="Name of the trained model to evaluate"
    )
    dataset_ref: str = Field(
        description="Reference name of the registered dataset to evaluate on (e.g., 'val_data', 'test_data')"
    )
    target_column: str = Field(
        description="Name of the target column in the dataset"
    )
    # Threshold tuning options
    optimize_threshold: bool = Field(
        default=False,
        description="If True, find optimal classification threshold instead of using 0.5. "
                    "Useful for imbalanced data. Only works for binary classification."
    )
    optimize_for: Literal["f1", "precision", "recall", "balanced_accuracy"] = Field(
        default="f1",
        description="Metric to optimize when finding threshold: 'f1', 'precision', 'recall', or 'balanced_accuracy'"
    )
    min_precision: Optional[float] = Field(
        default=None,
        description="Minimum precision constraint when optimizing threshold (e.g., 0.3 means precision >= 30%)"
    )
    min_recall: Optional[float] = Field(
        default=None,
        description="Minimum recall constraint when optimizing threshold (e.g., 0.5 means recall >= 50%)"
    )


@tool("evaluate_model", args_schema=EvaluateModelInput)
def evaluate_model_tool(
    model_name: str,
    dataset_ref: str,
    target_column: str,
    optimize_threshold: bool = False,
    optimize_for: Literal["f1", "precision", "recall", "balanced_accuracy"] = "f1",
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
) -> str:
    """Evaluate a trained model on a registered dataset and return metrics.
    
    WHAT IT DOES:
    Loads a trained model and a registered dataset, makes predictions,
    and computes comprehensive classification/regression metrics.
    
    WHEN TO USE:
    - After training, evaluate on validation data to check performance
    - Evaluate on test data for final metrics
    - Compare different models on the same dataset
    
    THRESHOLD TUNING (classification only):
    Set optimize_threshold=True to find the optimal decision threshold instead of 0.5.
    This is valuable for imbalanced data where 0.5 is often suboptimal.
    - optimize_for="f1": Maximize F1 score (balance precision/recall)
    - optimize_for="recall": Maximize recall (catch more positives)
    - optimize_for="precision": Maximize precision (fewer false positives)
    - min_precision/min_recall: Add constraints (e.g., "maximize F1 but keep precision >= 0.3")
    
    METRICS COMPUTED:
    - For classification: accuracy, ROC-AUC, Brier score, precision, recall, F1, confusion matrix
    - For regression: MSE, RMSE, MAE, R²
    
    Args:
        model_name: Name of the trained model in the registry
        dataset_ref: Reference name of the registered dataset (must be registered via register_dataset)
        target_column: Name of the target column to evaluate against
        optimize_threshold: If True, find optimal threshold (binary classification only)
        optimize_for: Metric to optimize ('f1', 'precision', 'recall', 'balanced_accuracy')
        min_precision: Optional minimum precision constraint
        min_recall: Optional minimum recall constraint
    
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
    
    # Apply target discretization if model was trained with auto-discretized target
    hp = (info.get("hyperparameters") or {})
    if hp.get("target_discretized") and hp.get("target_discretization_threshold") is not None:
        import numpy as np
        threshold = float(hp["target_discretization_threshold"])
        y_true = (y_true > threshold).astype(int)
    
    # Make predictions
    try:
        y_pred = model.predict(X)
        
        # Check if classification or regression based on the actual target values,
        # not model attributes (XGBRegressor can have predict_proba/classes_ inherited).
        from sklearn.utils.multiclass import type_of_target
        target_type = type_of_target(y_true)
        is_classification = target_type in ("binary", "multiclass", "multiclass-multioutput")
        
        if is_classification:
            # Classification metrics
            from sklearn.metrics import (
                accuracy_score, roc_auc_score, precision_score, recall_score,
                f1_score, confusion_matrix, classification_report, balanced_accuracy_score,
                brier_score_loss,
            )
            import numpy as np
            
            # Get probabilities if available
            y_proba = None
            roc_auc = None
            brier = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X)
                    if y_proba.shape[1] == 2:
                        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
                        brier = brier_score_loss(y_true, y_proba[:, 1])
                    else:
                        roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except Exception:
                    pass
            
            # Threshold optimization for binary classification
            optimal_threshold = 0.5
            threshold_results = None
            if optimize_threshold and y_proba is not None and y_proba.shape[1] == 2:
                # Sweep thresholds
                thresholds = np.arange(0.05, 0.96, 0.01)
                best_score = -1
                best_thresh = 0.5
                
                for thresh in thresholds:
                    y_pred_thresh = (y_proba[:, 1] >= thresh).astype(int)
                    
                    prec = precision_score(y_true, y_pred_thresh, zero_division=0)
                    rec = recall_score(y_true, y_pred_thresh, zero_division=0)
                    
                    # Check constraints
                    if min_precision is not None and prec < min_precision:
                        continue
                    if min_recall is not None and rec < min_recall:
                        continue
                    
                    # Calculate target metric
                    if optimize_for == "f1":
                        score = f1_score(y_true, y_pred_thresh, zero_division=0)
                    elif optimize_for == "precision":
                        score = prec
                    elif optimize_for == "recall":
                        score = rec
                    elif optimize_for == "balanced_accuracy":
                        score = balanced_accuracy_score(y_true, y_pred_thresh)
                    else:
                        score = f1_score(y_true, y_pred_thresh, zero_division=0)
                    
                    if score > best_score:
                        best_score = score
                        best_thresh = thresh
                
                optimal_threshold = best_thresh
                y_pred = (y_proba[:, 1] >= optimal_threshold).astype(int)
                
                # Metrics at optimal threshold
                threshold_results = {
                    "optimal_threshold": optimal_threshold,
                    "optimized_for": optimize_for,
                    "score_at_threshold": best_score,
                }
            
            accuracy = accuracy_score(y_true, y_pred)
            
            # Precision, recall, F1
            try:
                precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                # Also get binary metrics for positive class
                prec_pos = precision_score(y_true, y_pred, average='binary', zero_division=0)
                rec_pos = recall_score(y_true, y_pred, average='binary', zero_division=0)
                f1_pos = f1_score(y_true, y_pred, average='binary', zero_division=0)
            except Exception:
                precision = recall = f1 = None
                prec_pos = rec_pos = f1_pos = None
            
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
            ]
            
            # Show threshold tuning results
            if threshold_results:
                lines.extend([
                    "🎯 THRESHOLD OPTIMIZATION",
                    f"  Optimized for: {threshold_results['optimized_for']}",
                    f"  Optimal threshold: {threshold_results['optimal_threshold']:.2f} (default was 0.50)",
                    f"  Score at threshold: {threshold_results['score_at_threshold']:.4f}",
                ])
                if min_precision:
                    lines.append(f"  Constraint: precision >= {min_precision}")
                if min_recall:
                    lines.append(f"  Constraint: recall >= {min_recall}")
                lines.append("")
            
            lines.extend([
                "📈 CLASSIFICATION METRICS",
                f"  Accuracy: {accuracy:.4f}",
                f"  ROC-AUC: {roc_auc:.4f}" if roc_auc else "  ROC-AUC: N/A",
                f"  Brier Score (lower is better): {brier:.4f}" if brier is not None else "  Brier Score: N/A",
                f"  Precision (weighted): {precision:.4f}" if precision else "  Precision: N/A",
                f"  Recall (weighted): {recall:.4f}" if recall else "  Recall: N/A",
                f"  F1 Score (weighted): {f1:.4f}" if f1 else "  F1: N/A",
            ])
            
            if prec_pos is not None:
                lines.extend([
                    "",
                    "📈 POSITIVE CLASS METRICS",
                    f"  Precision: {prec_pos:.4f}",
                    f"  Recall: {rec_pos:.4f}",
                    f"  F1: {f1_pos:.4f}",
                ])
            
            lines.extend([
                "",
                "🔢 CONFUSION MATRIX",
                str(cm),
                "",
                "📋 CLASSIFICATION REPORT",
                report,
                "",
                "=" * 60,
            ])
            
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

