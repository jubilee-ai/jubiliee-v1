"""
XGBoost Training Tool for AI Data Scientist Agent

A LangChain tool that dynamically trains XGBoost models for classification
or regression tasks. Features iterative training with loss curves, early
stopping, and feature importance. Excellent for tabular underwriting,
fraud detection, and risk scoring tasks.
"""

import os
from datetime import datetime, timezone
from typing import Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from langchain.tools import tool
from model_storage import generate_model_path, register_model
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class XGBoostPipelineWrapper:
    """Wrapper to combine preprocessing and XGBoost model for serialization."""
    
    def __init__(self, preprocessor, model, label_encoder=None):
        self.preprocessor = preprocessor
        self.model = model
        self.label_encoder = label_encoder
        self.classes_ = model.classes_ if hasattr(model, 'classes_') else None
    
    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)


class XGBoostTrainingInput(BaseModel):
    """Input schema for training an XGBoost model."""

    # Model identification
    model_name: str = Field(
        description="Unique name for this model. Used for storage and retrieval. "
        "Example: 'fraud_xgb_v1', 'claim_predictor_xgb'"
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this model does.",
    )

    # Data specification
    data: list[dict] = Field(
        description="Training data as a list of dictionaries, where each dict is a row."
    )
    target_column: str = Field(
        description="Name of the column containing the target variable (y)."
    )
    feature_columns: Optional[list[str]] = Field(
        default=None,
        description="List of column names to use as features. If None, uses all except target.",
    )
    categorical_columns: Optional[list[str]] = Field(
        default=None,
        description="List of categorical column names (will be one-hot encoded).",
    )

    # Task type
    task_type: Literal["classification", "regression"] = Field(
        description="Type of prediction task."
    )

    # Core hyperparameters
    n_estimators: int = Field(
        default=100,
        description="Number of boosting rounds. More = better but slower. Default: 100",
        ge=1,
    )
    max_depth: int = Field(
        default=6,
        description="Maximum tree depth. Higher = more complex. 3-10 typical. Default: 6",
        ge=1,
    )
    learning_rate: float = Field(
        default=0.1,
        description="Step size shrinkage (eta). Lower = more rounds needed but often better. "
        "0.01-0.3 typical. Default: 0.1",
        gt=0,
        le=1,
    )
    
    # Regularization
    min_child_weight: int = Field(
        default=1,
        description="Minimum sum of instance weight in a child. Higher = more conservative. Default: 1",
        ge=0,
    )
    gamma: float = Field(
        default=0,
        description="Minimum loss reduction for split. Higher = more conservative. Default: 0",
        ge=0,
    )
    reg_alpha: float = Field(
        default=0,
        description="L1 regularization (lasso). Higher = sparser model. Default: 0",
        ge=0,
    )
    reg_lambda: float = Field(
        default=1,
        description="L2 regularization (ridge). Higher = simpler model. Default: 1",
        ge=0,
    )

    # Sampling
    subsample: float = Field(
        default=1.0,
        description="Fraction of samples per tree. <1.0 adds randomness. Default: 1.0",
        gt=0,
        le=1,
    )
    colsample_bytree: float = Field(
        default=1.0,
        description="Fraction of features per tree. <1.0 adds randomness. Default: 1.0",
        gt=0,
        le=1,
    )

    # Early stopping
    early_stopping_rounds: Optional[int] = Field(
        default=10,
        description="Stop if no improvement for N rounds. None = no early stopping. Default: 10",
    )

    # Other
    random_state: Optional[int] = Field(
        default=42, description="Random seed for reproducibility. Default: 42"
    )
    n_jobs: int = Field(
        default=-1, description="Parallel threads. -1 = all. Default: -1"
    )

    # Training configuration
    test_size: float = Field(
        default=0.2,
        description="Fraction for test set. Default: 0.2",
        gt=0.0,
        lt=1.0,
    )
    validation_size: float = Field(
        default=0.1,
        description="Fraction for validation (early stopping). Default: 0.1",
        gt=0.0,
        lt=0.5,
    )


class XGBoostTrainingOutput(BaseModel):
    """Output schema for trained XGBoost model."""

    success: bool = Field(description="Whether training completed successfully")
    model_type: str = Field(default="xgboost")
    task_type: Literal["classification", "regression"]

    # Dataset info
    n_samples: int
    n_features: int
    feature_names: list[str]

    # Classification-specific
    n_classes: Optional[int] = None
    class_distribution: Optional[dict[str, int]] = None

    # Metrics
    train_score: float
    val_score: float
    test_score: float
    additional_metrics: dict
    classification_report: Optional[str] = None
    confusion_matrix: Optional[list[list[int]]] = None

    # Training history (loss curve!)
    training_history: dict[str, list[float]] = Field(
        description="Loss values per boosting round for train and validation sets"
    )
    best_iteration: int = Field(description="Best boosting round (if early stopping)")
    actual_n_estimators: int = Field(description="Actual number of trees used")

    # Model info
    feature_importances: dict[str, float]

    # Artifacts
    saved_path: str
    model_name: str
    training_timestamp: str


def train_xgboost(input_data: XGBoostTrainingInput) -> XGBoostTrainingOutput:
    """
    Train an XGBoost model with preprocessing pipeline.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Run: pip install xgboost")

    # Convert input data to DataFrame
    df = pd.DataFrame(input_data.data)

    # Validate target column exists
    if input_data.target_column not in df.columns:
        raise ValueError(
            f"Target column '{input_data.target_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    # Determine feature columns
    if input_data.feature_columns:
        feature_cols = input_data.feature_columns
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
    else:
        feature_cols = [c for c in df.columns if c != input_data.target_column]

    # Separate features and target
    X = df[feature_cols].copy()
    y = df[input_data.target_column].copy()

    # Encode target for classification if needed
    label_encoder = None
    if input_data.task_type == "classification" and y.dtype == object:
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index)

    # Determine categorical vs numeric columns
    categorical_cols = input_data.categorical_columns or []
    categorical_cols = [c for c in categorical_cols if c in feature_cols]

    # Auto-detect categorical columns if not specified
    if not input_data.categorical_columns:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Build preprocessing
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

    # Split data: train / validation / test
    stratify = y if input_data.task_type == "classification" else None
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=input_data.test_size,
        random_state=input_data.random_state,
        stratify=stratify,
    )
    
    # Second split: train vs validation
    stratify_trainval = y_trainval if input_data.task_type == "classification" else None
    val_ratio = input_data.validation_size / (1 - input_data.test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=input_data.random_state,
        stratify=stratify_trainval,
    )

    # Fit preprocessor on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names
    try:
        feature_names_out = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names_out = feature_cols

    # Build XGBoost model
    if input_data.task_type == "classification":
        n_classes = len(np.unique(y))
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        eval_metric = "logloss" if n_classes == 2 else "mlogloss"
        
        model = xgb.XGBClassifier(
            n_estimators=input_data.n_estimators,
            max_depth=input_data.max_depth,
            learning_rate=input_data.learning_rate,
            min_child_weight=input_data.min_child_weight,
            gamma=input_data.gamma,
            reg_alpha=input_data.reg_alpha,
            reg_lambda=input_data.reg_lambda,
            subsample=input_data.subsample,
            colsample_bytree=input_data.colsample_bytree,
            objective=objective,
            eval_metric=eval_metric,
            random_state=input_data.random_state,
            n_jobs=input_data.n_jobs,
            early_stopping_rounds=input_data.early_stopping_rounds,
        )
    else:
        n_classes = None
        model = xgb.XGBRegressor(
            n_estimators=input_data.n_estimators,
            max_depth=input_data.max_depth,
            learning_rate=input_data.learning_rate,
            min_child_weight=input_data.min_child_weight,
            gamma=input_data.gamma,
            reg_alpha=input_data.reg_alpha,
            reg_lambda=input_data.reg_lambda,
            subsample=input_data.subsample,
            colsample_bytree=input_data.colsample_bytree,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=input_data.random_state,
            n_jobs=input_data.n_jobs,
            early_stopping_rounds=input_data.early_stopping_rounds,
        )

    # Train with evaluation set to capture loss history
    eval_set = [(X_train_processed, y_train), (X_val_processed, y_val)]
    
    model.fit(
        X_train_processed, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    # Extract training history
    results = model.evals_result()
    training_history = {}
    
    # Map the eval set names to readable names
    eval_names = list(results.keys())
    metric_name = list(results[eval_names[0]].keys())[0]
    
    training_history["train_loss"] = results[eval_names[0]][metric_name]
    training_history["val_loss"] = results[eval_names[1]][metric_name]

    # Best iteration
    best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else input_data.n_estimators
    actual_n_estimators = best_iteration + 1 if input_data.early_stopping_rounds else input_data.n_estimators

    # Predictions
    y_train_pred = model.predict(X_train_processed)
    y_val_pred = model.predict(X_val_processed)
    y_test_pred = model.predict(X_test_processed)

    # Metrics
    additional_metrics = {}
    classification_report_str = None
    conf_matrix = None
    class_dist = None

    if input_data.task_type == "classification":
        train_score = accuracy_score(y_train, y_train_pred)
        val_score = accuracy_score(y_val, y_val_pred)
        test_score = accuracy_score(y_test, y_test_pred)

        # ROC-AUC
        try:
            y_test_proba = model.predict_proba(X_test_processed)
            if n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_test_proba, multi_class="ovr", average="weighted")
            additional_metrics["roc_auc"] = roc_auc
        except ValueError:
            additional_metrics["roc_auc"] = None

        classification_report_str = classification_report(y_test, y_test_pred)
        conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
        
        class_dist = pd.Series(y).value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}

    else:
        train_score = r2_score(y_train, y_train_pred)
        val_score = r2_score(y_val, y_val_pred)
        test_score = r2_score(y_test, y_test_pred)

        additional_metrics["mae"] = mean_absolute_error(y_test, y_test_pred)
        additional_metrics["rmse"] = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Feature importances
    importances = model.feature_importances_
    feature_importances = {
        name: float(imp) for name, imp in zip(feature_names_out, importances)
    }

    # Create a wrapper that includes preprocessing
    pipeline = XGBoostPipelineWrapper(preprocessor, model, label_encoder)

    # Save model
    save_path = generate_model_path(input_data.model_name)
    joblib.dump(pipeline, save_path)

    # Register in registry
    metrics = {
        "train_score": train_score,
        "val_score": val_score,
        "test_score": test_score,
        **additional_metrics,
    }

    hyperparameters = {
        "task_type": input_data.task_type,
        "n_estimators": actual_n_estimators,
        "max_depth": input_data.max_depth,
        "learning_rate": input_data.learning_rate,
        "min_child_weight": input_data.min_child_weight,
        "gamma": input_data.gamma,
        "reg_alpha": input_data.reg_alpha,
        "reg_lambda": input_data.reg_lambda,
        "subsample": input_data.subsample,
        "colsample_bytree": input_data.colsample_bytree,
        "early_stopping_rounds": input_data.early_stopping_rounds,
        "random_state": input_data.random_state,
    }

    classes_list = (
        [str(c) for c in model.classes_]
        if input_data.task_type == "classification"
        else []
    )

    register_model(
        model_name=input_data.model_name,
        model_path=save_path,
        model_type="xgboost",
        description=input_data.description,
        metrics=metrics,
        feature_names=feature_names_out,
        target_column=input_data.target_column,
        hyperparameters=hyperparameters,
        training_samples=len(df),
        classes=classes_list,
    )

    return XGBoostTrainingOutput(
        success=True,
        task_type=input_data.task_type,
        n_samples=len(df),
        n_features=len(feature_names_out),
        feature_names=feature_names_out,
        n_classes=n_classes,
        class_distribution=class_dist,
        train_score=train_score,
        val_score=val_score,
        test_score=test_score,
        additional_metrics=additional_metrics,
        classification_report=classification_report_str,
        confusion_matrix=conf_matrix,
        training_history=training_history,
        best_iteration=best_iteration,
        actual_n_estimators=actual_n_estimators,
        feature_importances=feature_importances,
        saved_path=save_path,
        model_name=input_data.model_name,
        training_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class SklearnXGBoostToolInput(BaseModel):
    """Input for training an XGBoost model."""

    model_name: str = Field(description="Unique model name for storage")
    description: str = Field(default="", description="Model description")
    data: list[dict] = Field(description="Training data as list of row dicts")
    target_column: str = Field(description="Target column name")
    task_type: Literal["classification", "regression"] = Field(
        description="'classification' or 'regression'"
    )
    feature_columns: Optional[list[str]] = Field(default=None)
    categorical_columns: Optional[list[str]] = Field(default=None)
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    min_child_weight: int = Field(default=1, ge=0)
    gamma: float = Field(default=0, ge=0)
    reg_alpha: float = Field(default=0, ge=0)
    reg_lambda: float = Field(default=1, ge=0)
    subsample: float = Field(default=1.0, gt=0, le=1)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1)
    early_stopping_rounds: Optional[int] = Field(default=10)
    random_state: Optional[int] = Field(default=42)
    n_jobs: int = Field(default=-1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    validation_size: float = Field(default=0.1, gt=0, lt=0.5)


@tool("xgboost_train", args_schema=SklearnXGBoostToolInput)
def xgboost_train_tool(
    model_name: str,
    data: list[dict],
    target_column: str,
    task_type: Literal["classification", "regression"],
    description: str = "",
    feature_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_weight: int = 1,
    gamma: float = 0,
    reg_alpha: float = 0,
    reg_lambda: float = 1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    early_stopping_rounds: Optional[int] = 10,
    random_state: Optional[int] = 42,
    n_jobs: int = -1,
    test_size: float = 0.2,
    validation_size: float = 0.1,
) -> str:
    """Train an XGBoost model with loss curve tracking and early stopping.

    WHAT IT DOES:
    XGBoost is a gradient boosting algorithm that builds trees sequentially,
    with each tree correcting errors from previous trees. It often achieves
    state-of-the-art results on tabular data.

    Key advantages:
    - LOSS CURVE: Track training/validation loss per boosting round
    - EARLY STOPPING: Automatically stop when validation loss stops improving
    - REGULARIZATION: Built-in L1/L2 regularization to prevent overfitting
    - FEATURE IMPORTANCE: See which features matter most

    WHEN TO USE:
    - Tabular data where accuracy is critical
    - When you need the best possible performance
    - Complex non-linear relationships
    - After trying simpler baselines (logistic regression, random forest)

    WHEN NOT TO USE:
    - When interpretability is critical (use logistic regression)
    - Very small datasets (<100 samples)
    - When training speed is more important than accuracy

    HYPERPARAMETER GUIDANCE:
    - learning_rate: Start with 0.1. Lower (0.01-0.05) often better with more trees.
    - max_depth: 3-6 for most problems. Higher = more complex but risk overfitting.
    - n_estimators: 100-1000. Use early_stopping to find optimal.
    - subsample/colsample_bytree: 0.7-0.9 adds regularization.
    - min_child_weight: Increase (5-10) if overfitting.

    LOSS CURVE:
    The output shows loss at each boosting round:
    ```
    Round  1: Train=0.693, Val=0.693
    Round 10: Train=0.312, Val=0.345
    Round 20: Train=0.198, Val=0.287  ← validation starts diverging
    Round 25: Train=0.156, Val=0.291  ← early stopping triggered
    Best iteration: 20
    ```

    STORAGE:
    Models are saved to trained_models/ and registered for later use.

    Args:
        model_name: Unique name for this model
        data: Training data as list of dictionaries
        target_column: Target variable column
        task_type: 'classification' or 'regression'
        n_estimators: Maximum boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        early_stopping_rounds: Stop if no improvement for N rounds
        ... (other regularization and sampling parameters)

    Returns:
        Training results with loss curve, metrics, and feature importances.
    """
    if not XGBOOST_AVAILABLE:
        return "❌ XGBoost is not installed. Run: pip install xgboost"

    try:
        result = train_xgboost(
            XGBoostTrainingInput(
                model_name=model_name,
                description=description,
                data=data,
                target_column=target_column,
                task_type=task_type,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                early_stopping_rounds=early_stopping_rounds,
                random_state=random_state,
                n_jobs=n_jobs,
                test_size=test_size,
                validation_size=validation_size,
            )
        )

        # Format output
        task_label = "CLASSIFICATION" if result.task_type == "classification" else "REGRESSION"
        score_label = "Accuracy" if result.task_type == "classification" else "R²"

        output_lines = [
            "=" * 60,
            f"XGBOOST {task_label} TRAINING COMPLETE",
            "=" * 60,
            "",
            "📊 DATASET SUMMARY",
            f"  Samples: {result.n_samples}",
            f"  Features: {result.n_features}",
        ]

        if result.task_type == "classification":
            output_lines.extend([
                f"  Classes: {result.n_classes}",
                f"  Distribution: {result.class_distribution}",
            ])

        output_lines.extend([
            "",
            "📈 PERFORMANCE METRICS",
            f"  Train {score_label}: {result.train_score:.4f}",
            f"  Val {score_label}:   {result.val_score:.4f}",
            f"  Test {score_label}:  {result.test_score:.4f}",
        ])

        if result.task_type == "classification" and result.additional_metrics.get("roc_auc"):
            output_lines.append(f"  Test ROC-AUC:  {result.additional_metrics['roc_auc']:.4f}")
        elif result.task_type == "regression":
            output_lines.extend([
                f"  Test MAE:      {result.additional_metrics['mae']:.4f}",
                f"  Test RMSE:     {result.additional_metrics['rmse']:.4f}",
            ])

        # LOSS CURVE - The key feature!
        output_lines.extend([
            "",
            "📉 TRAINING LOSS CURVE (per boosting round):",
            "-" * 50,
        ])
        
        train_loss = result.training_history["train_loss"]
        val_loss = result.training_history["val_loss"]
        
        # Show key points in the loss curve
        n_rounds = len(train_loss)
        show_rounds = [0, n_rounds//4, n_rounds//2, 3*n_rounds//4, n_rounds-1]
        show_rounds = sorted(set(min(r, n_rounds-1) for r in show_rounds))
        
        for r in show_rounds:
            marker = " ← best" if r == result.best_iteration else ""
            train_bar = "█" * int((1 - min(train_loss[r], 1)) * 20)
            val_bar = "▓" * int((1 - min(val_loss[r], 1)) * 20)
            output_lines.append(
                f"  Round {r+1:3d}: Train={train_loss[r]:.4f} {train_bar}"
            )
            output_lines.append(
                f"            Val={val_loss[r]:.4f}   {val_bar}{marker}"
            )
        
        output_lines.extend([
            "",
            f"  Best iteration: {result.best_iteration + 1} / {n_rounds}",
            f"  Trees used: {result.actual_n_estimators}",
        ])

        if result.classification_report:
            output_lines.extend([
                "",
                "📋 CLASSIFICATION REPORT",
                result.classification_report,
            ])

        if result.confusion_matrix:
            output_lines.extend([
                "",
                "🔢 CONFUSION MATRIX",
                str(np.array(result.confusion_matrix)),
            ])

        # Feature importances
        output_lines.extend(["", "🎯 FEATURE IMPORTANCES (top 10):"])
        sorted_importances = sorted(
            result.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        for feat, imp in sorted_importances[:10]:
            bar = "█" * int(imp * 50)
            output_lines.append(f"  {feat}: {imp:.4f} {bar}")

        output_lines.extend([
            "",
            f"💾 MODEL REGISTERED: {result.model_name}",
            f"   Path: {result.saved_path}",
            "",
            f"⏱️ Timestamp: {result.training_timestamp}",
            "=" * 60,
        ])

        return "\n".join(output_lines)

    except Exception as e:
        return (
            f"❌ TRAINING FAILED\n"
            f"Error: {str(e)}\n\n"
            f"Please check:\n"
            f"- Data format (list of dicts)\n"
            f"- Target column exists\n"
            f"- XGBoost is installed (pip install xgboost)"
        )

