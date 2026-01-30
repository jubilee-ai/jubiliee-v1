"""
Random Forest Training Tool for AI Data Scientist Agent

A LangChain tool that dynamically trains sklearn RandomForest models
for classification or regression tasks. Designed for underwriting,
fraud detection, risk scoring, and other tabular ML use cases.
"""

import os
from datetime import datetime, timezone
from typing import Literal, Optional, Union

import joblib
import numpy as np
import pandas as pd
from langchain.tools import tool
from model_storage import generate_model_path, register_model
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class RandomForestTrainingInput(BaseModel):
    """Input schema for training a Random Forest model."""
    model_config = ConfigDict(extra="forbid")

    # Model identification
    model_name: str = Field(
        description="Unique name for this model. Used for storage and retrieval. "
        "Example: 'fraud_detector_v1', 'claim_amount_predictor'"
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this model does. "
        "Example: 'Predicts fraudulent claims based on claim attributes'",
    )

    # Data specification
    data: list[dict] = Field(
        description="Training data as a list of dictionaries, where each dict is a row. "
        "Example: [{'age': 25, 'income': 50000, 'target': 1}, ...]"
    )
    target_column: str = Field(
        description="Name of the column containing the target variable (y). "
        "Must be present in each data row."
    )
    feature_columns: Optional[list[str]] = Field(
        default=None,
        description="List of column names to use as features. If None, uses all columns except target_column.",
    )
    categorical_columns: Optional[list[str]] = Field(
        default=None,
        description="List of column names that are categorical (will be one-hot encoded). "
        "Numeric columns will be standardized. If None, auto-detects from dtypes.",
    )
    sample_weight_column: Optional[str] = Field(
        default=None,
        description="Column containing per-row sample weights. Excluded from features.",
    )

    # Task type
    task_type: Literal["classification", "regression"] = Field(
        description="Type of prediction task. 'classification' for discrete classes, "
        "'regression' for continuous values."
    )

    # Model hyperparameters
    n_estimators: int = Field(
        default=100,
        description="Number of trees in the forest. More trees = better performance but slower. "
        "Typical range: 50-500. Default: 100",
        ge=1,
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Maximum depth of each tree. None = nodes expand until all leaves are pure. "
        "Set to 5-20 to prevent overfitting. Default: None",
    )
    min_samples_split: int = Field(
        default=2,
        description="Minimum samples required to split an internal node. "
        "Higher values prevent overfitting. Default: 2",
        ge=2,
    )
    min_samples_leaf: int = Field(
        default=1,
        description="Minimum samples required at a leaf node. "
        "Higher values create smoother predictions. Default: 1",
        ge=1,
    )
    max_features: Union[Literal["sqrt", "log2"], float, None] = Field(
        default="sqrt",
        description="Number of features to consider for best split. "
        "'sqrt' = sqrt(n_features), 'log2' = log2(n_features), float = fraction. Default: 'sqrt'",
    )
    bootstrap: bool = Field(
        default=True,
        description="Whether to use bootstrap samples when building trees. "
        "True enables out-of-bag error estimation. Default: True",
    )
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = Field(
        default=None,
        description="Weight classes for imbalanced data. 'balanced' adjusts weights inversely "
        "proportional to class frequencies. Only used for classification. Default: None",
    )
    random_state: Optional[int] = Field(
        default=42, description="Random seed for reproducibility. Default: 42"
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs. -1 = use all processors. Default: -1",
    )

    # Training configuration
    test_size: float = Field(
        default=0.2,
        description="Fraction of data to hold out for evaluation (0.0 to 1.0). Default: 0.2",
        gt=0.0,
        lt=1.0,
    )


class RandomForestTrainingOutput(BaseModel):
    """Output schema for trained Random Forest model."""

    success: bool = Field(description="Whether training completed successfully")
    model_type: str = Field(default="sklearn_random_forest")
    task_type: Literal["classification", "regression"] = Field(
        description="Whether this is a classification or regression model"
    )

    # Dataset info
    n_samples: int = Field(description="Total number of samples in dataset")
    n_features: int = Field(description="Number of features used")
    feature_names: list[str] = Field(description="Names of features used")

    # Classification-specific
    n_classes: Optional[int] = Field(
        default=None, description="Number of unique classes (classification only)"
    )
    class_distribution: Optional[dict[str, int]] = Field(
        default=None, description="Count of samples per class (classification only)"
    )

    # Metrics
    train_score: float = Field(
        description="Training score (accuracy for classification, R² for regression)"
    )
    test_score: float = Field(
        description="Test score (accuracy for classification, R² for regression)"
    )
    additional_metrics: dict = Field(
        description="Additional metrics (ROC-AUC, MAE, RMSE, etc.)"
    )
    classification_report: Optional[str] = Field(
        default=None, description="Full sklearn classification report (classification only)"
    )
    confusion_matrix: Optional[list[list[int]]] = Field(
        default=None, description="Confusion matrix (classification only)"
    )

    # Model info
    feature_importances: dict[str, float] = Field(
        description="Feature importance scores (mean decrease in impurity)"
    )
    n_estimators: int = Field(description="Number of trees in the forest")
    oob_score: Optional[float] = Field(
        default=None, description="Out-of-bag score if bootstrap=True"
    )

    # Artifacts
    saved_path: str = Field(description="Path where model was saved")
    model_name: str = Field(description="Name of the model in the registry")
    training_timestamp: str = Field(description="ISO timestamp of training completion")


def train_random_forest(
    input_data: RandomForestTrainingInput,
) -> RandomForestTrainingOutput:
    """
    Train a Random Forest model with preprocessing pipeline.

    Creates a full sklearn Pipeline with:
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    - RandomForestClassifier or RandomForestRegressor

    This ensures inference uses identical transformations as training.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data.data)

    # Validate target column exists
    if input_data.target_column not in df.columns:
        raise ValueError(
            f"Target column '{input_data.target_column}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )

    # Extract sample weights if provided
    sample_weights = None
    if input_data.sample_weight_column:
        if input_data.sample_weight_column not in df.columns:
            raise ValueError(f"Sample weight column '{input_data.sample_weight_column}' not found.")
        sample_weights = df[input_data.sample_weight_column].values
    
    # Determine feature columns (exclude target and sample_weight_column)
    exclude_cols = [input_data.target_column]
    if input_data.sample_weight_column:
        exclude_cols.append(input_data.sample_weight_column)
    
    if input_data.feature_columns:
        feature_cols = [c for c in input_data.feature_columns if c not in exclude_cols]
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not found in data: {missing}")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Separate features and target
    X = df[feature_cols]
    y = df[input_data.target_column]

    # Determine categorical vs numeric columns
    categorical_cols = input_data.categorical_columns or []
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Auto-detect categorical columns if not specified (object/string dtype)
    if not input_data.categorical_columns:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Build preprocessing pipeline
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

    # Build the model based on task type
    if input_data.task_type == "classification":
        model = RandomForestClassifier(
            n_estimators=input_data.n_estimators,
            max_depth=input_data.max_depth,
            min_samples_split=input_data.min_samples_split,
            min_samples_leaf=input_data.min_samples_leaf,
            max_features=input_data.max_features,
            bootstrap=input_data.bootstrap,
            class_weight=input_data.class_weight,
            random_state=input_data.random_state,
            n_jobs=input_data.n_jobs,
            oob_score=input_data.bootstrap,  # Enable OOB if bootstrap is True
        )
    else:
        model = RandomForestRegressor(
            n_estimators=input_data.n_estimators,
            max_depth=input_data.max_depth,
            min_samples_split=input_data.min_samples_split,
            min_samples_leaf=input_data.min_samples_leaf,
            max_features=input_data.max_features,
            bootstrap=input_data.bootstrap,
            random_state=input_data.random_state,
            n_jobs=input_data.n_jobs,
            oob_score=input_data.bootstrap,
        )

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # NOTE: No internal train_test_split - data is already split by the pipeline
    # The training agent passes pre-split training data here
    # Validation/test evaluation happens via evaluate_model tool

    # Fit pipeline on ALL provided data (it's already the training set)
    if sample_weights is not None:
        pipeline.fit(X, y, model__sample_weight=sample_weights)
    else:
        pipeline.fit(X, y)

    # Get transformed feature names
    try:
        feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    except AttributeError:
        feature_names_out = feature_cols

    # Predictions on training data (for sanity check metrics)
    y_pred = pipeline.predict(X)

    # Extract the fitted model for feature importances and OOB score
    fitted_model = pipeline.named_steps["model"]

    # Calculate metrics based on task type
    # NOTE: These are TRAINING metrics only - use evaluate_model for val/test metrics
    additional_metrics = {}
    classification_report_str = None
    conf_matrix = None
    n_classes = None
    class_dist = None

    if input_data.task_type == "classification":
        train_score = accuracy_score(y, y_pred)

        # Additional classification metrics
        classes = fitted_model.classes_
        n_classes = len(classes)

        try:
            y_proba = pipeline.predict_proba(X)
            if n_classes == 2:
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(
                    y, y_proba, multi_class="ovr", average="weighted"
                )
            additional_metrics["roc_auc"] = roc_auc
        except ValueError:
            additional_metrics["roc_auc"] = None

        classification_report_str = classification_report(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred).tolist()

        # Class distribution
        class_dist = y.value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}

    else:  # regression
        train_score = r2_score(y, y_pred)

        # Additional regression metrics
        additional_metrics["mae"] = mean_absolute_error(y, y_pred)
        additional_metrics["rmse"] = np.sqrt(mean_squared_error(y, y_pred))
        additional_metrics["mse"] = mean_squared_error(y, y_pred)

    # Feature importances
    importances = fitted_model.feature_importances_
    feature_importances = {
        name: float(imp) for name, imp in zip(feature_names_out, importances)
    }

    # OOB score if available
    oob_score = None
    if input_data.bootstrap and hasattr(fitted_model, "oob_score_"):
        oob_score = fitted_model.oob_score_

    # Save model to trained_models directory
    save_path = generate_model_path(input_data.model_name)
    joblib.dump(pipeline, save_path)

    # Prepare metrics and hyperparameters for registry
    # NOTE: These are TRAINING metrics only - use evaluate_model for val/test metrics
    metrics = {
        "train_score": train_score,
        **additional_metrics,
    }

    hyperparameters = {
        "task_type": input_data.task_type,
        "n_estimators": input_data.n_estimators,
        "max_depth": input_data.max_depth,
        "min_samples_split": input_data.min_samples_split,
        "min_samples_leaf": input_data.min_samples_leaf,
        "max_features": input_data.max_features,
        "bootstrap": input_data.bootstrap,
        "class_weight": input_data.class_weight,
        "random_state": input_data.random_state,
        "n_jobs": input_data.n_jobs,
    }

    # Register model in the registry
    classes_list = (
        [str(c) for c in fitted_model.classes_]
        if input_data.task_type == "classification"
        else []
    )

    register_model(
        model_name=input_data.model_name,
        model_path=save_path,
        model_type="sklearn_random_forest",
        description=input_data.description,
        metrics=metrics,
        feature_names=feature_names_out,
        target_column=input_data.target_column,
        hyperparameters=hyperparameters,
        training_samples=len(df),
        classes=classes_list,
    )

    return RandomForestTrainingOutput(
        success=True,
        task_type=input_data.task_type,
        n_samples=len(df),
        n_features=len(feature_names_out),
        feature_names=feature_names_out,
        n_classes=n_classes,
        class_distribution=class_dist,
        train_score=train_score,
        test_score=train_score,  # Same as train (no internal split anymore)
        additional_metrics=additional_metrics,
        classification_report=classification_report_str,
        confusion_matrix=conf_matrix,
        feature_importances=feature_importances,
        n_estimators=input_data.n_estimators,
        oob_score=oob_score,
        saved_path=save_path,
        model_name=input_data.model_name,
        training_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class SklearnRandomForestToolInput(BaseModel):
    """Input for training a sklearn Random Forest model."""

    model_name: str = Field(
        description="Unique name for this model. Used to save, load, and reference the model. "
        "Example: 'fraud_detector_v1', 'claim_predictor'"
    )
    description: str = Field(
        default="", description="Human-readable description of the model's purpose."
    )
    train_dataset_ref: str = Field(
        description="Reference name of the registered training dataset. "
        "The dataset must be registered via register_dataset(). "
        "Example: 'basic_train', 'pipeline_train_features'"
    )
    target_column: str = Field(
        description="Name of target column to predict. Must exist in the dataset."
    )
    task_type: Literal["classification", "regression"] = Field(
        description="'classification' for discrete classes (fraud/not fraud), "
        "'regression' for continuous values (claim amount)"
    )
    feature_columns: Optional[list[str]] = Field(
        default=None,
        description="Feature column names to use. If None, uses all columns except target.",
    )
    categorical_columns: Optional[list[str]] = Field(
        default=None,
        description="Columns to one-hot encode. If None, auto-detects from dtypes.",
    )
    sample_weight_column: Optional[str] = Field(
        default=None,
        description="Column with per-row weights. Use for recency, policy size, or label confidence."
    )
    n_estimators: int = Field(
        default=100, description="Number of trees in forest. Default: 100", ge=1
    )
    max_depth: Optional[int] = Field(
        default=None,
        description="Max tree depth. None=unlimited. Set 5-20 to prevent overfitting.",
    )
    min_samples_split: int = Field(
        default=2, description="Min samples to split a node. Default: 2", ge=2
    )
    min_samples_leaf: int = Field(
        default=1, description="Min samples at leaf node. Default: 1", ge=1
    )
    max_features: Union[Literal["sqrt", "log2"], float, None] = Field(
        default="sqrt",
        description="Features per split: 'sqrt', 'log2', or float fraction. Default: 'sqrt'",
    )
    bootstrap: bool = Field(
        default=True, description="Use bootstrap samples. Enables OOB score. Default: True"
    )
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = Field(
        default=None,
        description="For imbalanced classification: 'balanced' or 'balanced_subsample'",
    )
    random_state: Optional[int] = Field(
        default=42, description="Random seed for reproducibility. Default: 42"
    )
    n_jobs: int = Field(
        default=-1, description="Parallel jobs. -1=all processors. Default: -1"
    )
    test_size: float = Field(
        default=0.2,
        description="Fraction held out for evaluation (0-1). Default: 0.2",
        gt=0.0,
        lt=1.0,
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


@tool("sklearn_random_forest", args_schema=SklearnRandomForestToolInput)
def sklearn_random_forest_tool(
    model_name: str,
    train_dataset_ref: str,
    target_column: str,
    task_type: Literal["classification", "regression"],
    description: str = "",
    feature_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    sample_weight_column: Optional[str] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Union[Literal["sqrt", "log2"], float, None] = "sqrt",
    bootstrap: bool = True,
    class_weight: Optional[Literal["balanced", "balanced_subsample"]] = None,
    random_state: Optional[int] = 42,
    n_jobs: int = -1,
    test_size: float = 0.2,
) -> str:
    """Train a Random Forest model (sklearn) on tabular data with automatic preprocessing.

    WHAT IT DOES:
    Random Forest is an ensemble of decision trees that captures non-linearities and
    feature interactions with minimal feature engineering, making it a strong baseline
    for tabular underwriting, fraud detection, and risk scoring tasks. It is robust to
    mixed feature types and noisy signals, but can be less interpretable and may require
    probability calibration for threshold-based decisions.

    The tool trains a RandomForest model using scikit-learn and saves the full
    preprocessing + model pipeline to avoid train/serve skew.

    Pipeline includes:
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features
    - RandomForestClassifier or RandomForestRegressor

    WHEN TO USE:
    - Tabular classification: fraud detection, risk scoring, churn prediction
    - Tabular regression: claim amount prediction, pricing, LTV estimation
    - When you need to capture non-linear relationships and feature interactions
    - When you have mixed feature types (numeric + categorical)
    - As a strong baseline before trying boosting methods (XGBoost, LightGBM)

    WHEN NOT TO USE:
    - Very high-dimensional sparse data (text, images) - use specialized models
    - When strict interpretability is required - use logistic regression
    - When you need calibrated probabilities - consider calibration wrapper
    - Real-time inference with tight latency constraints - trees can be slow

    HYPERPARAMETER GUIDANCE:
    - n_estimators: Start with 100. More trees = better but diminishing returns after 300-500.
    - max_depth: None for full trees. Set 5-15 to prevent overfitting on small datasets.
    - min_samples_split/leaf: Increase to 5-10 for noisy data or small datasets.
    - max_features: 'sqrt' is good for classification, try 0.3-0.5 for regression.
    - class_weight: Use 'balanced' for imbalanced classification (rare fraud, defaults).
    
    FOR IMBALANCED DATA (positive rate < 20%):
    - ALWAYS use class_weight='balanced'
    - When switching FROM logistic regression that used class_weight='balanced', 
      you MUST also use class_weight='balanced' here
    
    FOR OVERFITTING (train_score >> val_score by >0.1):
    - Reduce max_depth (try 4-6)
    - Increase min_samples_leaf (try 5-10)
    - Increase min_samples_split (try 5-10)

    OUTPUT INTERPRETATION:
    - test_score: Accuracy (classification) or R² (regression). Compare to baseline.
    - roc_auc: For classification. >0.7 acceptable, >0.8 good, >0.9 excellent.
    - feature_importances: Higher = more predictive. Use for feature selection.
    - oob_score: Out-of-bag estimate of generalization error (if bootstrap=True).

    STORAGE:
    Models are automatically saved to the trained_models/ directory and registered
    in the model registry. Use list_trained_models to see all models, and
    predict_with_model to make predictions with a saved model.

    EXAMPLE (Classification):
    ```
    result = sklearn_random_forest(
        model_name="fraud_detector_v1",
        description="Detects fraudulent insurance claims",
        data=[
            {"claim_amount": 5000, "age": 35, "prior_claims": 0, "fraud": 0},
            {"claim_amount": 50000, "age": 25, "prior_claims": 3, "fraud": 1},
            ...
        ],
        target_column="fraud",
        task_type="classification",
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
    )
    ```

    EXAMPLE (Regression):
    ```
    result = sklearn_random_forest(
        model_name="claim_amount_predictor",
        description="Predicts expected claim amount",
        data=[
            {"age": 35, "policy_type": "comprehensive", "claim_amount": 5000},
            {"age": 55, "policy_type": "basic", "claim_amount": 2000},
            ...
        ],
        target_column="claim_amount",
        task_type="regression",
        n_estimators=100,
        max_depth=15,
    )
    ```

    Args:
        model_name: Unique name for this model (used to save and load)
        description: Human-readable description of the model
        data: Training data as list of row dictionaries
        target_column: Name of the target column to predict
        task_type: 'classification' or 'regression'
        feature_columns: Which columns to use as features (None = all except target)
        categorical_columns: Which columns are categorical (None = auto-detect)
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples at a leaf node
        max_features: Features to consider per split ('sqrt', 'log2', or float)
        bootstrap: Use bootstrap samples (enables OOB score)
        class_weight: Class weighting for imbalanced data (classification only)
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = all processors)
        test_size: Fraction of data for test set evaluation

    Returns:
        Formatted string with training results including scores, feature importances,
        and model registry info.
    """
    try:
        # Load dataset from registry
        df = _load_dataset_from_ref(train_dataset_ref)
        data = df.to_dict(orient="records")
        
        result = train_random_forest(
            RandomForestTrainingInput(
                model_name=model_name,
                description=description,
                data=data,
                target_column=target_column,
                task_type=task_type,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                sample_weight_column=sample_weight_column,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs,
                test_size=test_size,
            )
        )

        # Format output for agent consumption
        task_label = "CLASSIFICATION" if result.task_type == "classification" else "REGRESSION"
        score_label = "Accuracy" if result.task_type == "classification" else "R²"

        output_lines = [
            "=" * 60,
            f"RANDOM FOREST {task_label} TRAINING COMPLETE",
            "=" * 60,
            "",
            "📊 DATASET SUMMARY",
            f"  Samples: {result.n_samples}",
            f"  Features: {result.n_features}",
        ]

        if result.task_type == "classification":
            output_lines.extend(
                [
                    f"  Classes: {result.n_classes}",
                    f"  Class Distribution: {result.class_distribution}",
                ]
            )

        output_lines.extend(
            [
                "",
                "📈 PERFORMANCE METRICS",
                f"  Train {score_label}: {result.train_score:.4f}",
                f"  Test {score_label}:  {result.test_score:.4f}",
            ]
        )

        # Additional metrics
        if result.task_type == "classification":
            if result.additional_metrics.get("roc_auc") is not None:
                output_lines.append(
                    f"  Test ROC-AUC:   {result.additional_metrics['roc_auc']:.4f}"
                )
        else:
            output_lines.extend(
                [
                    f"  Test MAE:       {result.additional_metrics['mae']:.4f}",
                    f"  Test RMSE:      {result.additional_metrics['rmse']:.4f}",
                ]
            )

        if result.oob_score is not None:
            output_lines.append(f"  OOB Score:      {result.oob_score:.4f}")

        # Classification report
        if result.classification_report:
            output_lines.extend(
                ["", "📋 CLASSIFICATION REPORT", result.classification_report]
            )

        # Confusion matrix
        if result.confusion_matrix:
            output_lines.extend(
                ["", "🔢 CONFUSION MATRIX", str(np.array(result.confusion_matrix))]
            )

        # Feature importances (top 10)
        output_lines.extend(["", "🌲 FEATURE IMPORTANCES (top 10):"])
        sorted_importances = sorted(
            result.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        for feat, imp in sorted_importances[:10]:
            bar = "█" * int(imp * 50)
            output_lines.append(f"  {feat}: {imp:.4f} {bar}")

        output_lines.extend(
            [
                "",
                f"🌳 Model: {result.n_estimators} trees",
                "",
                f"💾 MODEL REGISTERED: {result.model_name}",
                f"   Path: {result.saved_path}",
                "   Use predict_with_model or list_trained_models to access",
                "",
                f"⏱️ Timestamp: {result.training_timestamp}",
                "=" * 60,
            ]
        )

        return "\n".join(output_lines)

    except Exception as e:
        return (
            f"❌ TRAINING FAILED\n"
            f"Error: {str(e)}\n\n"
            f"Please check:\n"
            f"- Data format (list of dicts with consistent keys)\n"
            f"- Target column exists in data\n"
            f"- Task type matches target (classification for discrete, regression for continuous)\n"
            f"- Feature columns exist if specified"
        )

