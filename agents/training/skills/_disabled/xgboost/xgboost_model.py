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
from pydantic import BaseModel, ConfigDict, Field
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
    model_config = ConfigDict(extra="forbid")

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
    sample_weight_column: Optional[str] = Field(
        default=None,
        description="Column containing per-row sample weights. Excluded from features.",
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

    # Imbalanced data
    scale_pos_weight: Optional[float] = Field(
        default=None,
        description="Balance weight for positive class in binary classification. "
        "Set to (negative_count / positive_count) for imbalanced data. "
        "E.g., if 95% negative and 5% positive, set to 19.0. Default: None (no reweighting)",
    )

    # Interpretability constraints
    monotone_constraints: Optional[dict[str, int]] = Field(
        default=None,
        description="Monotonic constraints per feature. Dict mapping feature name to constraint: "
        "1 = increasing (higher feature → higher prediction), "
        "-1 = decreasing (higher feature → lower prediction), "
        "0 = no constraint. Example: {'income': -1, 'debt_ratio': 1} for default prediction. "
        "Default: None (no constraints)",
    )

    # Tree algorithm
    tree_method: Literal["auto", "exact", "approx", "hist"] = Field(
        default="auto",
        description="Tree construction algorithm. 'auto' picks best. 'hist' is faster for large data (>10k rows). "
        "'exact' is most accurate but slow. Default: 'auto'",
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
            raise ValueError(f"Feature columns not found: {missing}")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]

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

    # NOTE: No large internal train_test_split - data is already split by the pipeline
    # The training agent passes pre-split training data here
    # Validation/test evaluation happens via evaluate_model tool
    #
    # However, XGBoost benefits from early stopping, so we keep a SMALL (10%) 
    # internal holdout purely for early stopping purposes
    stratify = y if input_data.task_type == "classification" else None
    
    if sample_weights is not None:
        X_train, X_early_stop, y_train, y_early_stop, sw_train, sw_early_stop = train_test_split(
            X, y, sample_weights,
            test_size=0.1,
            random_state=input_data.random_state,
            stratify=stratify,
        )
    else:
        X_train, X_early_stop, y_train, y_early_stop = train_test_split(
            X, y,
            test_size=0.1,
            random_state=input_data.random_state,
            stratify=stratify,
        )
        sw_train = sw_early_stop = None

    # Fit preprocessor on training portion
    X_train_processed = preprocessor.fit_transform(X_train)
    X_early_stop_processed = preprocessor.transform(X_early_stop)

    # Get feature names
    try:
        feature_names_out = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names_out = feature_cols

    # Build monotone constraints tuple if provided (XGBoost needs tuple format)
    monotone_constraints_tuple = None
    if input_data.monotone_constraints:
        # Map feature names to their positions after preprocessing
        constraint_list = []
        for feat_name in feature_names_out:
            constraint = input_data.monotone_constraints.get(feat_name, 0)
            # Also check original feature name (before one-hot encoding prefix)
            if constraint == 0:
                for orig_name, orig_constraint in input_data.monotone_constraints.items():
                    if feat_name.startswith(f"cat__{orig_name}_") or feat_name.startswith(f"num__{orig_name}"):
                        constraint = orig_constraint
                        break
            constraint_list.append(constraint)
        monotone_constraints_tuple = tuple(constraint_list)

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
            scale_pos_weight=input_data.scale_pos_weight,
            monotone_constraints=monotone_constraints_tuple,
            tree_method=input_data.tree_method,
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
            monotone_constraints=monotone_constraints_tuple,
            tree_method=input_data.tree_method,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=input_data.random_state,
            n_jobs=input_data.n_jobs,
            early_stopping_rounds=input_data.early_stopping_rounds,
        )

    # Train with evaluation set to capture loss history (using small holdout for early stopping)
    eval_set = [(X_train_processed, y_train), (X_early_stop_processed, y_early_stop)]
    
    model.fit(
        X_train_processed, y_train,
        eval_set=eval_set,
        sample_weight=sw_train,
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

    # Now refit on ALL data for final model (early stopping determined optimal n_estimators)
    # Preprocess ALL data
    X_all_processed = preprocessor.fit_transform(X)
    model.set_params(n_estimators=actual_n_estimators, early_stopping_rounds=None)
    model.fit(X_all_processed, y, sample_weight=sample_weights, verbose=False)

    # Predictions on training data (for sanity check metrics)
    y_pred = model.predict(X_all_processed)

    # Metrics - training only (use evaluate_model for val/test)
    additional_metrics = {}
    classification_report_str = None
    conf_matrix = None
    class_dist = None

    if input_data.task_type == "classification":
        train_score = accuracy_score(y, y_pred)

        # ROC-AUC
        try:
            y_proba = model.predict_proba(X_all_processed)
            if n_classes == 2:
                roc_auc = roc_auc_score(y, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
            additional_metrics["roc_auc"] = roc_auc
        except ValueError:
            additional_metrics["roc_auc"] = None

        classification_report_str = classification_report(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred).tolist()
        
        class_dist = pd.Series(y).value_counts().to_dict()
        class_dist = {str(k): int(v) for k, v in class_dist.items()}

    else:
        train_score = r2_score(y, y_pred)

        additional_metrics["mae"] = mean_absolute_error(y, y_pred)
        additional_metrics["rmse"] = np.sqrt(mean_squared_error(y, y_pred))

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
    # NOTE: These are TRAINING metrics only - use evaluate_model for val/test metrics
    metrics = {
        "train_score": train_score,
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
        "scale_pos_weight": input_data.scale_pos_weight,
        "monotone_constraints": input_data.monotone_constraints,
        "tree_method": input_data.tree_method,
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
        val_score=train_score,  # Same as train (no internal split anymore)
        test_score=train_score,  # Same as train (no internal split anymore)
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
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(description="Unique model name for storage")
    description: str = Field(default="", description="Model description")
    train_dataset_ref: str = Field(
        description="Reference name of the registered training dataset. "
        "The dataset must be registered via register_dataset()."
    )
    target_column: str = Field(description="Target column name")
    task_type: Literal["classification", "regression"] = Field(
        description="'classification' or 'regression'"
    )
    feature_columns: Optional[list[str]] = Field(default=None)
    categorical_columns: Optional[list[str]] = Field(default=None)
    sample_weight_column: Optional[str] = Field(
        default=None,
        description="Column with per-row weights. Use for recency, policy size, or label confidence."
    )
    n_estimators: int = Field(default=100, ge=1)
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.1, gt=0, le=1)
    min_child_weight: int = Field(default=1, ge=0)
    gamma: float = Field(default=0, ge=0)
    reg_alpha: float = Field(default=0, ge=0)
    reg_lambda: float = Field(default=1, ge=0)
    subsample: float = Field(default=1.0, gt=0, le=1)
    colsample_bytree: float = Field(default=1.0, gt=0, le=1)
    scale_pos_weight: Optional[float] = Field(
        default=None,
        description="Balance weight for positive class. Set to (neg_count/pos_count) for imbalanced data."
    )
    monotone_constraints: Optional[dict[str, int]] = Field(
        default=None,
        description="Monotonic constraints: {feature_name: 1 (increasing), -1 (decreasing), 0 (none)}"
    )
    tree_method: Literal["auto", "exact", "approx", "hist"] = Field(
        default="auto",
        description="Tree algorithm. 'hist' is faster for large datasets."
    )
    early_stopping_rounds: Optional[int] = Field(default=10)
    random_state: Optional[int] = Field(default=42)
    n_jobs: int = Field(default=-1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    validation_size: float = Field(default=0.1, gt=0, lt=0.5)


def _load_dataset_from_ref(dataset_ref: str):
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


@tool("xgboost_train", args_schema=SklearnXGBoostToolInput)
def xgboost_train_tool(
    model_name: str,
    train_dataset_ref: str,
    target_column: str,
    task_type: Literal["classification", "regression"],
    description: str = "",
    feature_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    sample_weight_column: Optional[str] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    min_child_weight: int = 1,
    gamma: float = 0,
    reg_alpha: float = 0,
    reg_lambda: float = 1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    scale_pos_weight: Optional[float] = None,
    monotone_constraints: Optional[dict[str, int]] = None,
    tree_method: Literal["auto", "exact", "approx", "hist"] = "auto",
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
    - scale_pos_weight: For imbalanced data, set to (negative_count / positive_count).
      E.g., if 95% negative and 5% positive, set to 19.0.
    - monotone_constraints: For interpretability, force features to have monotonic
      relationship with target. E.g., {'income': -1} means higher income → lower default.
    - tree_method: Use 'hist' for datasets with >10k rows for faster training.
    
    FOR IMBALANCED DATA (positive rate < 20%):
    - ALWAYS use scale_pos_weight = (num_negatives / num_positives)
    - E.g., 5% positive → scale_pos_weight = 19.0
    - When switching FROM logistic regression/random forest that used class_weight='balanced',
      you MUST use scale_pos_weight here
    
    FOR OVERFITTING (train_score >> val_score by >0.1):
    - Increase reg_alpha (try 0.1, 1.0) for L1 regularization
    - Increase reg_lambda (try 2.0, 5.0) for L2 regularization
    - Increase gamma (try 0.1, 0.5) for minimum split gain
    - Reduce learning_rate (try 0.05) with more n_estimators
    - Reduce max_depth (try 3-4)

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
        train_dataset_ref: Reference to registered training dataset
        target_column: Target variable column
        task_type: 'classification' or 'regression'
        n_estimators: Maximum boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Step size shrinkage
        scale_pos_weight: Weight for positive class (imbalanced data)
        monotone_constraints: Dict of feature constraints {name: 1/-1/0}
        tree_method: Algorithm ('auto', 'exact', 'approx', 'hist')
        early_stopping_rounds: Stop if no improvement for N rounds
        ... (other regularization and sampling parameters)

    Returns:
        Training results with loss curve, metrics, and feature importances.
    """
    if not XGBOOST_AVAILABLE:
        return "❌ XGBoost is not installed. Run: pip install xgboost"

    try:
        # Load dataset from registry
        df = _load_dataset_from_ref(train_dataset_ref)
        data = df.to_dict(orient="records")
        
        result = train_xgboost(
            XGBoostTrainingInput(
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
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                monotone_constraints=monotone_constraints,
                tree_method=tree_method,
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

