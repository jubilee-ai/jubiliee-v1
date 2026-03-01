"""
Logistic Regression Training Tool for AI Data Scientist Agent

A LangChain tool that dynamically trains sklearn LogisticRegression models
for binary/multiclass classification tasks. Designed for underwriting,
risk assessment, and governed decisioning use cases.
"""

import os
from datetime import datetime, timezone
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from langchain.tools import tool
from model_storage import generate_model_path, register_model
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LogisticRegressionTrainingInput(BaseModel):
    """Input schema for training a logistic regression model."""
    model_config = ConfigDict(extra="forbid")
    
    # Model identification
    model_name: str = Field(
        description="Unique name for this model. Used for storage and retrieval. "
                    "Example: 'loan_default_v1', 'credit_risk_prod'"
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this model does. "
                    "Example: 'Predicts loan default probability for retail customers'"
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
        description="List of column names to use as features. If None, uses all columns except target_column."
    )
    categorical_columns: Optional[list[str]] = Field(
        default=None,
        description="List of column names that are categorical (will be one-hot encoded). "
                    "Numeric columns will be standardized."
    )
    sample_weight_column: Optional[str] = Field(
        default=None,
        description="Column containing per-row sample weights. Excluded from features."
    )
    
    # Model hyperparameters
    C: float = Field(
        default=1.0,
        description="Inverse regularization strength. Smaller = stronger regularization. "
                    "Range: (0, inf). Default 1.0. Use 0.01-0.1 for high regularization, 10-100 for low.",
        gt=0
    )
    l1_ratio: float = Field(
        default=0.0,
        description="Elastic-Net mixing: 0.0 = pure L2 (Ridge), 1.0 = pure L1 (Lasso), "
                    "0 < x < 1 = Elastic-Net. L1 promotes sparsity (feature selection).",
        ge=0.0,
        le=1.0
    )
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = Field(
        default="lbfgs",
        description="Optimization algorithm. 'lbfgs' is a good default. Use 'saga' for L1/Elastic-Net, "
                    "'liblinear' for small datasets, 'newton-cholesky' for n_samples >> n_features."
    )
    max_iter: int = Field(
        default=100,
        description="Maximum iterations for solver convergence. Increase (500-1000) if convergence warnings appear.",
        ge=1
    )
    class_weight: Optional[Literal["balanced"]] = Field(
        default=None,
        description="Set to 'balanced' to auto-adjust weights inversely proportional to class frequencies. "
                    "Useful for imbalanced datasets (e.g., rare defaults)."
    )
    fit_intercept: bool = Field(
        default=True,
        description="Whether to add a bias/intercept term. Usually True unless data is already centered."
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility. Set to fixed value for consistent results."
    )
    
    # Training configuration
    test_size: float = Field(
        default=0.2,
        description="Fraction of data to hold out for evaluation (0.0 to 1.0). Default 0.2 = 20% test set.",
        gt=0.0,
        lt=1.0
    )


class LogisticRegressionTrainingOutput(BaseModel):
    """Output schema for trained logistic regression model."""
    
    success: bool = Field(description="Whether training completed successfully")
    model_type: str = Field(default="sklearn_logistic_regression")
    
    # Dataset info
    n_samples: int = Field(description="Total number of samples in dataset")
    n_features: int = Field(description="Number of features used")
    n_classes: int = Field(description="Number of unique classes in target")
    class_distribution: dict[str, int] = Field(description="Count of samples per class")
    feature_names: list[str] = Field(description="Names of features used")
    
    # Metrics
    train_accuracy: float = Field(description="Accuracy on training set")
    test_accuracy: float = Field(description="Accuracy on held-out test set")
    test_roc_auc: Optional[float] = Field(description="ROC-AUC score on test set (binary or OvR for multiclass)")
    classification_report: str = Field(description="Full sklearn classification report (precision, recall, F1)")
    confusion_matrix: list[list[int]] = Field(description="Confusion matrix as nested list")
    
    # Model info
    coefficients: dict[str, list[float]] = Field(description="Feature coefficients per class")
    intercepts: list[float] = Field(description="Intercept values per class")
    n_iterations: int = Field(description="Actual iterations to converge")
    
    # Artifacts
    saved_path: str = Field(description="Path where model was saved")
    model_name: str = Field(description="Name of the model in the registry")
    training_timestamp: str = Field(description="ISO timestamp of training completion")


def train_logistic_regression(input_data: LogisticRegressionTrainingInput) -> LogisticRegressionTrainingOutput:
    """
    Train a logistic regression model with preprocessing pipeline.
    
    Creates a full sklearn Pipeline with:
    - OneHotEncoder for categorical features
    - StandardScaler for numeric features  
    - LogisticRegression classifier
    
    This ensures inference uses identical transformations as training.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data.data)
    
    # Validate target column exists
    if input_data.target_column not in df.columns:
        raise ValueError(f"Target column '{input_data.target_column}' not found in data. "
                        f"Available columns: {list(df.columns)}")
    
    # Extract sample weights if provided
    sample_weights = None
    if input_data.sample_weight_column:
        if input_data.sample_weight_column not in df.columns:
            raise ValueError(f"Sample weight column '{input_data.sample_weight_column}' not found in data.")
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
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    
    # Build preprocessing pipeline
    transformers = []
    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )
    
    # Validate solver/l1_ratio compatibility
    l1_ratio = input_data.l1_ratio
    solver = input_data.solver
    
    if l1_ratio > 0 and l1_ratio < 1 and solver != 'saga':
        # Elastic-Net requires saga
        solver = 'saga'
    elif l1_ratio == 1.0 and solver not in ('saga', 'liblinear'):
        # Pure L1 requires saga or liblinear
        solver = 'saga'
    
    # Build full pipeline
    model = LogisticRegression(
        C=input_data.C,
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=input_data.max_iter,
        class_weight=input_data.class_weight,
        fit_intercept=input_data.fit_intercept,
        random_state=input_data.random_state,
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    # NOTE: No internal train_test_split - data is already split by the pipeline
    # The training agent passes pre-split training data here
    # Validation/test evaluation happens via evaluate_model tool
    
    # Fit pipeline on ALL provided data (it's already the training set)
    if sample_weights is not None:
        pipeline.fit(X, y, classifier__sample_weight=sample_weights)
    else:
        pipeline.fit(X, y)
    
    # Predictions on training data (for sanity check metrics)
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)
    
    # Training metrics (use for sanity check, not model selection)
    train_acc = accuracy_score(y, y_pred)
    
    # ROC-AUC (handle binary vs multiclass)
    classes = pipeline.classes_
    n_classes = len(classes)
    try:
        if n_classes == 2:
            train_roc_auc = roc_auc_score(y, y_proba[:, 1])
        else:
            train_roc_auc = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
    except ValueError:
        train_roc_auc = None
    
    clf_report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred).tolist()
    
    # Extract coefficients with feature names
    classifier = pipeline.named_steps['classifier']
    
    # Get transformed feature names
    try:
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()
    except AttributeError:
        feature_names_out = feature_cols
    
    # Coefficients per class
    # Note: For binary classification, sklearn stores only 1 row of coefficients
    # representing log-odds for the positive class (classes_[1])
    coef = classifier.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    
    if n_classes == 2:
        # Binary case: coef has shape (1, n_features)
        # Show coefficients for positive class only
        coefficients = {str(classes[1]): coef[0].tolist()}
    else:
        # Multiclass case: coef has shape (n_classes, n_features)
        coefficients = {str(c): coef[i].tolist() for i, c in enumerate(classes)}
    intercepts = classifier.intercept_.tolist()
    
    # Class distribution
    class_dist = y.value_counts().to_dict()
    class_dist = {str(k): int(v) for k, v in class_dist.items()}
    
    # Save model to trained_models directory
    save_path = generate_model_path(input_data.model_name)
    joblib.dump(pipeline, save_path)
    
    # Prepare metrics and hyperparameters for registry
    # NOTE: These are TRAINING metrics only - use evaluate_model for val/test metrics
    metrics = {
        "train_accuracy": train_acc,
        "train_roc_auc": train_roc_auc,
    }
    
    hyperparameters = {
        "C": input_data.C,
        "l1_ratio": input_data.l1_ratio,
        "solver": solver,  # Use the potentially adjusted solver
        "max_iter": input_data.max_iter,
        "class_weight": input_data.class_weight,
        "fit_intercept": input_data.fit_intercept,
        "random_state": input_data.random_state,
    }
    
    # Register model in the registry
    register_model(
        model_name=input_data.model_name,
        model_path=save_path,
        model_type="sklearn_logistic_regression",
        description=input_data.description,
        metrics=metrics,
        feature_names=feature_names_out,
        target_column=input_data.target_column,
        hyperparameters=hyperparameters,
        training_samples=len(df),
        classes=[str(c) for c in classes],
    )
    
    return LogisticRegressionTrainingOutput(
        success=True,
        n_samples=len(df),
        n_features=len(feature_names_out),
        n_classes=n_classes,
        class_distribution=class_dist,
        feature_names=feature_names_out,
        train_accuracy=train_acc,
        test_accuracy=train_acc,  # Same as train (no internal split anymore)
        test_roc_auc=train_roc_auc,  # Same as train (no internal split anymore)
        classification_report=clf_report,
        confusion_matrix=conf_matrix,
        coefficients=coefficients,
        intercepts=intercepts,
        n_iterations=int(classifier.n_iter_[0]) if hasattr(classifier.n_iter_, '__len__') else int(classifier.n_iter_),
        saved_path=save_path,
        model_name=input_data.model_name,
        training_timestamp=datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class SklearnLogisticRegressionToolInput(BaseModel):
    """Input for training a sklearn Logistic Regression model."""
    
    model_name: str = Field(
        description="Unique name for this model. Used to save, load, and reference the model. "
                    "Example: 'loan_default_v1', 'credit_risk_classifier'"
    )
    description: str = Field(
        default="",
        description="Human-readable description of the model's purpose."
    )
    train_dataset_ref: str = Field(
        description="Reference name of the registered training dataset. "
                    "The dataset must be registered via register_dataset(). "
                    "Example: 'basic_train', 'pipeline_train_features'"
    )
    target_column: str = Field(
        description="Name of target column to predict. Must exist in the dataset."
    )
    feature_columns: Optional[list[str]] = Field(
        default=None,
        description="Feature column names to use. If None, uses all columns except target."
    )
    categorical_columns: Optional[list[str]] = Field(
        default=None,
        description="Columns to one-hot encode. If None, auto-detects from dtypes."
    )
    sample_weight_column: Optional[str] = Field(
        default=None,
        description="Column containing per-row sample weights. Use for: weighting recent data more, "
                    "weighting by policy size, or confidence in labels. Column is excluded from features."
    )
    C: float = Field(
        default=1.0,
        description="Regularization strength (inverse). Lower = stronger regularization. Default: 1.0",
        gt=0
    )
    l1_ratio: float = Field(
        default=0.0,
        description="L1 vs L2 mixing: 0.0=L2 (Ridge), 1.0=L1 (Lasso), 0<x<1=ElasticNet. Default: 0.0",
        ge=0.0,
        le=1.0
    )
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = Field(
        default="lbfgs",
        description="Optimizer algorithm. 'lbfgs' default. Use 'saga' for L1/ElasticNet."
    )
    max_iter: int = Field(
        default=100,
        description="Max solver iterations. Increase if model doesn't converge. Default: 100",
        ge=1
    )
    class_weight: Optional[Literal["balanced"]] = Field(
        default=None,
        description="Set 'balanced' for imbalanced classes (auto-adjusts weights)."
    )
    fit_intercept: bool = Field(
        default=True,
        description="Include bias term. Default: True"
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility. Default: 42"
    )
    test_size: float = Field(
        default=0.2,
        description="Fraction held out for evaluation (0-1). Default: 0.2",
        gt=0.0,
        lt=1.0
    )


def _load_dataset_from_ref(dataset_ref: str) -> pd.DataFrame:
    """Load a dataset from the registry by reference name."""
    import sys
    from pathlib import Path
    
    # Add data-tools path
    data_tools_path = str(Path(__file__).parent.parent.parent / "data-tools")
    if data_tools_path not in sys.path:
        sys.path.insert(0, data_tools_path)
    
    from utils import get_registered_dataset
    
    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry. "
                        "Make sure to register it with register_dataset() first.")
    return df


@tool("sklearn_logistic_regression", args_schema=SklearnLogisticRegressionToolInput)
def sklearn_logistic_regression_tool(
    model_name: str,
    train_dataset_ref: str,
    target_column: str,
    description: str = "",
    feature_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    sample_weight_column: Optional[str] = None,
    C: float = 1.0,
    l1_ratio: float = 0.0,
    solver: Literal["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] = "lbfgs",
    max_iter: int = 100,
    class_weight: Optional[Literal["balanced"]] = None,
    fit_intercept: bool = True,
    random_state: Optional[int] = 42,
    test_size: float = 0.2,
) -> str:
    """Train a Logistic Regression classifier (sklearn) on tabular data with automatic preprocessing.
    
    WHAT IT DOES:
    Trains a logistic regression model with a full preprocessing pipeline:
    - StandardScaler for numeric features
    - OneHotEncoder for categorical features  
    - LogisticRegression classifier
    
    The complete pipeline is saved as one artifact, ensuring inference uses identical transformations.
    
    WHEN TO USE:
    - Binary classification: approve/decline, risk flag, lapse yes/no
    - Multiclass classification: risk tiers, customer segments
    - When interpretability matters: coefficient weights show feature importance
    - As a strong baseline before trying complex models
    - For governed decisioning where model transparency is required
    
    WHEN NOT TO USE:
    - Regression problems (continuous targets) - use LinearRegression instead
    - High-dimensional sparse data - consider SGDClassifier
    - Non-linear relationships - consider tree-based models
    - Very large datasets (>1M rows) - consider incremental learning
    
    HYPERPARAMETER GUIDANCE:
    - C: Start with 1.0. If overfitting, try 0.1 or 0.01. If underfitting, try 10 or 100.
    - l1_ratio: Use 0.0 (L2) by default. Use 1.0 (L1) for feature selection/sparsity.
      Try l1_ratio=0.5 (ElasticNet) with solver='saga' when features > 10.
    - solver: 'lbfgs' works for most cases. Switch to 'saga' if using L1 or ElasticNet.
    - class_weight: Set to 'balanced' if classes are imbalanced (e.g., 95% negative, 5% positive).
    - max_iter: Increase to 500-1000 if you see convergence warnings.
    
    FOR IMBALANCED DATA (positive rate < 20%):
    - ALWAYS use class_weight='balanced'
    
    FOR OVERFITTING (train_score >> val_score by >0.1):
    - Reduce C (try 0.1 or 0.01 for stronger regularization)
    - Try l1_ratio=0.5 with solver='saga' for feature selection
    
    OUTPUT INTERPRETATION:
    - test_accuracy: Primary metric. Compare to baseline (majority class rate).
    - test_roc_auc: Ranking quality. >0.7 is acceptable, >0.8 is good, >0.9 is excellent.
    - classification_report: Precision/recall/F1 per class. Check minority class metrics.
    - coefficients: Positive = increases probability of class, negative = decreases.
    
    STORAGE:
    Models are automatically saved to the trained_models/ directory and registered
    in the model registry. Use list_trained_models to see all models, and
    predict_with_model to make predictions with a saved model.
    
    EXAMPLE:
    ```
    # First register your training data
    register_dataset("my_train_data", train_df)
    
    # Then train using the reference
    result = sklearn_logistic_regression(
        model_name="loan_default_v1",
        description="Predicts loan default for retail customers",
        train_dataset_ref="my_train_data",
        target_column="default",
        categorical_columns=["employed"],
        C=0.1,
        class_weight="balanced",
    )
    ```
    
    Args:
        model_name: Unique name for this model (used to save and load)
        description: Human-readable description of the model
        train_dataset_ref: Reference name of registered training dataset
        target_column: Name of the target column to predict
        feature_columns: Which columns to use as features (None = all except target)
        categorical_columns: Which columns are categorical (None = auto-detect)
        C: Inverse regularization strength (>0, smaller = stronger regularization)
        l1_ratio: L1 vs L2 penalty mixing (0.0=L2, 1.0=L1, between=ElasticNet)
        solver: Optimization algorithm
        max_iter: Maximum iterations for convergence
        class_weight: Set 'balanced' for imbalanced datasets
        fit_intercept: Whether to fit intercept term
        random_state: Random seed for reproducibility
        test_size: Fraction of data for test set evaluation
    
    Returns:
        Formatted string with training results including accuracy, ROC-AUC,
        classification report, feature coefficients, and model registry info.
    """
    try:
        # Load dataset from registry
        df = _load_dataset_from_ref(train_dataset_ref)
        data = df.to_dict(orient="records")
        
        result = train_logistic_regression(LogisticRegressionTrainingInput(
            model_name=model_name,
            description=description,
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            sample_weight_column=sample_weight_column,
            C=C,
            l1_ratio=l1_ratio,
            solver=solver,
            max_iter=max_iter,
            class_weight=class_weight,
            fit_intercept=fit_intercept,
            random_state=random_state,
            test_size=test_size,
        ))
        
        # Format output for agent consumption
        output_lines = [
            "=" * 60,
            "LOGISTIC REGRESSION TRAINING COMPLETE",
            "=" * 60,
            "",
            "📊 DATASET SUMMARY",
            f"  Samples: {result.n_samples}",
            f"  Features: {result.n_features}",
            f"  Classes: {result.n_classes}",
            f"  Class Distribution: {result.class_distribution}",
            "",
            "📈 PERFORMANCE METRICS",
            f"  Train Accuracy: {result.train_accuracy:.4f}",
            f"  Test Accuracy:  {result.test_accuracy:.4f}",
            f"  Test ROC-AUC:   {result.test_roc_auc:.4f}" if result.test_roc_auc else "  Test ROC-AUC:   N/A",
            "",
            "📋 CLASSIFICATION REPORT",
            result.classification_report,
            "",
            "🔢 CONFUSION MATRIX",
            str(np.array(result.confusion_matrix)),
            "",
            "⚖️ MODEL COEFFICIENTS (top features by absolute magnitude):",
        ]
        
        # Show top coefficients per class
        for class_name, coefs in result.coefficients.items():
            coef_pairs = list(zip(result.feature_names, coefs))
            coef_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            top_coefs = coef_pairs[:5]
            output_lines.append(f"  Class '{class_name}':")
            for feat, coef in top_coefs:
                sign = "+" if coef > 0 else ""
                output_lines.append(f"    {feat}: {sign}{coef:.4f}")
        
        output_lines.extend([
            "",
            f"🔄 Convergence: {result.n_iterations} iterations",
        ])
        
        output_lines.extend([
            "",
            f"💾 MODEL REGISTERED: {result.model_name}",
            f"   Path: {result.saved_path}",
            "   Use predict_with_model or list_trained_models to access",
        ])
        
        output_lines.extend([
            "",
            f"⏱️ Timestamp: {result.training_timestamp}",
            "=" * 60,
        ])
        
        return "\n".join(output_lines)
        
    except Exception as e:
        return f"❌ TRAINING FAILED\nError: {str(e)}\n\nPlease check:\n- Data format (list of dicts with consistent keys)\n- Target column exists in data\n- Feature columns exist if specified\n- Solver/penalty compatibility"

