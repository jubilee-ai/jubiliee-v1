"""
Naive Bayes Training Tool for AI Data Scientist Agent

A LangChain tool that dynamically trains sklearn Naive Bayes models
for binary/multiclass classification tasks. Supports Gaussian,
Multinomial, and Complement NB variants.
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder,
                                   StandardScaler)


class NaiveBayesTrainingInput(BaseModel):
    """Input schema for training a Naive Bayes model."""
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        description="Unique name for this model. Used for storage and retrieval. "
                    "Example: 'spam_classifier_v1', 'sentiment_nb'"
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this model does. "
                    "Example: 'Classifies emails as spam or not spam'"
    )

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
                    "Numeric columns will be scaled."
    )
    sample_weight_column: Optional[str] = Field(
        default=None,
        description="Column containing per-row sample weights. Excluded from features."
    )

    variant: Literal["gaussian", "multinomial", "complement"] = Field(
        default="gaussian",
        description="Naive Bayes variant. "
                    "'gaussian': continuous features, assumes normal distribution per class (default). "
                    "'multinomial': for count/frequency data (e.g. word counts, tf-idf). "
                    "'complement': improved multinomial for imbalanced datasets."
    )
    var_smoothing: float = Field(
        default=1e-9,
        description="(Gaussian only) Portion of the largest variance of all features added to "
                    "variances for stability. Increase (e.g. 1e-5, 1e-3) if model is overfitting "
                    "or features have very small variance.",
        gt=0
    )
    alpha: float = Field(
        default=1.0,
        description="(Multinomial/Complement only) Additive (Laplace/Lidstone) smoothing parameter. "
                    "1.0 = Laplace smoothing. 0 = no smoothing (risk of zero probabilities). "
                    "Larger values = stronger smoothing.",
        ge=0.0
    )
    fit_prior: bool = Field(
        default=True,
        description="Whether to learn class prior probabilities from data. "
                    "If False, uses uniform priors."
    )
    class_prior: Optional[list[float]] = Field(
        default=None,
        description="Prior probabilities of the classes. If specified, priors are not "
                    "adjusted according to the data. Must sum to 1.0. "
                    "Example for binary: [0.3, 0.7]"
    )
    test_size: float = Field(
        default=0.2,
        description="Fraction of data to hold out for evaluation (0.0 to 1.0). Default 0.2 = 20% test set.",
        gt=0.0,
        lt=1.0
    )


class NaiveBayesTrainingOutput(BaseModel):
    """Output schema for trained Naive Bayes model."""

    success: bool = Field(description="Whether training completed successfully")
    model_type: str = Field(default="sklearn_naive_bayes")
    variant: str = Field(description="NB variant used (gaussian, multinomial, complement)")

    n_samples: int = Field(description="Total number of samples in dataset")
    n_features: int = Field(description="Number of features used")
    n_classes: int = Field(description="Number of unique classes in target")
    class_distribution: dict[str, int] = Field(description="Count of samples per class")
    feature_names: list[str] = Field(description="Names of features used")

    train_accuracy: float = Field(description="Accuracy on training set")
    test_accuracy: float = Field(description="Accuracy on held-out test set")
    test_roc_auc: Optional[float] = Field(description="ROC-AUC score on test set (binary or OvR for multiclass)")
    classification_report: str = Field(description="Full sklearn classification report (precision, recall, F1)")
    confusion_matrix: list[list[int]] = Field(description="Confusion matrix as nested list")

    class_log_prior: list[float] = Field(description="Log prior probabilities of each class")

    saved_path: str = Field(description="Path where model was saved")
    model_name: str = Field(description="Name of the model in the registry")
    training_timestamp: str = Field(description="ISO timestamp of training completion")
    discretized_target: bool = Field(default=False, description="Whether continuous target was auto-discretized")


def train_naive_bayes(input_data: NaiveBayesTrainingInput) -> NaiveBayesTrainingOutput:
    """
    Train a Naive Bayes model with preprocessing pipeline.

    Creates a full sklearn Pipeline with:
    - StandardScaler (Gaussian) or MinMaxScaler (Multinomial/Complement) for numeric features
    - OneHotEncoder for categorical features
    - The selected Naive Bayes classifier

    This ensures inference uses identical transformations as training.
    """
    df = pd.DataFrame(input_data.data)

    if input_data.target_column not in df.columns:
        raise ValueError(f"Target column '{input_data.target_column}' not found in data. "
                        f"Available columns: {list(df.columns)}")

    sample_weights = None
    if input_data.sample_weight_column:
        if input_data.sample_weight_column not in df.columns:
            raise ValueError(f"Sample weight column '{input_data.sample_weight_column}' not found in data.")
        sample_weights = df[input_data.sample_weight_column].values

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

    X = df[feature_cols]
    y = df[input_data.target_column]

    categorical_cols = input_data.categorical_columns or []
    categorical_cols = [c for c in categorical_cols if c in feature_cols]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    if not input_data.categorical_columns:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()

    # --- Auto-discretize continuous targets for classification ---
    n_unique = y.nunique()
    target_is_continuous = (
        np.issubdtype(y.dtype, np.floating)
        and n_unique > 20
    )
    discretized_target = False
    if target_is_continuous:
        median_val = y.median()
        y = (y > median_val).astype(int)
        df[input_data.target_column] = y
        discretized_target = True

    if input_data.variant == "gaussian":
        num_scaler = StandardScaler()
    else:
        num_scaler = MinMaxScaler()

    transformers = []
    if numeric_cols:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', num_scaler),
        ])
        transformers.append(('num', num_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough'
    )

    if input_data.variant == "gaussian":
        classifier = GaussianNB(
            var_smoothing=input_data.var_smoothing,
            priors=input_data.class_prior,
        )
    elif input_data.variant == "multinomial":
        classifier = MultinomialNB(
            alpha=input_data.alpha,
            fit_prior=input_data.fit_prior,
            class_prior=input_data.class_prior,
        )
    else:
        classifier = ComplementNB(
            alpha=input_data.alpha,
            fit_prior=input_data.fit_prior,
            class_prior=input_data.class_prior,
            norm=True,
        )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    from sklearn.model_selection import train_test_split

    min_class_count = y.value_counts().min()
    can_stratify = min_class_count >= 2
    stratify_arg = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=input_data.test_size, random_state=42, stratify=stratify_arg
    )
    train_weights, test_weights = None, None
    if sample_weights is not None:
        train_weights, test_weights = train_test_split(
            sample_weights, test_size=input_data.test_size, random_state=42, stratify=stratify_arg
        )

    if train_weights is not None:
        pipeline.fit(X_train, y_train, classifier__sample_weight=train_weights)
    else:
        pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)

    classes = pipeline.classes_
    n_classes = len(classes)
    try:
        if n_classes == 2:
            test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
        else:
            test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
    except ValueError:
        test_roc_auc = None

    clf_report = classification_report(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()

    try:
        feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()
    except AttributeError:
        feature_names_out = feature_cols

    nb_model = pipeline.named_steps['classifier']
    if hasattr(nb_model, 'class_log_prior_'):
        class_log_prior = nb_model.class_log_prior_.tolist()
    else:
        class_log_prior = np.log(nb_model.class_prior_).tolist() if hasattr(nb_model, 'class_prior_') else []

    class_dist = y.value_counts().to_dict()
    class_dist = {str(k): int(v) for k, v in class_dist.items()}

    # Re-fit on the full dataset so the saved model uses all available data.
    if sample_weights is not None:
        pipeline.fit(X, y, classifier__sample_weight=sample_weights)
    else:
        pipeline.fit(X, y)

    save_path = generate_model_path(input_data.model_name)
    joblib.dump(pipeline, save_path)

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "test_roc_auc": test_roc_auc,
    }

    hyperparameters: dict = {
        "variant": input_data.variant,
        "fit_prior": input_data.fit_prior,
    }
    if input_data.variant == "gaussian":
        hyperparameters["var_smoothing"] = input_data.var_smoothing
    else:
        hyperparameters["alpha"] = input_data.alpha
    if input_data.class_prior is not None:
        hyperparameters["class_prior"] = input_data.class_prior
    if discretized_target:
        hyperparameters["target_discretized"] = True
        hyperparameters["target_discretization_threshold"] = float(median_val)

    register_model(
        model_name=input_data.model_name,
        model_path=save_path,
        model_type="sklearn_naive_bayes",
        description=input_data.description,
        metrics=metrics,
        feature_names=feature_names_out,
        target_column=input_data.target_column,
        hyperparameters=hyperparameters,
        training_samples=len(df),
        classes=[str(c) for c in classes],
    )

    return NaiveBayesTrainingOutput(
        success=True,
        variant=input_data.variant,
        n_samples=len(df),
        n_features=len(feature_names_out),
        n_classes=n_classes,
        class_distribution=class_dist,
        feature_names=feature_names_out,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        test_roc_auc=test_roc_auc,
        classification_report=clf_report,
        confusion_matrix=conf_matrix,
        class_log_prior=class_log_prior,
        saved_path=save_path,
        model_name=input_data.model_name,
        training_timestamp=datetime.now(timezone.utc).isoformat(),
        discretized_target=discretized_target,
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class SklearnNaiveBayesToolInput(BaseModel):
    """Input for training a sklearn Naive Bayes model."""

    model_name: str = Field(
        description="Unique name for this model. Used to save, load, and reference the model. "
                    "Example: 'spam_nb_v1', 'sentiment_classifier'"
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
    variant: Literal["gaussian", "multinomial", "complement"] = Field(
        default="gaussian",
        description="NB variant: 'gaussian' (continuous features, default), "
                    "'multinomial' (count data), 'complement' (imbalanced data). Default: gaussian"
    )
    var_smoothing: float = Field(
        default=1e-9,
        description="(Gaussian only) Variance smoothing. Increase if overfitting. Default: 1e-9",
        gt=0
    )
    alpha: float = Field(
        default=1.0,
        description="(Multinomial/Complement only) Smoothing parameter. 1.0=Laplace. Default: 1.0",
        ge=0.0
    )
    fit_prior: bool = Field(
        default=True,
        description="Learn class priors from data. Set False for uniform priors. Default: True"
    )
    class_prior: Optional[list[float]] = Field(
        default=None,
        description="Explicit class priors (must sum to 1.0). If None, learned from data."
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

    data_tools_path = str(Path(__file__).parent.parent.parent / "data-tools")
    if data_tools_path not in sys.path:
        sys.path.insert(0, data_tools_path)

    from utils import get_registered_dataset

    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry. "
                        "Make sure to register it with register_dataset() first.")
    return df


@tool("sklearn_naive_bayes", args_schema=SklearnNaiveBayesToolInput)
def sklearn_naive_bayes_tool(
    model_name: str,
    train_dataset_ref: str,
    target_column: str,
    description: str = "",
    feature_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    sample_weight_column: Optional[str] = None,
    variant: Literal["gaussian", "multinomial", "complement"] = "gaussian",
    var_smoothing: float = 1e-9,
    alpha: float = 1.0,
    fit_prior: bool = True,
    class_prior: Optional[list[float]] = None,
    test_size: float = 0.2,
) -> str:
    """Train a Naive Bayes classifier (sklearn) on tabular data with automatic preprocessing.

    WHAT IT DOES:
    Trains a Naive Bayes model with a full preprocessing pipeline:
    - StandardScaler (Gaussian) or MinMaxScaler (Multinomial/Complement) for numeric features
    - OneHotEncoder for categorical features
    - Selected Naive Bayes classifier variant

    The complete pipeline is saved as one artifact, ensuring inference uses identical transformations.

    WHEN TO USE:
    - Fast baseline classifier before trying more complex models
    - Small to medium datasets where training speed matters
    - Text or document classification (use multinomial or complement variant)
    - High-dimensional feature spaces where tree-based models may struggle
    - When you need probabilistic class predictions
    - Imbalanced datasets (use complement variant)
    - When training data is limited — NB needs very few samples to estimate parameters

    WHEN NOT TO USE:
    - Regression problems (continuous targets) — NB is classification only
    - When features have strong dependencies/correlations (violates naive assumption)
    - When you need well-calibrated probability estimates
    - Complex non-linear decision boundaries (tree-based models will outperform)

    VARIANT SELECTION:
    - 'gaussian' (DEFAULT): Best for general tabular data with continuous features.
      Assumes features follow a normal distribution within each class.
    - 'multinomial': Best for count/frequency data (word counts, tf-idf, event counts).
      Requires non-negative features — pipeline uses MinMaxScaler automatically.
    - 'complement': Improved multinomial variant designed for IMBALANCED datasets.
      Uses complement class statistics, often outperforms standard multinomial.

    HYPERPARAMETER GUIDANCE:
    - var_smoothing (Gaussian): Start with 1e-9 default. If overfitting, increase to 1e-5 or 1e-3.
      This adds a fraction of the largest variance to all feature variances for stability.
    - alpha (Multinomial/Complement): Start with 1.0 (Laplace smoothing).
      If underfitting, try smaller values (0.1, 0.01). If features are sparse, keep at 1.0.
    - fit_prior: Keep True unless you have domain knowledge suggesting uniform class priors.
    - class_prior: Only set if you want to override learned priors with domain knowledge.

    FOR IMBALANCED DATA (positive rate < 20%):
    - Use variant='complement' — specifically designed for class imbalance
    - Or set class_prior to reflect the desired decision boundary

    OUTPUT INTERPRETATION:
    - test_accuracy: Primary metric. Compare to baseline (majority class rate).
    - test_roc_auc: Ranking quality. NB often has decent AUC even if accuracy is moderate.
    - classification_report: Check precision/recall per class, especially minority class.
    - class_log_prior: Log prior probabilities — verify they reflect the class distribution.

    STORAGE:
    Models are automatically saved to the trained_models/ directory and registered
    in the model registry. Use list_trained_models to see all models, and
    predict_with_model to make predictions with a saved model.

    Args:
        model_name: Unique name for this model (used to save and load)
        description: Human-readable description of the model
        train_dataset_ref: Reference name of registered training dataset
        target_column: Name of the target column to predict
        feature_columns: Which columns to use as features (None = all except target)
        categorical_columns: Which columns are categorical (None = auto-detect)
        sample_weight_column: Column with per-row sample weights
        variant: NB variant — 'gaussian', 'multinomial', or 'complement'
        var_smoothing: Variance smoothing for Gaussian variant
        alpha: Smoothing parameter for Multinomial/Complement variants
        fit_prior: Whether to learn class priors from data
        class_prior: Explicit class prior probabilities (must sum to 1.0)
        test_size: Fraction of data for test set evaluation

    Returns:
        Formatted string with training results including accuracy, ROC-AUC,
        classification report, class priors, and model registry info.
    """
    try:
        df = _load_dataset_from_ref(train_dataset_ref)
        data = df.to_dict(orient="records")

        result = train_naive_bayes(NaiveBayesTrainingInput(
            model_name=model_name,
            description=description,
            data=data,
            target_column=target_column,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            sample_weight_column=sample_weight_column,
            variant=variant,
            var_smoothing=var_smoothing,
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            test_size=test_size,
        ))

        variant_label = {
            "gaussian": "Gaussian",
            "multinomial": "Multinomial",
            "complement": "Complement",
        }[result.variant]

        output_lines = [
            "=" * 60,
            f"NAIVE BAYES ({variant_label}) TRAINING COMPLETE",
            "=" * 60,
            "",
        ]
        if result.discretized_target:
            output_lines.extend([
                "⚠️ TARGET AUTO-DISCRETIZED",
                f"  Original target '{target_column}' was continuous.",
                "  Binarized at median: 0 = below/equal median, 1 = above median.",
                "",
            ])
        output_lines.extend([
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
            "📊 CLASS LOG PRIORS:",
        ])

        for i, log_prior in enumerate(result.class_log_prior):
            prior = np.exp(log_prior)
            output_lines.append(f"  Class {i}: log_prior={log_prior:.4f} (prior={prior:.4f})")

        output_lines.extend([
            "",
            f"💾 MODEL REGISTERED: {result.model_name}",
            f"   Path: {result.saved_path}",
            f"   Variant: {variant_label}",
            "   Use predict_with_model or list_trained_models to access",
        ])

        output_lines.extend([
            "",
            f"⏱️ Timestamp: {result.training_timestamp}",
            "=" * 60,
        ])

        return "\n".join(output_lines)

    except Exception as e:
        return (f"❌ TRAINING FAILED\nError: {str(e)}\n\nPlease check:\n"
                "- Data format (list of dicts with consistent keys)\n"
                "- Target column exists in data\n"
                "- Feature columns exist if specified\n"
                "- For multinomial/complement: features should be non-negative counts or frequencies")
