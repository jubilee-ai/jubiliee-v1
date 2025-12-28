"""
Generalized Linear Model (GLM) Training Tool for AI Data Scientist Agent

A LangChain tool that trains sklearn GLM models for insurance pricing and
actuarial modeling. Supports Poisson (frequency), Gamma (severity), and
Tweedie (combined) distributions - the foundation of actuarial pricing.
"""

import os
from datetime import datetime, timezone
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from langchain.tools import tool
from model_storage import generate_model_path, register_model
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (GammaRegressor, PoissonRegressor,
                                  TweedieRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class GLMTrainingInput(BaseModel):
    """Input schema for training a Generalized Linear Model."""

    # Model identification
    model_name: str = Field(
        description="Unique name for this model. Used for storage and retrieval. "
        "Example: 'claim_frequency_v1', 'severity_model_prod'"
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this model does. "
        "Example: 'Poisson model for auto claim frequency prediction'",
    )

    # Data specification
    data: list[dict] = Field(
        description="Training data as a list of dictionaries, where each dict is a row."
    )
    target_column: str = Field(
        description="Name of the column containing the target variable. "
        "For Poisson: count data (0, 1, 2, ...). For Gamma: positive continuous. "
        "For Tweedie: non-negative with point mass at zero."
    )
    feature_columns: Optional[list[str]] = Field(
        default=None,
        description="List of column names to use as features. If None, uses all except target.",
    )
    categorical_columns: Optional[list[str]] = Field(
        default=None,
        description="List of categorical column names (will be one-hot encoded).",
    )
    exposure_column: Optional[str] = Field(
        default=None,
        description="Column for exposure/offset (e.g., policy duration, years at risk). "
        "Used in Poisson to model rates instead of counts. "
        "If provided, predictions are per unit of exposure.",
    )

    # Distribution
    distribution: Literal["poisson", "gamma", "tweedie"] = Field(
        description="GLM distribution family:\n"
        "- 'poisson': For count data (claim frequency). Target must be non-negative integers.\n"
        "- 'gamma': For positive continuous data (claim severity). Target must be > 0.\n"
        "- 'tweedie': For non-negative data with zeros (total claim cost). "
        "Combines Poisson frequency and Gamma severity."
    )
    tweedie_power: float = Field(
        default=1.5,
        description="Tweedie power parameter (only used if distribution='tweedie'). "
        "1 < power < 2 for compound Poisson-Gamma (typical insurance). "
        "1.5 is a good default. Range: (1, 2) for insurance use cases.",
        gt=1,
        lt=2,
    )

    # Regularization
    alpha: float = Field(
        default=1.0,
        description="L2 regularization strength. 0 = no regularization. "
        "Higher = simpler model. Default: 1.0",
        ge=0,
    )

    # Solver settings
    solver: Literal["lbfgs", "newton-cholesky"] = Field(
        default="lbfgs",
        description="Optimization algorithm. 'lbfgs' is good default. "
        "'newton-cholesky' better for n_samples >> n_features.",
    )
    max_iter: int = Field(
        default=100,
        description="Maximum iterations for solver convergence. Increase if not converging.",
        ge=1,
    )
    fit_intercept: bool = Field(
        default=True,
        description="Whether to fit an intercept term. Usually True.",
    )

    # Training configuration
    test_size: float = Field(
        default=0.2,
        description="Fraction of data for test set. Default: 0.2",
        gt=0.0,
        lt=1.0,
    )
    random_state: Optional[int] = Field(
        default=42, description="Random seed for reproducibility."
    )


class GLMTrainingOutput(BaseModel):
    """Output schema for trained GLM model."""

    success: bool
    model_type: str = "sklearn_glm"
    distribution: str

    # Dataset info
    n_samples: int
    n_features: int
    feature_names: list[str]
    target_stats: dict[str, float]

    # Metrics
    train_deviance: float = Field(description="D² score on training set (like R² for GLMs)")
    test_deviance: float = Field(description="D² score on test set")
    train_mae: float
    test_mae: float
    train_rmse: float
    test_rmse: float

    # Model info
    coefficients: dict[str, float] = Field(description="Feature coefficients (log scale for GLM)")
    intercept: float
    n_iterations: int

    # Artifacts
    saved_path: str
    model_name: str
    training_timestamp: str


def train_glm(input_data: GLMTrainingInput) -> GLMTrainingOutput:
    """
    Train a Generalized Linear Model with preprocessing pipeline.
    """
    # Convert input data to DataFrame
    df = pd.DataFrame(input_data.data)

    # Validate target column exists
    if input_data.target_column not in df.columns:
        raise ValueError(
            f"Target column '{input_data.target_column}' not found. "
            f"Available: {list(df.columns)}"
        )

    # Determine feature columns
    exclude_cols = [input_data.target_column]
    if input_data.exposure_column:
        if input_data.exposure_column not in df.columns:
            raise ValueError(f"Exposure column '{input_data.exposure_column}' not found.")
        exclude_cols.append(input_data.exposure_column)

    if input_data.feature_columns:
        feature_cols = input_data.feature_columns
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Separate features and target
    X = df[feature_cols].copy()
    y = df[input_data.target_column].copy()

    # Validate target values based on distribution
    if input_data.distribution == "poisson":
        if (y < 0).any():
            raise ValueError("Poisson target must be non-negative (counts).")
    elif input_data.distribution == "gamma":
        if (y <= 0).any():
            raise ValueError("Gamma target must be strictly positive.")
    elif input_data.distribution == "tweedie":
        if (y < 0).any():
            raise ValueError("Tweedie target must be non-negative.")

    # Get exposure if provided
    exposure = None
    if input_data.exposure_column:
        exposure = df[input_data.exposure_column].values

    # Determine categorical vs numeric columns
    categorical_cols = input_data.categorical_columns or []
    categorical_cols = [c for c in categorical_cols if c in feature_cols]

    if not input_data.categorical_columns:
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Build preprocessing pipeline
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="passthrough")

    # Build the GLM model
    if input_data.distribution == "poisson":
        model = PoissonRegressor(
            alpha=input_data.alpha,
            fit_intercept=input_data.fit_intercept,
            solver=input_data.solver,
            max_iter=input_data.max_iter,
        )
    elif input_data.distribution == "gamma":
        model = GammaRegressor(
            alpha=input_data.alpha,
            fit_intercept=input_data.fit_intercept,
            solver=input_data.solver,
            max_iter=input_data.max_iter,
        )
    else:  # tweedie
        model = TweedieRegressor(
            power=input_data.tweedie_power,
            alpha=input_data.alpha,
            fit_intercept=input_data.fit_intercept,
            solver=input_data.solver,
            max_iter=input_data.max_iter,
        )

    # Create pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("glm", model)])

    # Train/test split
    if exposure is not None:
        X_train, X_test, y_train, y_test, exp_train, exp_test = train_test_split(
            X, y, exposure,
            test_size=input_data.test_size,
            random_state=input_data.random_state,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=input_data.test_size,
            random_state=input_data.random_state,
        )
        exp_train = exp_test = None

    # Fit pipeline
    # Note: For exposure, we need to fit the model differently
    if exp_train is not None:
        # Fit preprocessor first
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Fit GLM with sample_weight=exposure for rate modeling
        model.fit(X_train_processed, y_train, sample_weight=exp_train)
        
        # Create wrapper for predictions
        class GLMWithExposure:
            def __init__(self, preprocessor, model):
                self.preprocessor = preprocessor
                self.model = model
            
            def predict(self, X, exposure=None):
                X_processed = self.preprocessor.transform(X)
                pred = self.model.predict(X_processed)
                if exposure is not None:
                    pred = pred * exposure
                return pred
        
        pipeline = GLMWithExposure(preprocessor, model)
        
        # Predictions
        y_train_pred = model.predict(X_train_processed)
        y_test_pred = model.predict(X_test_processed)
        
        # Scores
        train_deviance = model.score(X_train_processed, y_train, sample_weight=exp_train)
        test_deviance = model.score(X_test_processed, y_test, sample_weight=exp_test)
    else:
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        # Scores (D² deviance explained)
        train_deviance = pipeline.score(X_train, y_train)
        test_deviance = pipeline.score(X_test, y_test)

    # Get feature names
    try:
        feature_names_out = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        feature_names_out = feature_cols

    # Calculate additional metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Get coefficients
    glm_model = model if exp_train is not None else pipeline.named_steps["glm"]
    coefficients = {name: float(coef) for name, coef in zip(feature_names_out, glm_model.coef_)}
    intercept = float(glm_model.intercept_)
    n_iterations = int(glm_model.n_iter_)

    # Target statistics
    target_stats = {
        "mean": float(y.mean()),
        "std": float(y.std()),
        "min": float(y.min()),
        "max": float(y.max()),
        "zeros_pct": float((y == 0).mean() * 100),
    }

    # Save model
    save_path = generate_model_path(input_data.model_name)
    joblib.dump(pipeline, save_path)

    # Register in registry
    metrics = {
        "train_deviance": train_deviance,
        "test_deviance": test_deviance,
        "test_score": test_deviance,  # For compatibility with list_models
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }

    hyperparameters = {
        "task_type": "regression",
        "distribution": input_data.distribution,
        "tweedie_power": input_data.tweedie_power if input_data.distribution == "tweedie" else None,
        "alpha": input_data.alpha,
        "solver": input_data.solver,
        "max_iter": input_data.max_iter,
        "fit_intercept": input_data.fit_intercept,
        "exposure_column": input_data.exposure_column,
        "random_state": input_data.random_state,
    }

    register_model(
        model_name=input_data.model_name,
        model_path=save_path,
        model_type="sklearn_glm",
        description=input_data.description,
        metrics=metrics,
        feature_names=feature_names_out,
        target_column=input_data.target_column,
        hyperparameters=hyperparameters,
        training_samples=len(df),
        classes=[],
    )

    return GLMTrainingOutput(
        success=True,
        distribution=input_data.distribution,
        n_samples=len(df),
        n_features=len(feature_names_out),
        feature_names=feature_names_out,
        target_stats=target_stats,
        train_deviance=train_deviance,
        test_deviance=test_deviance,
        train_mae=train_mae,
        test_mae=test_mae,
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        coefficients=coefficients,
        intercept=intercept,
        n_iterations=n_iterations,
        saved_path=save_path,
        model_name=input_data.model_name,
        training_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class SklearnGLMToolInput(BaseModel):
    """Input for training a sklearn GLM model."""

    model_name: str = Field(description="Unique model name for storage")
    description: str = Field(default="", description="Model description")
    data: list[dict] = Field(description="Training data as list of row dicts")
    target_column: str = Field(description="Target column name")
    distribution: Literal["poisson", "gamma", "tweedie"] = Field(
        description="GLM distribution: 'poisson' (counts), 'gamma' (positive continuous), 'tweedie' (zeros + positive)"
    )
    feature_columns: Optional[list[str]] = Field(default=None)
    categorical_columns: Optional[list[str]] = Field(default=None)
    exposure_column: Optional[str] = Field(
        default=None,
        description="Exposure column for rate modeling (e.g., policy duration)"
    )
    tweedie_power: float = Field(default=1.5, gt=1, lt=2)
    alpha: float = Field(default=1.0, ge=0)
    solver: Literal["lbfgs", "newton-cholesky"] = Field(default="lbfgs")
    max_iter: int = Field(default=100, ge=1)
    fit_intercept: bool = Field(default=True)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: Optional[int] = Field(default=42)


@tool("sklearn_glm", args_schema=SklearnGLMToolInput)
def sklearn_glm_tool(
    model_name: str,
    data: list[dict],
    target_column: str,
    distribution: Literal["poisson", "gamma", "tweedie"],
    description: str = "",
    feature_columns: Optional[list[str]] = None,
    categorical_columns: Optional[list[str]] = None,
    exposure_column: Optional[str] = None,
    tweedie_power: float = 1.5,
    alpha: float = 1.0,
    solver: Literal["lbfgs", "newton-cholesky"] = "lbfgs",
    max_iter: int = 100,
    fit_intercept: bool = True,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
) -> str:
    """Train a Generalized Linear Model (GLM) for insurance pricing and actuarial modeling.

    WHAT IT DOES:
    GLMs are the foundation of actuarial pricing models. They extend linear regression
    to handle non-normal distributions common in insurance:
    - Poisson: For count data (claim frequency)
    - Gamma: For positive continuous data (claim severity)
    - Tweedie: For data with zeros and positive values (total claim cost)

    All use a log link function, so coefficients represent multiplicative effects
    (e.g., coef=0.1 means 10% increase in expected value).

    WHEN TO USE:
    - **Poisson**: Predicting claim counts (0, 1, 2, 3...)
      Example: "How many claims will this policy have?"
    - **Gamma**: Predicting claim severity (always > 0)
      Example: "Given a claim occurs, how much will it cost?"
    - **Tweedie**: Predicting total claim cost (many zeros, some positive)
      Example: "What is the expected total loss for this policy?"

    INSURANCE USE CASES:
    - Auto insurance premium calculation
    - Health insurance claim cost prediction
    - Property damage severity modeling
    - Loss reserving and IBNR estimation
    - Workers compensation frequency/severity

    EXPOSURE MODELING:
    Use exposure_column for rate modeling:
    - Policy duration (0.5 = 6 months, 1.0 = full year)
    - Years at risk
    - Miles driven (for usage-based insurance)

    INTERPRETING COEFFICIENTS:
    GLM coefficients are on log scale. To interpret:
    - coef = 0.10 → exp(0.10) = 1.105 → 10.5% increase
    - coef = -0.20 → exp(-0.20) = 0.819 → 18.1% decrease

    EXAMPLE (Claim Frequency):
    ```
    result = sklearn_glm(
        model_name="auto_frequency_v1",
        description="Poisson model for auto claim frequency",
        data=policy_data,
        target_column="num_claims",
        distribution="poisson",
        exposure_column="policy_years",
        categorical_columns=["vehicle_type", "region"],
    )
    ```

    EXAMPLE (Claim Severity):
    ```
    result = sklearn_glm(
        model_name="auto_severity_v1",
        description="Gamma model for claim amount given claim occurred",
        data=claims_data,  # Only records where claim > 0
        target_column="claim_amount",
        distribution="gamma",
    )
    ```

    EXAMPLE (Pure Premium / Total Cost):
    ```
    result = sklearn_glm(
        model_name="pure_premium_v1",
        description="Tweedie model for expected total loss",
        data=policy_data,
        target_column="total_claim_cost",  # Many zeros, some positive
        distribution="tweedie",
        tweedie_power=1.5,
    )
    ```

    Args:
        model_name: Unique name for this model
        data: Training data as list of dictionaries
        target_column: Target variable column
        distribution: 'poisson', 'gamma', or 'tweedie'
        exposure_column: Optional exposure for rate modeling
        tweedie_power: Tweedie power parameter (1-2, default 1.5)
        alpha: L2 regularization strength (0 = none)
        ... (other parameters)

    Returns:
        Training results with coefficients, deviance scores, and model info.
    """
    try:
        result = train_glm(
            GLMTrainingInput(
                model_name=model_name,
                description=description,
                data=data,
                target_column=target_column,
                distribution=distribution,
                feature_columns=feature_columns,
                categorical_columns=categorical_columns,
                exposure_column=exposure_column,
                tweedie_power=tweedie_power,
                alpha=alpha,
                solver=solver,
                max_iter=max_iter,
                fit_intercept=fit_intercept,
                test_size=test_size,
                random_state=random_state,
            )
        )

        # Format distribution name
        dist_names = {
            "poisson": "POISSON (Claim Frequency)",
            "gamma": "GAMMA (Claim Severity)",
            "tweedie": f"TWEEDIE (Pure Premium, power={tweedie_power})",
        }
        dist_label = dist_names.get(result.distribution, result.distribution.upper())

        output_lines = [
            "=" * 60,
            f"GLM {dist_label} TRAINING COMPLETE",
            "=" * 60,
            "",
            "📊 DATASET SUMMARY",
            f"  Samples: {result.n_samples}",
            f"  Features: {result.n_features}",
            "",
            "📈 TARGET STATISTICS",
            f"  Mean: {result.target_stats['mean']:.4f}",
            f"  Std: {result.target_stats['std']:.4f}",
            f"  Range: [{result.target_stats['min']:.2f}, {result.target_stats['max']:.2f}]",
            f"  Zeros: {result.target_stats['zeros_pct']:.1f}%",
            "",
            "📈 PERFORMANCE METRICS",
            f"  Train D² (deviance explained): {result.train_deviance:.4f}",
            f"  Test D² (deviance explained):  {result.test_deviance:.4f}",
            f"  Train MAE: {result.train_mae:.4f}",
            f"  Test MAE:  {result.test_mae:.4f}",
            f"  Train RMSE: {result.train_rmse:.4f}",
            f"  Test RMSE:  {result.test_rmse:.4f}",
            "",
            "⚖️ MODEL COEFFICIENTS (log scale, sorted by magnitude):",
            f"  Intercept: {result.intercept:.4f} → base rate = exp({result.intercept:.4f}) = {np.exp(result.intercept):.4f}",
            "",
        ]

        # Sort coefficients by absolute magnitude
        sorted_coefs = sorted(
            result.coefficients.items(), key=lambda x: abs(x[1]), reverse=True
        )

        for feat, coef in sorted_coefs[:10]:
            multiplier = np.exp(coef)
            if coef >= 0:
                effect = f"+{(multiplier - 1) * 100:.1f}%"
            else:
                effect = f"{(multiplier - 1) * 100:.1f}%"
            bar = "█" * int(abs(coef) * 20)
            sign = "+" if coef >= 0 else ""
            output_lines.append(f"  {feat}: {sign}{coef:.4f} → {effect} {bar}")

        if len(sorted_coefs) > 10:
            output_lines.append(f"  ... and {len(sorted_coefs) - 10} more features")

        output_lines.extend([
            "",
            f"🔄 Convergence: {result.n_iterations} iterations",
            "",
            f"💾 MODEL REGISTERED: {result.model_name}",
            f"   Path: {result.saved_path}",
            "",
            "💡 INTERPRETATION:",
        ])

        if result.distribution == "poisson":
            output_lines.append("   Coefficients show effect on expected claim COUNT")
        elif result.distribution == "gamma":
            output_lines.append("   Coefficients show effect on expected claim AMOUNT")
        else:
            output_lines.append("   Coefficients show effect on expected TOTAL COST")

        output_lines.extend([
            "   Positive coef → higher expected value",
            "   Negative coef → lower expected value",
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
            f"- Target values match distribution:\n"
            f"  • Poisson: non-negative integers (0, 1, 2, ...)\n"
            f"  • Gamma: strictly positive (> 0)\n"
            f"  • Tweedie: non-negative (>= 0)"
        )

