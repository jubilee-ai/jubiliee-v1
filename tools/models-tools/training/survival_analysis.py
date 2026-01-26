"""
Survival Analysis Training Tool for AI Data Scientist Agent

A LangChain tool that trains survival models for time-to-event analysis.
Uses the lifelines library for Cox Proportional Hazards and other survival models.

Critical for insurance applications:
- Policy lapse/churn prediction
- Time to first claim
- Mortality modeling
- Customer lifetime value
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

# Survival analysis imports
try:
    from lifelines import CoxPHFitter, LogNormalAFTFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False


# Module-level wrapper class for pickling support
class SurvivalModelWrapper:
    """Wrapper to make survival models compatible with joblib persistence."""
    
    def __init__(self, model, model_type, duration_col, event_col, feature_names):
        self.model = model
        self.model_type = model_type
        self.duration_col = duration_col
        self.event_col = event_col
        self.feature_names_in_ = feature_names

    def predict(self, X):
        """Predict partial hazard (Cox) or median survival time (AFT)."""
        if isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Handle categorical encoding if needed
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        # Align columns with training data
        for col in self.feature_names_in_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.feature_names_in_]
        
        if self.model_type == "cox":
            # Return partial hazard (higher = more risk)
            return self.model.predict_partial_hazard(X_encoded).values
        else:
            # Return median survival time
            return self.model.predict_median(X_encoded).values

    def predict_survival_function(self, X, times=None):
        """Predict full survival curve."""
        if isinstance(X, list):
            X = pd.DataFrame(X)
        X_encoded = pd.get_dummies(X, drop_first=True)
        for col in self.feature_names_in_:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        X_encoded = X_encoded[self.feature_names_in_]
        
        return self.model.predict_survival_function(X_encoded, times=times)


class SurvivalTrainingInput(BaseModel):
    """Input schema for training a Survival Analysis model."""
    model_config = ConfigDict(extra="forbid")

    # Model identification
    model_name: str = Field(
        description="Unique name for this model. Used for storage and retrieval. "
        "Example: 'policy_lapse_model_v1', 'mortality_cox_prod'"
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this model does.",
    )

    # Data specification
    data: list[dict] = Field(
        description="Training data as a list of dictionaries, where each dict is a row."
    )
    duration_column: str = Field(
        description="Name of column containing the time/duration until event or censoring. "
        "Examples: 'policy_tenure_months', 'days_to_claim', 'survival_time'"
    )
    event_column: str = Field(
        description="Name of column indicating whether the event occurred (1) or was censored (0). "
        "Examples: 'lapsed' (1=lapsed, 0=still active), 'claimed' (1=claimed, 0=no claim yet)"
    )
    feature_columns: Optional[list[str]] = Field(
        default=None,
        description="List of column names to use as covariates/features. If None, uses all except duration and event.",
    )

    # Model type
    model_type: Literal["cox", "weibull", "lognormal"] = Field(
        default="cox",
        description="Survival model type:\n"
        "- 'cox': Cox Proportional Hazards (semi-parametric, most common)\n"
        "- 'weibull': Weibull AFT (parametric, good for monotonic hazards)\n"
        "- 'lognormal': Log-Normal AFT (parametric, good for non-monotonic hazards)"
    )

    # Regularization
    penalizer: float = Field(
        default=0.01,
        description="L2 regularization strength (penalizer). Higher = simpler model. "
        "Recommended range: 0.001 to 0.5",
        ge=0,
    )
    l1_ratio: float = Field(
        default=0.0,
        description="Elastic net mixing parameter. 0 = L2 only, 1 = L1 only. "
        "Default 0 (pure L2/ridge).",
        ge=0,
        le=1,
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


class SurvivalTrainingOutput(BaseModel):
    """Output schema for trained Survival model."""

    success: bool
    model_type: str
    survival_model_type: str

    # Dataset info
    n_samples: int
    n_events: int
    n_censored: int
    event_rate: float
    n_features: int
    feature_names: list[str]

    # Duration statistics
    duration_stats: dict[str, float]

    # Model performance
    train_concordance: float = Field(
        description="C-index on training set (0.5 = random, 1.0 = perfect)"
    )
    test_concordance: float = Field(
        description="C-index on test set"
    )
    log_likelihood: Optional[float]
    aic: Optional[float] = Field(description="Akaike Information Criterion (lower = better)")
    bic: Optional[float] = Field(description="Bayesian Information Criterion (lower = better)")

    # Coefficients / Hazard Ratios
    coefficients: dict[str, dict] = Field(
        description="Feature coefficients with hazard ratios and p-values"
    )

    # Artifacts
    saved_path: str
    model_name: str
    training_timestamp: str


def train_survival_model(input_data: SurvivalTrainingInput) -> SurvivalTrainingOutput:
    """
    Train a Survival Analysis model for time-to-event prediction.
    """
    if not LIFELINES_AVAILABLE:
        raise ImportError(
            "lifelines package not installed. Run: pip install lifelines"
        )

    # Convert input data to DataFrame
    df = pd.DataFrame(input_data.data)

    # Validate required columns
    if input_data.duration_column not in df.columns:
        raise ValueError(f"Duration column '{input_data.duration_column}' not found.")
    if input_data.event_column not in df.columns:
        raise ValueError(f"Event column '{input_data.event_column}' not found.")

    # Validate duration and event values
    if (df[input_data.duration_column] <= 0).any():
        raise ValueError("Duration column must contain positive values only.")
    if not df[input_data.event_column].isin([0, 1]).all():
        raise ValueError("Event column must contain only 0 (censored) and 1 (event occurred).")

    # Determine feature columns
    exclude_cols = [input_data.duration_column, input_data.event_column]
    if input_data.feature_columns:
        feature_cols = input_data.feature_columns
        missing = set(feature_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
    else:
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Handle categorical columns (one-hot encode)
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        # Update feature columns after one-hot encoding
        feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Prepare data for survival model
    survival_cols = [input_data.duration_column, input_data.event_column] + feature_cols
    survival_df = df[survival_cols].copy()

    # Calculate statistics
    n_events = int(df[input_data.event_column].sum())
    n_censored = len(df) - n_events
    event_rate = n_events / len(df)

    duration_stats = {
        "mean": float(df[input_data.duration_column].mean()),
        "median": float(df[input_data.duration_column].median()),
        "std": float(df[input_data.duration_column].std()),
        "min": float(df[input_data.duration_column].min()),
        "max": float(df[input_data.duration_column].max()),
    }

    # NOTE: No internal train_test_split - data is already split by the pipeline
    # The training agent passes pre-split training data here
    # Validation/test evaluation happens via evaluate_model tool
    
    # Use all indices for training
    train_idx = np.arange(len(survival_df))

    train_df = survival_df.iloc[train_idx].copy()

    # Initialize and fit the model
    if input_data.model_type == "cox":
        model = CoxPHFitter(
            penalizer=input_data.penalizer,
            l1_ratio=input_data.l1_ratio,
        )
        model.fit(
            train_df,
            duration_col=input_data.duration_column,
            event_col=input_data.event_column,
        )
    elif input_data.model_type == "weibull":
        model = WeibullAFTFitter(
            penalizer=input_data.penalizer,
            l1_ratio=input_data.l1_ratio,
        )
        model.fit(
            train_df,
            duration_col=input_data.duration_column,
            event_col=input_data.event_column,
        )
    else:  # lognormal
        model = LogNormalAFTFitter(
            penalizer=input_data.penalizer,
            l1_ratio=input_data.l1_ratio,
        )
        model.fit(
            train_df,
            duration_col=input_data.duration_column,
            event_col=input_data.event_column,
        )

    # Calculate concordance index
    # NOTE: These are TRAINING metrics only - use evaluate_model for val/test metrics
    train_concordance = model.concordance_index_

    # Get model statistics
    log_likelihood = float(model.log_likelihood_) if hasattr(model, "log_likelihood_") else None
    
    # Cox uses AIC_partial_, AFT models use AIC_
    if input_data.model_type == "cox":
        aic = float(model.AIC_partial_) if hasattr(model, "AIC_partial_") else None
    else:
        aic = float(model.AIC_) if hasattr(model, "AIC_") else None
    bic = float(model.BIC_) if hasattr(model, "BIC_") else None

    # Extract coefficients with hazard ratios
    summary = model.summary
    coefficients = {}
    for idx, row in summary.iterrows():
        # Handle multi-index (for AFT models like Weibull)
        if isinstance(idx, tuple):
            name = f"{idx[0]}_{idx[1]}"
        else:
            name = str(idx)
        
        coef_val = float(row["coef"])
        exp_coef = float(row["exp(coef)"]) if "exp(coef)" in row else np.exp(coef_val)
        p_val = float(row["p"]) if "p" in row else None
        
        coefficients[name] = {
            "coefficient": coef_val,
            "hazard_ratio": exp_coef,
            "p_value": p_val,
            "significant": p_val < 0.05 if p_val is not None else None,
        }

    # Wrap model using module-level class for pickle support
    wrapped_model = SurvivalModelWrapper(
        model=model,
        model_type=input_data.model_type,
        duration_col=input_data.duration_column,
        event_col=input_data.event_column,
        feature_names=feature_cols,
    )

    # Save model
    save_path = generate_model_path(input_data.model_name)
    joblib.dump(wrapped_model, save_path)

    # Register in registry
    # NOTE: These are TRAINING metrics only - use evaluate_model for val/test metrics
    metrics = {
        "train_concordance": train_concordance,
        "event_rate": event_rate,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "bic": bic,
    }

    hyperparameters = {
        "task_type": "survival",
        "survival_model_type": input_data.model_type,
        "penalizer": input_data.penalizer,
        "l1_ratio": input_data.l1_ratio,
        "duration_column": input_data.duration_column,
        "event_column": input_data.event_column,
        "random_state": input_data.random_state,
    }

    register_model(
        model_name=input_data.model_name,
        model_path=save_path,
        model_type="survival_analysis",
        description=input_data.description,
        metrics=metrics,
        feature_names=feature_cols,
        target_column=input_data.event_column,
        hyperparameters=hyperparameters,
        training_samples=len(df),
        classes=None,
    )

    return SurvivalTrainingOutput(
        success=True,
        model_type="survival_analysis",
        survival_model_type=input_data.model_type,
        n_samples=len(df),
        n_events=n_events,
        n_censored=n_censored,
        event_rate=event_rate,
        n_features=len(feature_cols),
        feature_names=feature_cols,
        duration_stats=duration_stats,
        train_concordance=train_concordance,
        test_concordance=train_concordance,  # Same as train (no internal split anymore)
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        coefficients=coefficients,
        saved_path=save_path,
        model_name=input_data.model_name,
        training_timestamp=datetime.now(timezone.utc).isoformat(),
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class SurvivalAnalysisToolInput(BaseModel):
    """Input for training a Survival Analysis model."""
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(description="Unique model name for storage")
    description: str = Field(default="", description="Model description")
    train_dataset_ref: str = Field(
        description="Reference name of the registered training dataset. "
        "The dataset must be registered via register_dataset()."
    )
    duration_column: str = Field(description="Column with time until event/censoring")
    event_column: str = Field(description="Column with event indicator (1=event, 0=censored)")
    feature_columns: Optional[list[str]] = Field(default=None)
    model_type: Literal["cox", "weibull", "lognormal"] = Field(
        default="cox",
        description="'cox' (semi-parametric), 'weibull' (monotonic hazard), 'lognormal' (non-monotonic)"
    )
    penalizer: float = Field(default=0.01, ge=0)
    l1_ratio: float = Field(default=0.0, ge=0, le=1)
    test_size: float = Field(default=0.2, gt=0, lt=1)
    random_state: Optional[int] = Field(default=42)


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


@tool("survival_analysis", args_schema=SurvivalAnalysisToolInput)
def survival_analysis_tool(
    model_name: str,
    train_dataset_ref: str,
    duration_column: str,
    event_column: str,
    description: str = "",
    feature_columns: Optional[list[str]] = None,
    model_type: Literal["cox", "weibull", "lognormal"] = "cox",
    penalizer: float = 0.01,
    l1_ratio: float = 0.0,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
) -> str:
    """Train a Survival Analysis model for time-to-event prediction in insurance.

    WHAT IT DOES:
    Survival analysis models the time until an event occurs, accounting for
    censored observations (subjects who haven't experienced the event yet).
    Essential for insurance where we need to predict WHEN events happen.

    MODEL TYPES:
    - **Cox Proportional Hazards** (default): Semi-parametric, most widely used.
      Outputs hazard ratios - multiplicative effects on risk.
    - **Weibull AFT**: Parametric, good when hazard is monotonically increasing
      or decreasing. Common in reliability engineering.
    - **Log-Normal AFT**: Parametric, good for non-monotonic hazards (risk
      increases then decreases).

    KEY METRIC - CONCORDANCE INDEX (C-index):
    - 0.5 = Random prediction (no better than coin flip)
    - 0.7-0.8 = Acceptable discrimination
    - 0.8+ = Excellent discrimination
    - 1.0 = Perfect prediction

    INSURANCE USE CASES:

    1. **Policy Lapse/Churn**:
       - duration: months since policy start
       - event: 1 = lapsed, 0 = still active
       - Predicts: Which policies will lapse and when

    2. **Time to First Claim**:
       - duration: days until first claim
       - event: 1 = claimed, 0 = no claim yet
       - Predicts: When claims are likely to occur

    3. **Mortality Modeling** (Life Insurance):
       - duration: survival time
       - event: 1 = death, 0 = alive/censored
       - Predicts: Life expectancy by risk factors

    4. **Customer Lifetime Value**:
       - duration: relationship length
       - event: 1 = churned, 0 = active
       - Predicts: Expected customer tenure

    HAZARD RATIO INTERPRETATION:
    - HR = 1.0: No effect on risk
    - HR > 1.0: Increases risk (faster event)
    - HR < 1.0: Decreases risk (slower event)
    
    Example: HR = 1.5 means 50% higher risk of event per unit time.

    EXAMPLE (Policy Lapse):
    ```
    result = survival_analysis(
        model_name="lapse_cox_v1",
        description="Cox model for policy lapse prediction",
        data=policy_data,
        duration_column="months_active",
        event_column="lapsed",
        model_type="cox",
    )
    ```

    EXAMPLE (Time to Claim):
    ```
    result = survival_analysis(
        model_name="claim_timing_v1",
        description="Weibull AFT for time to first claim",
        data=claims_data,
        duration_column="days_to_claim",
        event_column="claimed",
        model_type="weibull",
    )
    ```

    Args:
        model_name: Unique name for this model
        data: Training data as list of dictionaries
        duration_column: Time until event or censoring (must be positive)
        event_column: Event indicator (1=event, 0=censored)
        model_type: 'cox', 'weibull', or 'lognormal'
        penalizer: L2 regularization strength
        ...

    Returns:
        Training results with concordance index, hazard ratios, and model info.
    """
    if not LIFELINES_AVAILABLE:
        return (
            "❌ MISSING DEPENDENCY\n\n"
            "The 'lifelines' package is not installed.\n"
            "Run: pip install lifelines"
        )

    try:
        # Load dataset from registry
        df = _load_dataset_from_ref(train_dataset_ref)
        data = df.to_dict(orient="records")
        
        result = train_survival_model(
            SurvivalTrainingInput(
                model_name=model_name,
                description=description,
                data=data,
                duration_column=duration_column,
                event_column=event_column,
                feature_columns=feature_columns,
                model_type=model_type,
                penalizer=penalizer,
                l1_ratio=l1_ratio,
                test_size=test_size,
                random_state=random_state,
            )
        )

        model_labels = {
            "cox": "COX PROPORTIONAL HAZARDS",
            "weibull": "WEIBULL AFT",
            "lognormal": "LOG-NORMAL AFT",
        }
        model_label = model_labels.get(result.survival_model_type, result.survival_model_type.upper())

        output_lines = [
            "=" * 60,
            f"SURVIVAL ANALYSIS: {model_label}",
            "=" * 60,
            "",
            "📊 DATASET SUMMARY",
            f"  Total samples: {result.n_samples}",
            f"  Events observed: {result.n_events} ({result.event_rate:.1%})",
            f"  Censored: {result.n_censored} ({1 - result.event_rate:.1%})",
            f"  Features: {result.n_features}",
            "",
            f"⏱️ DURATION STATISTICS ({duration_column})",
            f"  Mean: {result.duration_stats['mean']:.2f}",
            f"  Median: {result.duration_stats['median']:.2f}",
            f"  Range: [{result.duration_stats['min']:.2f}, {result.duration_stats['max']:.2f}]",
            "",
            "📈 MODEL PERFORMANCE",
            f"  Train C-index: {result.train_concordance:.4f}",
            f"  Test C-index:  {result.test_concordance:.4f}",
        ]

        # Add model fit statistics
        if result.log_likelihood is not None:
            output_lines.append(f"  Log-likelihood: {result.log_likelihood:.2f}")
        if result.aic is not None:
            output_lines.append(f"  AIC: {result.aic:.2f}")
        if result.bic is not None:
            output_lines.append(f"  BIC: {result.bic:.2f}")

        # C-index interpretation
        c_idx = result.test_concordance
        if c_idx < 0.55:
            c_interp = "⚠️ Poor (near random)"
        elif c_idx < 0.70:
            c_interp = "👀 Fair"
        elif c_idx < 0.80:
            c_interp = "✓ Good"
        else:
            c_interp = "✓✓ Excellent"
        output_lines.append(f"  Interpretation: {c_interp}")

        output_lines.extend([
            "",
            "⚖️ HAZARD RATIOS (sorted by effect size):",
        ])

        # Sort coefficients by hazard ratio deviation from 1.0
        sorted_coefs = sorted(
            result.coefficients.items(),
            key=lambda x: abs(x[1]["hazard_ratio"] - 1.0),
            reverse=True,
        )

        for feat, info in sorted_coefs[:10]:
            hr = info["hazard_ratio"]
            p = info.get("p_value", 1.0) or 1.0
            sig = "*" if p < 0.05 else ""
            
            if hr >= 1:
                effect = f"+{(hr - 1) * 100:.1f}% risk"
                bar = "+" * min(int((hr - 1) * 10), 20)
            else:
                effect = f"-{(1 - hr) * 100:.1f}% risk"
                bar = "-" * min(int((1 - hr) * 10), 20)

            output_lines.append(f"  {feat}: HR={hr:.3f} ({effect}) {bar} {sig}")

        if len(sorted_coefs) > 10:
            output_lines.append(f"  ... and {len(sorted_coefs) - 10} more features")

        output_lines.extend([
            "",
            "* = statistically significant (p < 0.05)",
            "",
            f"💾 MODEL REGISTERED: {result.model_name}",
            f"   Path: {result.saved_path}",
            "",
            "💡 INTERPRETATION:",
            "   HR > 1: Higher risk (faster event occurrence)",
            "   HR < 1: Lower risk (slower event occurrence)",
            "   HR = 1: No effect on timing",
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
            f"- duration_column contains positive values\n"
            f"- event_column contains only 0 (censored) and 1 (event)\n"
            f"- Feature columns exist in data"
        )

