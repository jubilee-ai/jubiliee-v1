"""GLM training skill."""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_TRAINING_DIR = _SKILL_DIR.parent.parent
_DATA_TOOLS_DIR = _TRAINING_DIR.parent.parent / "data-tools"
for _p in [str(_SKILL_DIR), str(_TRAINING_DIR), str(_DATA_TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from glm import GLMTrainingInput, train_glm
from utils import get_registered_dataset


def run(params: dict) -> str:
    """Train a GLM model. See SKILL.md for parameters."""
    dataset_ref = params.pop("train_dataset_ref")
    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry.")

    input_fields = {
        "data": df.to_dict(orient="records"),
    }
    allowed = {
        "model_name", "description", "target_column", "distribution",
        "feature_columns", "categorical_columns", "exposure_column",
        "sample_weight_column", "tweedie_power", "alpha", "solver",
        "max_iter", "fit_intercept", "test_size", "random_state",
    }
    for key in allowed:
        if key in params:
            input_fields[key] = params[key]

    try:
        result = train_glm(GLMTrainingInput(**input_fields))
    except Exception as e:
        return f"TRAINING FAILED\nError: {e}"

    dist_names = {
        "poisson": "POISSON (Claim Frequency)",
        "gamma": "GAMMA (Claim Severity)",
        "tweedie": f"TWEEDIE (Pure Premium, power={result.distribution})",
    }
    dist_label = dist_names.get(result.distribution, result.distribution.upper())

    lines = [
        "=" * 60,
        f"GLM {dist_label} TRAINING COMPLETE",
        "=" * 60,
        "",
        f"Samples: {result.n_samples}",
        f"Features: {result.n_features}",
        "",
        "TARGET STATISTICS",
        f"  Mean: {result.target_stats['mean']:.4f}",
        f"  Std: {result.target_stats['std']:.4f}",
        f"  Range: [{result.target_stats['min']:.2f}, {result.target_stats['max']:.2f}]",
        f"  Zeros: {result.target_stats['zeros_pct']:.1f}%",
        "",
        f"Train D² (deviance explained): {result.train_deviance:.4f}",
        f"Test D² (deviance explained):  {result.test_deviance:.4f}",
        f"Train MAE: {result.train_mae:.4f}",
        f"Test MAE:  {result.test_mae:.4f}",
        f"Train RMSE: {result.train_rmse:.4f}",
        f"Test RMSE:  {result.test_rmse:.4f}",
        "",
        f"Intercept: {result.intercept:.4f} -> base rate = exp({result.intercept:.4f}) = {np.exp(result.intercept):.4f}",
        "",
        "COEFFICIENTS (top 10, sorted by magnitude, log scale):",
    ]

    sorted_coefs = sorted(
        result.coefficients.items(), key=lambda x: abs(x[1]), reverse=True
    )
    for feat, coef in sorted_coefs[:10]:
        multiplier = np.exp(coef)
        effect = f"+{(multiplier - 1) * 100:.1f}%" if coef >= 0 else f"{(multiplier - 1) * 100:.1f}%"
        lines.append(f"  {feat}: {'+' if coef >= 0 else ''}{coef:.4f} -> {effect}")

    if len(sorted_coefs) > 10:
        lines.append(f"  ... and {len(sorted_coefs) - 10} more features")

    lines.extend([
        "",
        f"Convergence: {result.n_iterations} iterations",
        f"MODEL REGISTERED: {result.model_name}",
        f"Path: {result.saved_path}",
        "=" * 60,
    ])
    return "\n".join(lines)
