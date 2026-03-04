"""Survival Analysis training skill."""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_TRAINING_DIR = _SKILL_DIR.parent.parent
_DATA_TOOLS_DIR = _TRAINING_DIR.parent.parent / "data-tools"
for _p in [str(_SKILL_DIR), str(_TRAINING_DIR), str(_DATA_TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from survival_analysis import SurvivalTrainingInput, train_survival_model
from utils import get_registered_dataset


def run(params: dict) -> str:
    """Train a survival model. See SKILL.md for parameters."""
    dataset_ref = params.pop("train_dataset_ref")
    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry.")

    input_fields = {
        "data": df.to_dict(orient="records"),
    }
    allowed = {
        "model_name", "description", "duration_column", "event_column",
        "feature_columns", "model_type", "penalizer", "l1_ratio",
        "test_size", "random_state",
    }
    for key in allowed:
        if key in params:
            input_fields[key] = params[key]

    try:
        result = train_survival_model(SurvivalTrainingInput(**input_fields))
    except Exception as e:
        return f"TRAINING FAILED\nError: {e}"

    model_labels = {
        "cox": "COX PROPORTIONAL HAZARDS",
        "weibull": "WEIBULL AFT",
        "lognormal": "LOG-NORMAL AFT",
    }
    model_label = model_labels.get(result.survival_model_type, result.survival_model_type.upper())

    lines = [
        "=" * 60,
        f"SURVIVAL ANALYSIS: {model_label}",
        "=" * 60,
        "",
        f"Total samples: {result.n_samples}",
        f"Events observed: {result.n_events} ({result.event_rate:.1%})",
        f"Censored: {result.n_censored} ({1 - result.event_rate:.1%})",
        f"Features: {result.n_features}",
        "",
        "DURATION STATISTICS",
        f"  Mean: {result.duration_stats['mean']:.2f}",
        f"  Median: {result.duration_stats['median']:.2f}",
        f"  Range: [{result.duration_stats['min']:.2f}, {result.duration_stats['max']:.2f}]",
        "",
        "MODEL PERFORMANCE",
        f"  Train C-index: {result.train_concordance:.4f}",
        f"  Test C-index:  {result.test_concordance:.4f}",
    ]

    if result.log_likelihood is not None:
        lines.append(f"  Log-likelihood: {result.log_likelihood:.2f}")
    if result.aic is not None:
        lines.append(f"  AIC: {result.aic:.2f}")
    if result.bic is not None:
        lines.append(f"  BIC: {result.bic:.2f}")

    c_idx = result.test_concordance
    if c_idx < 0.55:
        c_interp = "Poor (near random)"
    elif c_idx < 0.70:
        c_interp = "Fair"
    elif c_idx < 0.80:
        c_interp = "Good"
    else:
        c_interp = "Excellent"
    lines.append(f"  Interpretation: {c_interp}")

    lines.extend(["", "HAZARD RATIOS (sorted by effect size):"])
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
        else:
            effect = f"-{(1 - hr) * 100:.1f}% risk"
        lines.append(f"  {feat}: HR={hr:.3f} ({effect}) {sig}")

    if len(sorted_coefs) > 10:
        lines.append(f"  ... and {len(sorted_coefs) - 10} more features")

    lines.extend([
        "",
        "* = statistically significant (p < 0.05)",
        "",
        "INTERPRETATION:",
        "  HR > 1: Higher risk (faster event occurrence)",
        "  HR < 1: Lower risk (slower event occurrence)",
        "  HR = 1: No effect on timing",
        "",
        f"MODEL REGISTERED: {result.model_name}",
        f"Path: {result.saved_path}",
        "=" * 60,
    ])
    return "\n".join(lines)
