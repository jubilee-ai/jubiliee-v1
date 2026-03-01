"""Logistic Regression training skill."""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_TRAINING_DIR = _SKILL_DIR.parent.parent
_DATA_TOOLS_DIR = _TRAINING_DIR.parent.parent / "data-tools"
for _p in [str(_SKILL_DIR), str(_TRAINING_DIR), str(_DATA_TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from logistic_regression import (
    LogisticRegressionTrainingInput,
    train_logistic_regression,
)
from utils import get_registered_dataset


def run(params: dict) -> str:
    """Train a logistic regression model. See SKILL.md for parameters."""
    dataset_ref = params.pop("train_dataset_ref")
    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry.")

    input_fields = {
        "data": df.to_dict(orient="records"),
    }
    allowed = {
        "model_name", "description", "target_column", "feature_columns",
        "categorical_columns", "sample_weight_column", "C", "l1_ratio",
        "solver", "max_iter", "class_weight", "fit_intercept",
        "random_state", "test_size",
    }
    for key in allowed:
        if key in params:
            input_fields[key] = params[key]

    try:
        result = train_logistic_regression(
            LogisticRegressionTrainingInput(**input_fields)
        )
    except Exception as e:
        return f"TRAINING FAILED\nError: {e}"

    lines = [
        "=" * 60,
        "LOGISTIC REGRESSION TRAINING COMPLETE",
        "=" * 60,
        "",
        f"Samples: {result.n_samples}",
        f"Features: {result.n_features}",
        f"Classes: {result.n_classes}",
        f"Class Distribution: {result.class_distribution}",
        "",
        f"Train Accuracy: {result.train_accuracy:.4f}",
        f"Test Accuracy:  {result.test_accuracy:.4f}",
    ]
    if result.test_roc_auc is not None:
        lines.append(f"Test ROC-AUC:   {result.test_roc_auc:.4f}")
    else:
        lines.append("Test ROC-AUC:   N/A")

    lines.extend(["", "CLASSIFICATION REPORT", result.classification_report])
    lines.extend(["", f"CONFUSION MATRIX", str(np.array(result.confusion_matrix))])

    lines.extend(["", "TOP COEFFICIENTS (by magnitude):"])
    for class_name, coefs in result.coefficients.items():
        coef_pairs = sorted(
            zip(result.feature_names, coefs), key=lambda x: abs(x[1]), reverse=True
        )
        lines.append(f"  Class '{class_name}':")
        for feat, coef in coef_pairs[:5]:
            lines.append(f"    {feat}: {'+' if coef > 0 else ''}{coef:.4f}")

    lines.extend([
        "",
        f"Convergence: {result.n_iterations} iterations",
        f"MODEL REGISTERED: {result.model_name}",
        f"Path: {result.saved_path}",
        "=" * 60,
    ])
    return "\n".join(lines)
