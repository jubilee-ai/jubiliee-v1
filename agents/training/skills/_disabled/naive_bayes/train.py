"""Naive Bayes training skill."""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_TRAINING_DIR = _SKILL_DIR.parent.parent
_DATA_TOOLS_DIR = _TRAINING_DIR.parent.parent / "data-tools"
for _p in [str(_SKILL_DIR), str(_TRAINING_DIR), str(_DATA_TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from naive_bayes import NaiveBayesTrainingInput, train_naive_bayes
from utils import get_registered_dataset


def run(params: dict) -> str:
    """Train a naive bayes model. See SKILL.md for parameters."""
    dataset_ref = params.pop("train_dataset_ref")
    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry.")

    input_fields = {
        "data": df.to_dict(orient="records"),
    }
    allowed = {
        "model_name", "description", "target_column", "feature_columns",
        "categorical_columns", "sample_weight_column", "variant",
        "var_smoothing", "alpha", "fit_prior", "class_prior", "test_size",
    }
    for key in allowed:
        if key in params:
            input_fields[key] = params[key]

    try:
        result = train_naive_bayes(NaiveBayesTrainingInput(**input_fields))
    except Exception as e:
        return f"TRAINING FAILED\nError: {e}"

    variant_label = {
        "gaussian": "Gaussian",
        "multinomial": "Multinomial",
        "complement": "Complement",
    }.get(result.variant, result.variant)

    lines = [
        "=" * 60,
        f"NAIVE BAYES ({variant_label}) TRAINING COMPLETE",
        "=" * 60,
        "",
    ]

    if result.discretized_target:
        lines.extend([
            "WARNING: TARGET AUTO-DISCRETIZED",
            "  Original target was continuous. Binarized at median.",
            "",
        ])

    lines.extend([
        f"Samples: {result.n_samples}",
        f"Features: {result.n_features}",
        f"Classes: {result.n_classes}",
        f"Class Distribution: {result.class_distribution}",
        "",
        f"Train Accuracy: {result.train_accuracy:.4f}",
        f"Test Accuracy:  {result.test_accuracy:.4f}",
    ])
    if result.test_roc_auc is not None:
        lines.append(f"Test ROC-AUC:   {result.test_roc_auc:.4f}")
    else:
        lines.append("Test ROC-AUC:   N/A")

    lines.extend(["", "CLASSIFICATION REPORT", result.classification_report])
    lines.extend(["", "CONFUSION MATRIX", str(np.array(result.confusion_matrix))])

    lines.extend(["", "CLASS LOG PRIORS:"])
    for i, log_prior in enumerate(result.class_log_prior):
        prior = np.exp(log_prior)
        lines.append(f"  Class {i}: log_prior={log_prior:.4f} (prior={prior:.4f})")

    lines.extend([
        "",
        f"MODEL REGISTERED: {result.model_name}",
        f"Path: {result.saved_path}",
        f"Variant: {variant_label}",
        "=" * 60,
    ])
    return "\n".join(lines)
