"""Random Forest training skill."""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_TRAINING_DIR = _SKILL_DIR.parent.parent
_DATA_TOOLS_DIR = _TRAINING_DIR.parent.parent / "data-tools"
for _p in [str(_SKILL_DIR), str(_TRAINING_DIR), str(_DATA_TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from random_forest import RandomForestTrainingInput, train_random_forest
from utils import get_registered_dataset


def run(params: dict) -> str:
    """Train a random forest model. See SKILL.md for parameters."""
    dataset_ref = params.pop("train_dataset_ref")
    df = get_registered_dataset(dataset_ref)
    if df is None:
        raise ValueError(f"Dataset '{dataset_ref}' not found in registry.")

    input_fields = {
        "data": df.to_dict(orient="records"),
    }
    allowed = {
        "model_name", "description", "target_column", "task_type",
        "feature_columns", "categorical_columns", "sample_weight_column",
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "bootstrap", "class_weight", "random_state",
        "n_jobs", "test_size",
    }
    for key in allowed:
        if key in params:
            input_fields[key] = params[key]

    try:
        result = train_random_forest(RandomForestTrainingInput(**input_fields))
    except Exception as e:
        return f"TRAINING FAILED\nError: {e}"

    task_label = "CLASSIFICATION" if result.task_type == "classification" else "REGRESSION"
    score_label = "Accuracy" if result.task_type == "classification" else "R²"

    lines = [
        "=" * 60,
        f"RANDOM FOREST {task_label} TRAINING COMPLETE",
        "=" * 60,
        "",
        f"Samples: {result.n_samples}",
        f"Features: {result.n_features}",
    ]

    if result.task_type == "classification":
        lines.append(f"Classes: {result.n_classes}")
        lines.append(f"Class Distribution: {result.class_distribution}")

    lines.extend([
        "",
        f"Train {score_label}: {result.train_score:.4f}",
        f"Test {score_label}:  {result.test_score:.4f}",
    ])

    if result.task_type == "classification":
        roc = result.additional_metrics.get("roc_auc")
        lines.append(f"Test ROC-AUC:   {roc:.4f}" if roc is not None else "Test ROC-AUC:   N/A")
    else:
        lines.append(f"Test MAE:  {result.additional_metrics.get('mae', 0):.4f}")
        lines.append(f"Test RMSE: {result.additional_metrics.get('rmse', 0):.4f}")

    if result.oob_score is not None:
        lines.append(f"OOB Score: {result.oob_score:.4f}")

    if result.classification_report:
        lines.extend(["", "CLASSIFICATION REPORT", result.classification_report])
    if result.confusion_matrix:
        lines.extend(["", "CONFUSION MATRIX", str(np.array(result.confusion_matrix))])

    lines.extend(["", "FEATURE IMPORTANCES (top 10):"])
    sorted_imp = sorted(
        result.feature_importances.items(), key=lambda x: x[1], reverse=True
    )
    for feat, imp in sorted_imp[:10]:
        lines.append(f"  {feat}: {imp:.4f}")

    lines.extend([
        "",
        f"Trees: {result.n_estimators}",
        f"MODEL REGISTERED: {result.model_name}",
        f"Path: {result.saved_path}",
        "=" * 60,
    ])
    return "\n".join(lines)
