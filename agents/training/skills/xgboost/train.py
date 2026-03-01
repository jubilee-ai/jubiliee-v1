"""XGBoost training skill."""

import sys
from pathlib import Path

_SKILL_DIR = Path(__file__).parent
_TRAINING_DIR = _SKILL_DIR.parent.parent
_DATA_TOOLS_DIR = _TRAINING_DIR.parent.parent / "data-tools"
for _p in [str(_SKILL_DIR), str(_TRAINING_DIR), str(_DATA_TOOLS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from utils import get_registered_dataset
from xgboost_model import XGBoostTrainingInput, train_xgboost


def run(params: dict) -> str:
    """Train an XGBoost model. See SKILL.md for parameters."""
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
        "n_estimators", "max_depth", "learning_rate", "min_child_weight",
        "gamma", "reg_alpha", "reg_lambda", "subsample", "colsample_bytree",
        "scale_pos_weight", "monotone_constraints", "tree_method",
        "early_stopping_rounds", "random_state", "n_jobs",
        "test_size", "validation_size",
    }
    for key in allowed:
        if key in params:
            input_fields[key] = params[key]

    try:
        result = train_xgboost(XGBoostTrainingInput(**input_fields))
    except Exception as e:
        return f"TRAINING FAILED\nError: {e}"

    task_label = "CLASSIFICATION" if result.task_type == "classification" else "REGRESSION"
    score_label = "Accuracy" if result.task_type == "classification" else "R²"

    lines = [
        "=" * 60,
        f"XGBOOST {task_label} TRAINING COMPLETE",
        "=" * 60,
        "",
        f"Samples: {result.n_samples}",
        f"Features: {result.n_features}",
    ]

    if result.task_type == "classification":
        lines.append(f"Classes: {result.n_classes}")
        lines.append(f"Distribution: {result.class_distribution}")

    lines.extend([
        "",
        f"Train {score_label}: {result.train_score:.4f}",
        f"Val {score_label}:   {result.val_score:.4f}",
        f"Test {score_label}:  {result.test_score:.4f}",
    ])

    if result.task_type == "classification":
        roc = result.additional_metrics.get("roc_auc")
        if roc is not None:
            lines.append(f"Test ROC-AUC:  {roc:.4f}")
    else:
        lines.append(f"Test MAE:  {result.additional_metrics.get('mae', 0):.4f}")
        lines.append(f"Test RMSE: {result.additional_metrics.get('rmse', 0):.4f}")

    # Loss curve summary
    train_loss = result.training_history["train_loss"]
    val_loss = result.training_history["val_loss"]
    n_rounds = len(train_loss)
    show_rounds = sorted(set(
        min(r, n_rounds - 1)
        for r in [0, n_rounds // 4, n_rounds // 2, 3 * n_rounds // 4, n_rounds - 1]
    ))

    lines.extend(["", "LOSS CURVE:"])
    for r in show_rounds:
        marker = " <- best" if r == result.best_iteration else ""
        lines.append(
            f"  Round {r + 1:3d}: Train={train_loss[r]:.4f}, Val={val_loss[r]:.4f}{marker}"
        )
    lines.append(f"  Best iteration: {result.best_iteration + 1} / {n_rounds}")
    lines.append(f"  Trees used: {result.actual_n_estimators}")

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
        f"MODEL REGISTERED: {result.model_name}",
        f"Path: {result.saved_path}",
        "=" * 60,
    ])
    return "\n".join(lines)
