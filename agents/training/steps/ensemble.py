"""
Post-training optimization: ensemble blending and full-data retraining.
"""

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

_ROOT = Path(__file__).parents[3]
for _p in [
    str(_ROOT / "tools" / "models-tools" / "training"),
    str(_ROOT / "tools" / "data-tools"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_storage import (
    delete_model, generate_model_path, get_model_info, load_model, register_model,
)
from utils import get_registered_dataset


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class EnsembleModel:
    """Weighted blend of multiple sklearn-compatible models."""

    def __init__(self, models: list, weights: list[float], model_names: list[str]):
        self.models = models
        self.weights = np.array(weights, dtype=np.float64)
        self.model_names = model_names

        if hasattr(models[0], "classes_"):
            self.classes_ = models[0].classes_

    def predict_proba(self, X):
        return sum(w * m.predict_proba(X) for w, m in zip(self.weights, self.models))

    def predict(self, X):
        if hasattr(self, "classes_"):
            probas = self.predict_proba(X)
            return (probas[:, 1] >= 0.5).astype(int) if probas.shape[1] == 2 else probas.argmax(axis=1)
        return sum(w * m.predict(X) for w, m in zip(self.weights, self.models))

    def __repr__(self):
        parts = [f"{n}({w:.2f})" for n, w in zip(self.model_names, self.weights)]
        return f"EnsembleModel([{', '.join(parts)}])"


# =============================================================================
# HELPERS
# =============================================================================

def _score_metric(task_type: str):
    """Return (scoring_name, score_function) for the task type."""
    if task_type == "regression":
        return "r2", r2_score
    return "roc_auc", roc_auc_score


def _iteration_score(it: dict, task_type: str) -> float:
    if task_type == "regression":
        return it.get("val_r2") or it.get("train_r2") or float("-inf")
    return it.get("val_roc_auc") or it.get("val_accuracy") or float("-inf")


def _get_predictions(model, X, task_type: str) -> np.ndarray:
    """Get predictions appropriate for the task type."""
    if task_type == "classification" and hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        return probas[:, 1] if probas.shape[1] == 2 else probas
    return model.predict(X)


def _estimator_family(it: dict) -> str:
    """Extract a family key from an iteration dict.

    Uses the model_type from the registry (e.g. 'sklearn_XGBClassifier' → 'xgbclassifier')
    which is always set and doesn't need a hardcoded lookup table.
    """
    raw = it.get("tool", it.get("model_type", ""))
    name = raw.rsplit(":", 1)[-1].rsplit("_", 1)[-1].lower()
    return name or raw.lower()


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _select_diverse(iterations: list[dict], task_type: str, max_models: int = 5) -> list[dict]:
    """Pick the best model from each estimator family, up to max_models."""
    successful = [it for it in iterations if it.get("success")]
    if not successful:
        return []

    family_best: dict[str, dict] = {}
    for it in successful:
        family = _estimator_family(it)
        if family not in family_best or _iteration_score(it, task_type) > _iteration_score(family_best[family], task_type):
            family_best[family] = it

    ranked = sorted(family_best.values(), key=lambda it: _iteration_score(it, task_type), reverse=True)

    if len(ranked) < 2 and len(successful) >= 2:
        seen = {it["model_name"] for it in ranked}
        for it in sorted(successful, key=lambda it: _iteration_score(it, task_type), reverse=True):
            if it["model_name"] not in seen:
                ranked.append(it)
                seen.add(it["model_name"])
                if len(ranked) >= max_models:
                    break

    return ranked[:max_models]


def _optimize_weights(predictions: list[np.ndarray], y_true: np.ndarray, task_type: str) -> np.ndarray:
    """Find blend weights that maximize the validation metric."""
    from scipy.optimize import minimize

    n = len(predictions)
    if n == 1:
        return np.array([1.0])

    _, score_fn = _score_metric(task_type)

    def neg_score(raw_w):
        w = _softmax(raw_w)
        blended = sum(wi * pi for wi, pi in zip(w, predictions))
        try:
            return -score_fn(y_true, blended)
        except ValueError:
            return 0.0

    try:
        result = minimize(neg_score, np.zeros(n), method="Nelder-Mead",
                          options={"maxiter": 500, "xatol": 1e-6})
        return _softmax(result.x)
    except Exception:
        return np.ones(n) / n


# =============================================================================
# BUILD ENSEMBLE
# =============================================================================

def build_ensemble(
    iterations: list[dict],
    val_ref: str,
    target_column: str,
    task_type: str,
    ensemble_name: str,
    max_models: int = 5,
) -> dict[str, Any]:
    """Build a weighted ensemble from the best diverse models."""
    candidates = _select_diverse(iterations, task_type, max_models)
    if len(candidates) < 2:
        return {"success": False, "reason": "fewer_than_2_models"}

    val_df = get_registered_dataset(val_ref)
    if val_df is None:
        return {"success": False, "reason": f"validation dataset '{val_ref}' not found"}

    y_true = val_df[target_column].values
    X_val = val_df[[c for c in val_df.columns if c != target_column]]

    models, names, preds = [], [], []
    for it in candidates:
        name = it["model_name"]
        try:
            model = load_model(name)
            preds.append(_get_predictions(model, X_val, task_type))
            models.append(model)
            names.append(name)
        except Exception as e:
            print(f"[ensemble] Skipping {name}: {e}")

    if len(models) < 2:
        return {"success": False, "reason": "fewer_than_2_models_loaded"}

    _, score_fn = _score_metric(task_type)
    metric_name = "val_roc_auc" if task_type == "classification" else "val_r2"

    print(f"[ensemble] Optimizing blend weights for {len(models)} models: {names}")
    weights = _optimize_weights(preds, y_true, task_type)
    blended = sum(w * p for w, p in zip(weights, preds))

    ensemble_score = float(score_fn(y_true, blended))
    best_single = max(float(score_fn(y_true, p)) for p in preds)
    improvement = ensemble_score - best_single

    weight_map = {n: round(float(w), 4) for n, w in zip(names, weights)}
    print(f"[ensemble] Weights: {weight_map}")
    print(f"[ensemble] {metric_name}: {ensemble_score:.4f} (best single: {best_single:.4f}, +{improvement:+.4f})")

    if improvement < -0.001:
        print(f"[ensemble] Ensemble worse than best single — skipping")
        return {"success": False, "reason": "ensemble_worse_than_single"}

    ensemble_model = EnsembleModel(models, weights.tolist(), names)
    save_path = generate_model_path(ensemble_name)
    joblib.dump(ensemble_model, save_path)

    best_info = get_model_info(names[0]) or {}
    register_model(
        model_name=ensemble_name,
        model_path=save_path,
        model_type="ensemble_blend",
        description=f"Blend of {names}",
        metrics={metric_name: ensemble_score, "improvement": improvement},
        feature_names=best_info.get("feature_names", []),
        target_column=target_column,
        hyperparameters={"weights": weight_map},
        training_samples=best_info.get("training_samples", 0),
        classes=best_info.get("classes", []),
    )

    for name in names:
        delete_model(name)

    return {
        "success": True,
        "ensemble_name": ensemble_name,
        "weights": weight_map,
        "component_models": names,
        metric_name: ensemble_score,
        "improvement": improvement,
    }


# =============================================================================
# FULL-DATA RETRAIN
# =============================================================================

def retrain_on_full_data(
    model_name: str,
    train_ref: str,
    val_ref: str | None,
    test_ref: str | None,
    target_column: str,
    task_type: str,
    cv_folds: int = 5,
) -> dict[str, Any]:
    """Retrain a model on all available data (train + val + test) with k-fold CV."""
    model = load_model(model_name)
    info = get_model_info(model_name)
    if model is None or info is None:
        return {"success": False, "reason": f"model '{model_name}' not found"}

    if isinstance(model, EnsembleModel):
        return {"success": False, "reason": "retrain individual models before blending"}

    train_df = get_registered_dataset(train_ref)
    if train_df is None:
        return {"success": False, "reason": f"dataset '{train_ref}' not found"}

    dfs = [train_df] + [get_registered_dataset(r) for r in [val_ref, test_ref] if r]
    dfs = [d for d in dfs if d is not None]
    full_df = pd.concat(dfs, ignore_index=True)
    X, y = full_df.drop(columns=[target_column]), full_df[target_column]

    original_rows, full_rows = len(train_df), len(full_df)
    print(f"[retrain] {original_rows} → {full_rows} rows (+{full_rows - original_rows})")

    scoring, _ = _score_metric(task_type)
    is_clf = task_type == "classification"
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if is_clf else KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print(f"[retrain] {cv_folds}-fold CV ({scoring})...")
    cv_scores = cross_val_score(clone(model), X, y, cv=cv, scoring=scoring, n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    print(f"[retrain] CV: {cv_mean:.4f} ± {cv_std:.4f}")

    final = clone(model)
    final.fit(X, y)

    save_path = info["model_path"]
    joblib.dump(final, save_path)

    register_model(
        model_name=model_name,
        model_path=save_path,
        model_type=info["model_type"],
        description=info.get("description", "") + f" (retrained on {full_rows} rows)",
        metrics={**info.get("metrics", {}), f"cv_{scoring}": cv_mean, f"cv_{scoring}_std": cv_std},
        feature_names=info.get("feature_names", []),
        target_column=target_column,
        hyperparameters=info.get("hyperparameters", {}),
        training_samples=full_rows,
        classes=info.get("classes", []),
    )

    print(f"[retrain] '{model_name}' retrained ({full_rows} rows)")

    return {
        "success": True,
        "model_name": model_name,
        "original_rows": original_rows,
        "full_rows": full_rows,
        f"cv_{scoring}": cv_mean,
        f"cv_{scoring}_std": cv_std,
    }
