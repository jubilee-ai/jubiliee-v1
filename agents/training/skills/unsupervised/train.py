"""Generic sklearn unsupervised training skill.

Trains clustering, anomaly detection, and dimensionality reduction estimators
through a unified `run(params)` entry point. The skill builds preprocessing
(impute + scale/encode), fits the estimator, computes task-appropriate metrics,
saves, and registers the model artifact.

Optional ``eval_holdout_fraction`` (e.g. 0.15): internal train/holdout split before
the final full-data fit; emits ``val_*`` metrics where the estimator supports
``predict`` (KMeans, GMM, IsolationForest, etc.). Saved artifact is always fit on
all rows after preprocessing is refit on the full feature matrix.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import (AgglomerativeClustering, DBSCAN, KMeans,
                             MiniBatchKMeans)
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_ROOT = Path(__file__).parents[4]
for _p in [
    str(_ROOT / "tools" / "models-tools" / "training"),
    str(_ROOT / "tools" / "data-tools"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_storage import generate_model_path, register_model
from utils import get_registered_dataset


ESTIMATORS = {
    "KMeans": KMeans,
    "MiniBatchKMeans": MiniBatchKMeans,
    "DBSCAN": DBSCAN,
    "AgglomerativeClustering": AgglomerativeClustering,
    "GaussianMixture": GaussianMixture,
    "IsolationForest": IsolationForest,
    "PCA": PCA,
}


def _build_preprocessor(X: pd.DataFrame, categorical_cols: list[str]) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def _fit_unsupervised(
    estimator_name: str,
    hp: dict,
    X_t: np.ndarray | pd.DataFrame,
) -> tuple[object, np.ndarray | None]:
    """Fit estimator on transformed design matrix; return (estimator, labels or None for PCA)."""
    hp = dict(hp)
    hp.setdefault("random_state", 42)
    if estimator_name in {"DBSCAN", "AgglomerativeClustering", "PCA"}:
        hp.pop("random_state", None)
    est = ESTIMATORS[estimator_name](**hp)
    X_df = X_t if isinstance(X_t, pd.DataFrame) else pd.DataFrame(X_t)
    if estimator_name == "GaussianMixture":
        est.fit(X_df)
        labels = est.predict(X_df)
    elif estimator_name == "PCA":
        est.fit(X_df)
        labels = None
    else:
        labels = est.fit_predict(X_df)
    return est, labels


def _predict_labels_val(
    estimator_name: str, estimator: object, X_va_t: np.ndarray
) -> np.ndarray | None:
    """Assign cluster / anomaly labels on holdout for estimators that support out-of-sample predict."""
    if estimator_name in {"PCA", "DBSCAN", "AgglomerativeClustering"}:
        return None
    if not hasattr(estimator, "predict"):
        return None
    X_df = X_va_t if isinstance(X_va_t, pd.DataFrame) else pd.DataFrame(X_va_t)
    try:
        return np.asarray(estimator.predict(X_df))
    except Exception:
        return None


def _compute_unsupervised_metrics(
    estimator_name: str, X_transformed: np.ndarray, labels: np.ndarray | None
) -> dict:
    metrics: dict = {}
    if estimator_name in {"KMeans", "MiniBatchKMeans"} and labels is not None:
        uniq = np.unique(labels)
        if len(uniq) > 1:
            metrics["silhouette_score"] = float(silhouette_score(X_transformed, labels))
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_transformed, labels))
    elif estimator_name == "GaussianMixture":
        if labels is not None and len(np.unique(labels)) > 1:
            metrics["silhouette_score"] = float(silhouette_score(X_transformed, labels))
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_transformed, labels))
    elif estimator_name in {"DBSCAN", "AgglomerativeClustering"} and labels is not None:
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        if valid_mask.sum() >= 2 and len(np.unique(valid_labels)) > 1:
            valid_X = X_transformed[valid_mask]
            metrics["silhouette_score"] = float(silhouette_score(valid_X, valid_labels))
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(valid_X, valid_labels))
        metrics["noise_ratio"] = float((labels == -1).mean()) if estimator_name == "DBSCAN" else 0.0
    elif estimator_name == "IsolationForest" and labels is not None:
        # IsolationForest outputs {-1, 1}. Track anomaly fraction.
        metrics["anomaly_ratio"] = float((labels == -1).mean())
    elif estimator_name == "PCA":
        pass
    return metrics


def run(params: dict) -> str:
    estimator_name = params.get("estimator")
    if not estimator_name:
        return f"TRAINING FAILED\nError: 'estimator' is required. Available: {sorted(ESTIMATORS)}"
    if estimator_name not in ESTIMATORS:
        return f"TRAINING FAILED\nError: Unknown estimator '{estimator_name}'. Available: {sorted(ESTIMATORS)}"

    dataset_ref = params.get("train_dataset_ref")
    model_name = params.get("model_name", f"{estimator_name.lower()}_model")
    if not dataset_ref:
        return "TRAINING FAILED\nError: 'train_dataset_ref' is required."

    df = get_registered_dataset(dataset_ref)
    if df is None:
        return f"TRAINING FAILED\nError: Dataset '{dataset_ref}' not found."

    feature_columns = params.get("feature_columns")
    if feature_columns:
        feature_columns = [c for c in feature_columns if c in df.columns]
    else:
        feature_columns = list(df.columns)
    if not feature_columns:
        return "TRAINING FAILED\nError: No usable feature columns found."

    X = df[feature_columns].copy()
    categorical_cols = params.get("categorical_columns") or X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c in feature_columns]

    raw_hp = dict(params.get("hyperparameters", {}))
    rng_seed = int(raw_hp.get("random_state", 42))

    eval_raw = params.get("eval_holdout_fraction")
    try:
        eval_frac = float(eval_raw) if eval_raw is not None else 0.0
    except (TypeError, ValueError):
        eval_frac = 0.0
    use_holdout = 0 < eval_frac < 0.5 and len(X) >= 40

    hyperparameters = dict(raw_hp)
    hyperparameters.setdefault("random_state", 42)
    if estimator_name in {"DBSCAN", "AgglomerativeClustering", "PCA"}:
        hyperparameters.pop("random_state", None)

    preprocessor = _build_preprocessor(X, categorical_cols)
    holdout_metrics: dict = {}

    if use_holdout:
        X_tr, X_va = train_test_split(
            X, test_size=eval_frac, random_state=rng_seed, shuffle=True
        )
        preprocessor.fit(X_tr)
        X_tr_df = pd.DataFrame(preprocessor.transform(X_tr))
        X_va_df = pd.DataFrame(preprocessor.transform(X_va))
        try:
            est_h, _ = _fit_unsupervised(estimator_name, hyperparameters, X_tr_df)
            lab_va = _predict_labels_val(estimator_name, est_h, X_va_df)
            if lab_va is not None and len(np.unique(lab_va)) > 1:
                vm = _compute_unsupervised_metrics(
                    estimator_name, np.asarray(X_va_df), lab_va
                )
                holdout_metrics = {f"val_{k}": v for k, v in vm.items() if isinstance(v, (int, float))}
        except Exception:
            pass

    X_transformed = preprocessor.fit_transform(X)
    X_transformed_df = pd.DataFrame(X_transformed)

    lines = [
        "=" * 60,
        f"{estimator_name} UNSUPERVISED TRAINING COMPLETE",
        "=" * 60,
        "",
        f"Estimator: {estimator_name}",
        f"Samples: {len(X)}",
        f"Input features: {len(feature_columns)}",
    ]
    if use_holdout:
        lines.append(
            f"Holdout eval: fraction={eval_frac:.3f} (val_* = out-of-sample where supported; "
            "full-fit metrics below — DBSCAN/Agglomerative/PCA skip val predict)"
        )

    try:
        estimator, labels = _fit_unsupervised(estimator_name, hyperparameters, X_transformed_df)
    except Exception as exc:
        return f"TRAINING FAILED\nError during fit: {exc}"

    metrics = _compute_unsupervised_metrics(estimator_name, np.asarray(X_transformed), labels)
    metrics.update(holdout_metrics)
    if use_holdout:
        metrics["eval_holdout_fraction"] = float(eval_frac)
    if estimator_name in {"KMeans", "MiniBatchKMeans"} and hasattr(estimator, "inertia_"):
        metrics["inertia"] = float(estimator.inertia_)
    if estimator_name == "GaussianMixture":
        metrics["aic"] = float(estimator.aic(X_transformed_df))
        metrics["bic"] = float(estimator.bic(X_transformed_df))
    if estimator_name == "PCA":
        explained = estimator.explained_variance_ratio_
        metrics["explained_variance_ratio_sum"] = float(np.sum(explained))
        metrics["n_components"] = int(estimator.n_components_)

    if labels is not None:
        n_clusters = len(np.unique(labels[labels != -1])) if estimator_name == "DBSCAN" else len(np.unique(labels))
        metrics["n_clusters_or_groups"] = int(n_clusters)
        lines.append(f"Discovered groups: {n_clusters}")

    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")

    model_obj = {
        "preprocessor": preprocessor,
        "estimator": estimator,
        "estimator_name": estimator_name,
        "feature_columns": feature_columns,
    }

    save_path = generate_model_path(model_name)
    joblib.dump(model_obj, save_path)

    register_model(
        model_name=model_name,
        model_path=save_path,
        model_type=f"sklearn_unsupervised_{estimator_name}",
        description=params.get("description", f"{estimator_name} trained on {dataset_ref}"),
        metrics=metrics,
        feature_names=feature_columns,
        target_column="",
        hyperparameters={
            **hyperparameters,
            "task_type": "unsupervised",
            **({"eval_holdout_fraction": float(eval_frac)} if use_holdout else {}),
        },
        training_samples=len(df),
        classes=[],
    )

    lines.extend(["", f"MODEL REGISTERED: {model_name}", f"Path: {save_path}", "=" * 60])
    return "\n".join(lines)
