"""Generic sklearn training skill with automatic hyperparameter tuning.

Trains any scikit-learn estimator via a unified interface. Builds a
preprocessing pipeline (StandardScaler + OneHotEncoder), optionally runs
RandomizedSearchCV for hyperparameter tuning, evaluates, saves, and
registers the model — all in a single `run(params)` call.
"""

import importlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
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


# ── Estimator catalog ────────────────────────────────────────────────────
# Maps short name → (module_path, class_name).

ESTIMATORS: dict[str, tuple[str, str]] = {
    # Classification
    "LogisticRegression":         ("sklearn.linear_model", "LogisticRegression"),
    "SVC":                        ("sklearn.svm", "SVC"),
    "LinearSVC":                  ("sklearn.svm", "LinearSVC"),
    "KNeighborsClassifier":       ("sklearn.neighbors", "KNeighborsClassifier"),
    "DecisionTreeClassifier":     ("sklearn.tree", "DecisionTreeClassifier"),
    "RandomForestClassifier":     ("sklearn.ensemble", "RandomForestClassifier"),
    "GradientBoostingClassifier": ("sklearn.ensemble", "GradientBoostingClassifier"),
    "AdaBoostClassifier":         ("sklearn.ensemble", "AdaBoostClassifier"),
    "BaggingClassifier":          ("sklearn.ensemble", "BaggingClassifier"),
    "ExtraTreesClassifier":       ("sklearn.ensemble", "ExtraTreesClassifier"),
    "MLPClassifier":              ("sklearn.neural_network", "MLPClassifier"),
    "RidgeClassifier":            ("sklearn.linear_model", "RidgeClassifier"),
    "SGDClassifier":              ("sklearn.linear_model", "SGDClassifier"),
    "GaussianNB":                 ("sklearn.naive_bayes", "GaussianNB"),
    "MultinomialNB":              ("sklearn.naive_bayes", "MultinomialNB"),
    "ComplementNB":               ("sklearn.naive_bayes", "ComplementNB"),
    # Regression
    "SVR":                        ("sklearn.svm", "SVR"),
    "LinearSVR":                  ("sklearn.svm", "LinearSVR"),
    "KNeighborsRegressor":        ("sklearn.neighbors", "KNeighborsRegressor"),
    "DecisionTreeRegressor":      ("sklearn.tree", "DecisionTreeRegressor"),
    "RandomForestRegressor":      ("sklearn.ensemble", "RandomForestRegressor"),
    "GradientBoostingRegressor":  ("sklearn.ensemble", "GradientBoostingRegressor"),
    "AdaBoostRegressor":          ("sklearn.ensemble", "AdaBoostRegressor"),
    "BaggingRegressor":           ("sklearn.ensemble", "BaggingRegressor"),
    "ExtraTreesRegressor":        ("sklearn.ensemble", "ExtraTreesRegressor"),
    "MLPRegressor":               ("sklearn.neural_network", "MLPRegressor"),
    "LinearRegression":           ("sklearn.linear_model", "LinearRegression"),
    "Ridge":                      ("sklearn.linear_model", "Ridge"),
    "Lasso":                      ("sklearn.linear_model", "Lasso"),
    "ElasticNet":                 ("sklearn.linear_model", "ElasticNet"),
    "HuberRegressor":             ("sklearn.linear_model", "HuberRegressor"),
    "SGDRegressor":               ("sklearn.linear_model", "SGDRegressor"),
}

# Defaults applied at construction time (e.g. SVC needs probability=True).
_INIT_DEFAULTS: dict[str, dict] = {
    "SVC": {"probability": True},
    "SGDClassifier": {"loss": "modified_huber"},
    "LogisticRegression": {"max_iter": 1000},
    "RandomForestClassifier": {"n_jobs": -1},
    "RandomForestRegressor": {"n_jobs": -1},
    "ExtraTreesClassifier": {"n_jobs": -1},
    "ExtraTreesRegressor": {"n_jobs": -1},
}

# ── Hyperparameter search spaces ─────────────────────────────────────────
# Each value is a dict of param_name → list_of_candidates.
# Prefixed with the pipeline step name ("model__") at runtime.

_TREE = {"max_depth": [3, 5, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]}
_BOOST = {**_TREE, "n_estimators": [100, 200], "learning_rate": [0.01, 0.05, 0.1, 0.2], "subsample": [0.8, 0.9, 1.0]}
_KNN = {"n_neighbors": [3, 5, 7, 11, 15], "weights": ["uniform", "distance"]}
_FOREST = {"n_estimators": [100, 200], "max_depth": [5, 10, 20, 30], "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5]}

SEARCH_SPACES: dict[str, dict] = {
    "LogisticRegression":         {"C": [0.01, 0.1, 1, 10, 100]},
    "SVC":                        {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"]},
    "SVR":                        {"C": [0.1, 1, 10, 100], "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"], "epsilon": [0.01, 0.1, 0.2]},
    "KNeighborsClassifier":       _KNN,
    "KNeighborsRegressor":        _KNN,
    "DecisionTreeClassifier":     _TREE,
    "DecisionTreeRegressor":      _TREE,
    "RandomForestClassifier":     _FOREST,
    "RandomForestRegressor":      _FOREST,
    "GradientBoostingClassifier":  _BOOST,
    "GradientBoostingRegressor":   _BOOST,
    "AdaBoostClassifier":         {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]},
    "AdaBoostRegressor":          {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5, 1.0]},
    "BaggingClassifier":          {"n_estimators": [10, 50, 100], "max_samples": [0.5, 0.7, 1.0], "max_features": [0.5, 0.7, 1.0]},
    "BaggingRegressor":           {"n_estimators": [10, 50, 100], "max_samples": [0.5, 0.7, 1.0], "max_features": [0.5, 0.7, 1.0]},
    "ExtraTreesClassifier":       _FOREST,
    "ExtraTreesRegressor":        _FOREST,
    "MLPClassifier":              {"hidden_layer_sizes": [(100,), (50, 50), (100, 50), (100, 100)], "alpha": [0.0001, 0.001, 0.01], "learning_rate": ["constant", "adaptive"]},
    "MLPRegressor":               {"hidden_layer_sizes": [(100,), (50, 50), (100, 50), (100, 100)], "alpha": [0.0001, 0.001, 0.01], "learning_rate": ["constant", "adaptive"]},
    "Ridge":                      {"alpha": [0.01, 0.1, 1, 10, 100]},
    "RidgeClassifier":            {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso":                      {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "ElasticNet":                 {"alpha": [0.001, 0.01, 0.1, 1], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    "SGDClassifier":              {"alpha": [0.0001, 0.001, 0.01], "penalty": ["l2", "l1", "elasticnet"]},
    "SGDRegressor":               {"alpha": [0.0001, 0.001, 0.01], "penalty": ["l2", "l1", "elasticnet"]},
    "HuberRegressor":             {"epsilon": [1.1, 1.35, 1.5, 2.0], "alpha": [0.0001, 0.001, 0.01]},
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _resolve_estimator(name: str, overrides: dict | None = None):
    """Import and instantiate an sklearn estimator by name."""
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown estimator '{name}'. Available: {sorted(ESTIMATORS)}")
    module_path, class_name = ESTIMATORS[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**{**_INIT_DEFAULTS.get(name, {}), **(overrides or {})})


def _build_preprocessor(X: pd.DataFrame, categorical_cols: list[str]) -> ColumnTransformer:
    """ColumnTransformer: StandardScaler for numerics, OneHotEncoder for categoricals."""
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def _count_combinations(space: dict) -> int:
    result = 1
    for v in space.values():
        result *= len(v)
    return result


# ── Main entry point ─────────────────────────────────────────────────────

_SUBSAMPLE_SEARCH = 30_000
_MAX_TOTAL_FITS = 40

_SLOW_ESTIMATORS = {
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "MLPClassifier", "MLPRegressor",
    "SVC", "SVR",
    "KNeighborsClassifier", "KNeighborsRegressor",
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "BaggingClassifier", "BaggingRegressor",
    "AdaBoostClassifier", "AdaBoostRegressor",
}


def run(params: dict) -> str:
    """Train any sklearn estimator. See SKILL.md for parameters."""
    estimator_name = params.get("estimator")
    if not estimator_name:
        return f"TRAINING FAILED\nError: 'estimator' is required. Available: {sorted(ESTIMATORS)}"

    dataset_ref = params.get("train_dataset_ref")
    target_column = params.get("target_column")
    model_name = params.get("model_name", f"{estimator_name.lower()}_model")
    for required, label in [(dataset_ref, "train_dataset_ref"), (target_column, "target_column")]:
        if not required:
            return f"TRAINING FAILED\nError: '{label}' is required."

    # ── Load data ────────────────────────────────────────────────────────
    df = get_registered_dataset(dataset_ref)
    if df is None:
        return f"TRAINING FAILED\nError: Dataset '{dataset_ref}' not found."
    if target_column not in df.columns:
        return f"TRAINING FAILED\nError: Target '{target_column}' not in columns: {list(df.columns)}"

    feature_columns = params.get("feature_columns")
    if feature_columns:
        feature_columns = [c for c in feature_columns if c != target_column and c in df.columns]
    else:
        feature_columns = [c for c in df.columns if c != target_column]

    X, y = df[feature_columns], df[target_column]
    n_rows = len(X)

    categorical_cols = params.get("categorical_columns") or X.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c in feature_columns]

    print(f"[sklearn_generic] {estimator_name} | {n_rows} rows × {len(feature_columns)} features")

    # ── Build pipeline ───────────────────────────────────────────────────
    raw_hyperparameters = params.get("hyperparameters", {})
    fixed_params: dict = {}
    search_overrides: dict = {}
    for k, v in raw_hyperparameters.items():
        if isinstance(v, list):
            search_overrides[k] = v
        else:
            fixed_params[k] = v

    try:
        estimator = _resolve_estimator(estimator_name, fixed_params)
    except Exception as e:
        return f"TRAINING FAILED\nError: {e}"

    is_clf = is_classifier(estimator)
    task_type = "classification" if is_clf else "regression"
    step_name = "model"

    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(X, categorical_cols)),
        (step_name, estimator),
    ])

    # ── Auto-tune or direct fit ──────────────────────────────────────────
    auto_tune = params.get("auto_tune", True)
    n_search_iter = params.get("n_search_iter", 20)
    cv_folds = max(2, min(params.get("cv_folds", 5), len(y)))
    best_params: dict = {}
    cv_score: float | None = None

    base_space = SEARCH_SPACES.get(estimator_name, {}).copy()
    base_space.update(search_overrides)

    # ── Adaptive settings ────────────────────────────────────────────────
    is_slow = estimator_name in _SLOW_ESTIMATORS
    if is_slow:
        cv_folds = min(cv_folds, 3)
        n_search_iter = min(n_search_iter, 10)
    if n_rows > _SUBSAMPLE_SEARCH:
        cv_folds = min(cv_folds, 3)

    if auto_tune and base_space:
        space = {f"{step_name}__{k}": v for k, v in base_space.items()}
        scoring = "accuracy" if is_clf else "r2"
        total_combos = _count_combinations(space)
        actual_iter = min(n_search_iter, total_combos)

        # Hard cap on total fits to prevent runaway training
        if actual_iter * cv_folds > _MAX_TOTAL_FITS:
            actual_iter = max(2, _MAX_TOTAL_FITS // cv_folds)

        # Subsample for search when dataset exceeds threshold
        X_search, y_search = X, y
        subsampled = False
        if n_rows > _SUBSAMPLE_SEARCH:
            from sklearn.model_selection import train_test_split
            frac = _SUBSAMPLE_SEARCH / n_rows
            X_search, _, y_search, _ = train_test_split(
                X, y, train_size=frac, stratify=y if is_clf else None,
                random_state=params.get("random_state", 42),
            )
            subsampled = True
            print(f"[sklearn_generic] Subsampled {n_rows} → {len(X_search)} rows for hyperparameter search")

        print(f"[sklearn_generic] RandomizedSearchCV: {actual_iter} iters × {cv_folds}-fold CV"
              f" = {actual_iter * cv_folds} fits")

        search = RandomizedSearchCV(
            pipeline, space,
            n_iter=actual_iter,
            scoring=scoring,
            cv=cv_folds,
            n_jobs=-1,
            random_state=params.get("random_state", 42),
            error_score="raise",
            verbose=1,
        )
        try:
            search.fit(X_search, y_search)
        except Exception as e:
            return f"TRAINING FAILED (during auto-tune)\nError: {e}"
        best_params = search.best_params_
        cv_score = search.best_score_

        if subsampled:
            print(f"[sklearn_generic] Refitting best params on full {n_rows} rows...")
            best_raw = {k.split("__", 1)[-1]: v for k, v in best_params.items()}
            full_estimator = _resolve_estimator(estimator_name, {**fixed_params, **best_raw})
            pipeline = Pipeline([
                ("preprocessor", _build_preprocessor(X, categorical_cols)),
                (step_name, full_estimator),
            ])
            pipeline.fit(X, y)
        else:
            pipeline = search.best_estimator_
    else:
        print(f"[sklearn_generic] Fitting {estimator_name} directly (no search)...")
        try:
            pipeline.fit(X, y)
        except Exception as e:
            return f"TRAINING FAILED\nError: {e}"

    print(f"[sklearn_generic] Training complete.")

    # ── Evaluate on training data ────────────────────────────────────────
    y_pred = pipeline.predict(X)

    try:
        feature_names_out = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    except AttributeError:
        feature_names_out = feature_columns

    metrics: dict = {}
    lines = [
        "=" * 60,
        f"{estimator_name} TRAINING COMPLETE",
        "=" * 60,
        "",
        f"Estimator: {estimator_name}",
        f"Task type: {task_type}",
        f"Samples: {len(df)}",
        f"Features: {len(feature_names_out)}",
    ]

    if is_clf:
        train_acc = float(accuracy_score(y, y_pred))
        metrics["train_accuracy"] = train_acc
        lines.append(f"Train Accuracy: {train_acc:.4f}")

        if hasattr(pipeline, "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X)
                classes = pipeline.classes_
                roc = float(
                    roc_auc_score(y, y_proba[:, 1])
                    if len(classes) == 2
                    else roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")
                )
                metrics["train_roc_auc"] = roc
                lines.append(f"Train ROC-AUC: {roc:.4f}")
            except (ValueError, AttributeError):
                lines.append("Train ROC-AUC: N/A")

        classes_list = [str(c) for c in (pipeline.classes_ if hasattr(pipeline, "classes_") else sorted(y.unique()))]
        lines.extend(["", "CLASSIFICATION REPORT", classification_report(y, y_pred)])
        lines.extend(["CONFUSION MATRIX", str(np.array(confusion_matrix(y, y_pred).tolist()))])
    else:
        r2 = float(r2_score(y, y_pred))
        mae = float(mean_absolute_error(y, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
        metrics.update({"train_r2": r2, "train_mae": mae, "train_rmse": rmse})
        classes_list = []
        lines.extend([f"Train R2: {r2:.4f}", f"Train MAE: {mae:.4f}", f"Train RMSE: {rmse:.4f}"])

    if cv_score is not None:
        lines.extend(["", f"Cross-validation score ({cv_folds}-fold): {cv_score:.4f}"])

    # ── Report best hyperparameters ──────────────────────────────────────
    tuned_params = {k.split("__", 1)[-1]: v for k, v in best_params.items()} if best_params else {}
    clean_params = {**fixed_params, **tuned_params}
    if clean_params:
        lines.extend(["", "BEST HYPERPARAMETERS:"])
        for k, v in clean_params.items():
            lines.append(f"  {k}: {v}")

    # ── Save & register ─────────────────────────────────────────────────
    save_path = generate_model_path(model_name)
    joblib.dump(pipeline, save_path)

    register_model(
        model_name=model_name,
        model_path=save_path,
        model_type=f"sklearn_{estimator_name}",
        description=params.get("description", f"{estimator_name} trained on {dataset_ref}"),
        metrics=metrics,
        feature_names=feature_names_out,
        target_column=target_column,
        hyperparameters=clean_params,
        training_samples=len(df),
        classes=classes_list,
    )

    lines.extend(["", f"MODEL REGISTERED: {model_name}", f"Path: {save_path}", "=" * 60])
    return "\n".join(lines)
