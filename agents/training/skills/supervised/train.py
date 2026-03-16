"""Generic sklearn training skill with automatic hyperparameter tuning.

Trains any scikit-learn estimator via a unified interface. Builds a
preprocessing pipeline (impute + scale/encode), optionally runs
RandomizedSearchCV for hyperparameter tuning, evaluates, saves, and
registers the model — all in a single `run(params)` call.

Estimators are auto-discovered via `sklearn.utils.all_estimators()`,
so any classifier or regressor in the installed sklearn version is
available without code changes. Hyperparameter search uses scipy.stats
distributions for efficient continuous sampling.
"""

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import is_classifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.utils import all_estimators

_ROOT = Path(__file__).parents[4]
for _p in [
    str(_ROOT / "tools" / "models-tools" / "training"),
    str(_ROOT / "tools" / "data-tools"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_storage import generate_model_path, register_model
from utils import get_registered_dataset


# ── Auto-discover estimators ─────────────────────────────────────────────
# Dynamically finds every classifier and regressor in sklearn.
# Meta-estimators (VotingClassifier, StackingClassifier, etc.) that require
# sub-estimator arguments are filtered out automatically.

def _discover_estimators() -> dict[str, tuple[str, str]]:
    """Build estimator catalog from sklearn's own registry."""
    catalog = {}
    for name, cls in all_estimators(type_filter=["classifier", "regressor"]):
        try:
            sig = inspect.signature(cls.__init__)
            has_required = any(
                p.default is inspect.Parameter.empty and p.name != "self"
                for p in sig.parameters.values()
            )
            if has_required:
                continue
        except (ValueError, TypeError):
            continue
        catalog[name] = (cls.__module__, name)
    return catalog


ESTIMATORS: dict[str, tuple[str, str]] = _discover_estimators()

# Defaults applied at construction time.
_INIT_DEFAULTS: dict[str, dict] = {
    "SVC": {"probability": True},
    "SGDClassifier": {"loss": "modified_huber"},
    "LogisticRegression": {"max_iter": 1000},
    "MLPClassifier": {"max_iter": 500},
    "MLPRegressor": {"max_iter": 500},
}

_PARALLELIZABLE = {
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "BaggingClassifier", "BaggingRegressor",
}

def _is_tree_based(estimator) -> bool:
    """Detect tree-based models via sklearn class hierarchy."""
    from sklearn.tree import BaseDecisionTree

    if isinstance(estimator, BaseDecisionTree):
        return True

    name = type(estimator).__name__
    if any(kw in name for kw in ("Forest", "GradientBoosting", "HistGradientBoosting")):
        return True

    # Meta-estimators (AdaBoost, Bagging) — check their wrapped base estimator
    base = getattr(estimator, "estimator", None)
    if base is not None:
        return _is_tree_based(base)
    # AdaBoost/Bagging default to DecisionTree when estimator is None
    if any(kw in name for kw in ("AdaBoost", "Bagging")):
        return True

    return False


# ── Hyperparameter search spaces ─────────────────────────────────────────
# Uses scipy.stats distributions for continuous/integer parameters so
# RandomizedSearchCV samples broadly instead of from fixed grids.
# Estimators without a defined space still work — they just skip tuning.

_TREE = {
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
}
_FOREST = {
    **_TREE,
    "n_estimators": randint(100, 1000),
    "max_features": ["sqrt", "log2", 0.2, 0.3, 0.5],
}
_BOOST = {
    **_TREE,
    "n_estimators": randint(50, 300),
    "learning_rate": loguniform(5e-3, 0.5),
    "subsample": uniform(0.6, 0.4),
}
_KNN = {
    "n_neighbors": randint(3, 25),
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
_MLP = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (100, 100), (200,), (100, 50, 25)],
    "alpha": loguniform(1e-5, 1e-1),
    "learning_rate": ["constant", "invscaling", "adaptive"],
}

SEARCH_SPACES: dict[str, dict] = {
    # Linear classifiers
    "LogisticRegression":  {"C": loguniform(1e-3, 1e3), "solver": ["lbfgs", "saga"]},
    "RidgeClassifier":     {"alpha": loguniform(1e-3, 1e3)},
    "SGDClassifier":       {"alpha": loguniform(1e-5, 1e-1), "penalty": ["l2", "l1", "elasticnet"]},
    "PassiveAggressiveClassifier": {"C": loguniform(1e-3, 1e2)},
    # SVM
    "SVC":       {"C": loguniform(1e-2, 1e3), "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"]},
    "LinearSVC": {"C": loguniform(1e-2, 1e3)},
    "NuSVC":     {"nu": uniform(0.05, 0.9), "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"]},
    "SVR":       {"C": loguniform(1e-2, 1e3), "kernel": ["rbf", "linear", "poly"], "gamma": ["scale", "auto"], "epsilon": loguniform(0.01, 1.0)},
    "LinearSVR": {"C": loguniform(1e-2, 1e3), "epsilon": loguniform(0.01, 1.0)},
    "NuSVR":     {"nu": uniform(0.1, 0.8), "C": loguniform(1e-2, 1e3), "kernel": ["rbf", "linear", "poly"]},
    # KNN
    "KNeighborsClassifier": _KNN,
    "KNeighborsRegressor":  _KNN,
    # Trees
    "DecisionTreeClassifier": _TREE,
    "DecisionTreeRegressor":  _TREE,
    # Forests
    "RandomForestClassifier": _FOREST,
    "RandomForestRegressor":  _FOREST,
    "ExtraTreesClassifier":   _FOREST,
    "ExtraTreesRegressor":    _FOREST,
    # Gradient boosting
    "GradientBoostingClassifier": _BOOST,
    "GradientBoostingRegressor":  _BOOST,
    # Histogram-based gradient boosting (sklearn's fastest tree model, handles NaN natively)
    "HistGradientBoostingClassifier": {
        "max_iter": randint(100, 1500),
        "learning_rate": loguniform(5e-3, 0.3),
        "max_depth": randint(3, 15),
        "min_samples_leaf": randint(5, 50),
        "l2_regularization": loguniform(1e-6, 10),
        "max_bins": [63, 127, 255],
        "max_features": uniform(0.5, 0.5),
    },
    "HistGradientBoostingRegressor": {
        "max_iter": randint(100, 1500),
        "learning_rate": loguniform(5e-3, 0.3),
        "max_depth": randint(3, 15),
        "min_samples_leaf": randint(5, 50),
        "l2_regularization": loguniform(1e-6, 10),
        "max_bins": [63, 127, 255],
        "max_features": uniform(0.5, 0.5),
    },
    # Other boosting
    "AdaBoostClassifier": {"n_estimators": randint(30, 300), "learning_rate": loguniform(5e-3, 2.0)},
    "AdaBoostRegressor":  {"n_estimators": randint(30, 300), "learning_rate": loguniform(5e-3, 2.0)},
    "BaggingClassifier":  {"n_estimators": randint(10, 150), "max_samples": uniform(0.5, 0.5), "max_features": uniform(0.5, 0.5)},
    "BaggingRegressor":   {"n_estimators": randint(10, 150), "max_samples": uniform(0.5, 0.5), "max_features": uniform(0.5, 0.5)},
    # MLP
    "MLPClassifier": _MLP,
    "MLPRegressor":  _MLP,
    # Linear regressors
    "Ridge":           {"alpha": loguniform(1e-3, 1e3)},
    "Lasso":           {"alpha": loguniform(1e-4, 1e2)},
    "ElasticNet":      {"alpha": loguniform(1e-4, 1e2), "l1_ratio": uniform(0.05, 0.9)},
    "SGDRegressor":    {"alpha": loguniform(1e-5, 1e-1), "penalty": ["l2", "l1", "elasticnet"]},
    "HuberRegressor":  {"epsilon": uniform(1.05, 1.95), "alpha": loguniform(1e-5, 1e-1)},
    "PassiveAggressiveRegressor": {"C": loguniform(1e-3, 1e2)},
    # Naive Bayes
    "GaussianNB":    {"var_smoothing": loguniform(1e-12, 1e-6)},
    "MultinomialNB": {"alpha": loguniform(1e-3, 10)},
    "ComplementNB":  {"alpha": loguniform(1e-3, 10)},
    "BernoulliNB":   {"alpha": loguniform(1e-3, 10)},
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _resolve_estimator(name: str, overrides: dict | None = None):
    """Import and instantiate an sklearn estimator by name."""
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown estimator '{name}'. Available: {sorted(ESTIMATORS)}")
    module_path, class_name = ESTIMATORS[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    init_kw = {**_INIT_DEFAULTS.get(name, {})}
    if name in _PARALLELIZABLE:
        init_kw.setdefault("n_jobs", -1)
    init_kw.update(overrides or {})
    return cls(**init_kw)


def _build_preprocessor(X: pd.DataFrame, categorical_cols: list[str], is_tree: bool = False) -> ColumnTransformer:
    """ColumnTransformer: impute + scale numerics, impute + encode categoricals.

    For tree-based models: skip scaling (trees are scale-invariant) and use
    OrdinalEncoder instead of OneHotEncoder (avoids feature fragmentation).
    """
    from sklearn.preprocessing import OrdinalEncoder

    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    transformers = []
    if numeric_cols:
        if is_tree:
            transformers.append(("num", SimpleImputer(strategy="median"), numeric_cols))
        else:
            transformers.append(("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols))
    if categorical_cols:
        if is_tree:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]), categorical_cols))
        else:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols))
    return ColumnTransformer(transformers=transformers, remainder="passthrough")


def _select_scoring(is_clf: bool, y: pd.Series) -> str:
    """Pick the best CV scoring metric based on data characteristics."""
    if not is_clf:
        return "r2"
    n_classes = y.nunique()
    minority_ratio = y.value_counts().min() / len(y)
    if n_classes == 2 and minority_ratio < 0.3:
        return "roc_auc"
    if n_classes > 2 and minority_ratio < 0.15:
        return "f1_weighted"
    return "accuracy"


# ── Main entry point ─────────────────────────────────────────────────────

_SUBSAMPLE_SEARCH = 50_000
_MAX_TOTAL_FITS = 200

_SLOW_ESTIMATORS = {
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "MLPClassifier", "MLPRegressor",
    "SVC", "SVR", "NuSVC", "NuSVR",
    "KNeighborsClassifier", "KNeighborsRegressor",
}


def _get_tunable_params_summary(estimator_name: str) -> str:
    """Call extract_estimator_params and return a short summary of tunable params."""
    try:
        _script = Path(__file__).parent.parent / "scripts" / "extract_params.py"
        _spec = importlib.util.spec_from_file_location("extract_params", str(_script))
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        info = _mod.extract_estimator_params(estimator_name)
        if "_error" in info:
            return ""
        tunable = {
            k: v for k, v in info.items()
            if isinstance(v, dict) and v.get("tunable")
        }
        if not tunable:
            return ""
        lines = ["", "TUNABLE PARAMETERS (from extract_estimator_params):"]
        for name, meta in tunable.items():
            default = meta.get("default")
            kind = meta.get("type", "?")
            choices = meta.get("choices")
            scale = meta.get("scale", "")
            parts = [f"type={kind}", f"default={default}"]
            if choices:
                parts.append(f"choices={choices}")
            if scale:
                parts.append(f"scale={scale}")
            lines.append(f"  {name}: {', '.join(parts)}")
        return "\n".join(lines)
    except Exception:
        return ""


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
    is_tree = _is_tree_based(estimator)
    task_type = "classification" if is_clf else "regression"
    step_name = "model"

    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(X, categorical_cols, is_tree=is_tree)),
        (step_name, estimator),
    ])

    # ── Auto-tune or direct fit ──────────────────────────────────────────
    auto_tune = params.get("auto_tune", True)
    n_search_iter = params.get("n_search_iter", 30)
    cv_folds = max(2, min(params.get("cv_folds", 5), len(y)))
    best_params: dict = {}
    cv_score: float | None = None

    base_space = SEARCH_SPACES.get(estimator_name, {}).copy()
    base_space.update(search_overrides)

    # Don't search over params that were explicitly fixed by the user
    for k in fixed_params:
        base_space.pop(k, None)

    # ── Adaptive settings ────────────────────────────────────────────────
    is_slow = estimator_name in _SLOW_ESTIMATORS
    if is_slow:
        cv_folds = min(cv_folds, 3)
        n_search_iter = min(n_search_iter, 10)
    if n_rows > _SUBSAMPLE_SEARCH:
        cv_folds = min(cv_folds, 3)

    scoring = _select_scoring(is_clf, y)

    if auto_tune and base_space:
        space = {f"{step_name}__{k}": v for k, v in base_space.items()}
        actual_iter = n_search_iter

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
              f" ({scoring}) = {actual_iter * cv_folds} fits")

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
                ("preprocessor", _build_preprocessor(X, categorical_cols, is_tree=is_tree)),
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

    # ── Calibrate classifier probabilities ────────────────────────────────
    if is_clf and hasattr(pipeline, "predict_proba") and n_rows >= 500:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        from sklearn.model_selection import train_test_split as _cal_split

        try:
            cal_size = min(0.15, 5000 / n_rows)
            X_main, X_cal, y_main, y_cal = _cal_split(
                X, y, test_size=cal_size, stratify=y, random_state=42,
            )
            pipeline.fit(X_main, y_main)
            calibrated = CalibratedClassifierCV(FrozenEstimator(pipeline), method="isotonic")
            calibrated.fit(X_cal, y_cal)
            pipeline = calibrated
            print(f"[sklearn_generic] Calibrated probabilities (isotonic, {len(X_cal)} cal samples)")
        except Exception as e:
            print(f"[sklearn_generic] Calibration skipped: {e}")

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
        f"Scoring: {scoring}",
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
        lines.extend(["", f"Cross-validation score ({cv_folds}-fold, {scoring}): {cv_score:.4f}"])

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

    lines.extend(["", f"MODEL REGISTERED: {model_name}", f"Path: {save_path}"])

    tunable_summary = _get_tunable_params_summary(estimator_name)
    if tunable_summary:
        lines.append(tunable_summary)

    lines.append("=" * 60)
    return "\n".join(lines)
