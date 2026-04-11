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
import types
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker:",
    category=UserWarning,
    module=r"joblib\.externals\.loky",
)

# Stub out CuPy modules before sklearn walks its submodules via
# all_estimators().  sklearn 1.8+ ships an array_api_compat shim that
# references cupy; when CuPy is not installed the import fails and
# crashes _discover_estimators().
for _cupy_mod in [
    "sklearn.externals.array_api_compat.cupy",
    "sklearn.externals.array_api_compat.cupy.linalg",
]:
    if _cupy_mod not in sys.modules:
        sys.modules[_cupy_mod] = types.ModuleType(_cupy_mod)

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn import set_config
from sklearn.base import is_classifier, is_regressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils import all_estimators

set_config(array_api_dispatch=False)

_ROOT = Path(__file__).parents[4]
for _p in [
    str(_ROOT / "tools" / "models-tools" / "training"),
    str(_ROOT / "tools" / "data-tools"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_storage import (classification_roc_auc, generate_model_path,
                           register_model)
from utils import get_registered_dataset

# ── Auto-discover estimators ─────────────────────────────────────────────
# Dynamically finds every classifier and regressor in sklearn.
# Meta-estimators (VotingClassifier, StackingClassifier, etc.) that require
# sub-estimator arguments are filtered out automatically.

def _discover_estimators() -> dict[str, tuple[str, str]]:
    """Build estimator catalog from sklearn's own registry."""
    catalog = {}
    try:
        estimator_list = all_estimators(type_filter=["classifier", "regressor"])
    except Exception as exc:
        print(
            f"[supervised/train] sklearn.utils.all_estimators() failed "
            f"({type(exc).__name__}: {exc!r}) — will fall back to SEARCH_SPACES probe if needed."
        )
        estimator_list = []
    for name, cls in estimator_list:
        try:
            sig = inspect.signature(cls.__init__)
            has_required = any(
                p.default is inspect.Parameter.empty and p.name != "self"
                for p in sig.parameters.values()
            )
            if has_required:
                continue
        except (ValueError, TypeError, ModuleNotFoundError, ImportError):
            continue
        catalog[name] = (cls.__module__, name)
    return catalog


# ── Hyperparameter search spaces (before ESTIMATORS: fallback probes these keys) ──
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
    "n_estimators": randint(50, 300),
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
# XGBoost (optional dependency — only in ESTIMATORS when xgboost is installed)
_XGB = {
    "n_estimators": randint(50, 400),
    "max_depth": randint(3, 12),
    "learning_rate": loguniform(5e-3, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_alpha": loguniform(1e-8, 1.0),
    "reg_lambda": loguniform(1e-8, 5.0),
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
        "max_iter": randint(50, 500),
        "learning_rate": loguniform(5e-3, 0.5),
        "max_depth": randint(3, 15),
        "min_samples_leaf": randint(5, 50),
        "l2_regularization": loguniform(1e-6, 10),
        "max_bins": [63, 127, 255],
    },
    "HistGradientBoostingRegressor": {
        "max_iter": randint(50, 500),
        "learning_rate": loguniform(5e-3, 0.5),
        "max_depth": randint(3, 15),
        "min_samples_leaf": randint(5, 50),
        "l2_regularization": loguniform(1e-6, 10),
        "max_bins": [63, 127, 255],
    },
    "XGBClassifier": _XGB,
    "XGBRegressor": _XGB,
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


_SKLEARN_FALLBACK_PACKAGES: tuple[str, ...] = (
    "sklearn.ensemble",
    "sklearn.linear_model",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.tree",
    "sklearn.neural_network",
    "sklearn.naive_bayes",
)

_NON_SKLEARN_SEARCH_KEYS = frozenset({"XGBClassifier", "XGBRegressor"})


def _fallback_catalog_from_search_spaces() -> dict[str, tuple[str, str]]:
    """When all_estimators() fails: resolve sklearn estimators by probing SEARCH_SPACES keys."""
    catalog: dict[str, tuple[str, str]] = {}
    for est_name in SEARCH_SPACES:
        if est_name in _NON_SKLEARN_SEARCH_KEYS:
            continue
        resolved: tuple[str, str] | None = None
        for pkg in _SKLEARN_FALLBACK_PACKAGES:
            try:
                mod = importlib.import_module(pkg)
            except ImportError:
                continue
            cls = getattr(mod, est_name, None)
            if cls is None or not isinstance(cls, type):
                continue
            resolved = (pkg, est_name)
            break
        if resolved is not None:
            catalog[est_name] = resolved
        else:
            print(
                f"[supervised/train] Fallback probe could not resolve {est_name!r} "
                f"in sklearn packages {_SKLEARN_FALLBACK_PACKAGES}."
            )
    return catalog


def _maybe_register_xgboost(catalog: dict[str, tuple[str, str]]) -> None:
    """Register XGBoost estimators when the package is installed (optional dependency)."""
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return
    catalog["XGBClassifier"] = ("xgboost", "XGBClassifier")
    catalog["XGBRegressor"] = ("xgboost", "XGBRegressor")


def _build_estimator_catalog() -> dict[str, tuple[str, str]]:
    catalog = _discover_estimators()
    if not catalog:
        print(
            "[supervised/train] Estimator discovery returned empty catalog; "
            "using probed fallback from SEARCH_SPACES keys (see earlier log if all_estimators failed)."
        )
        catalog = _fallback_catalog_from_search_spaces()
    _maybe_register_xgboost(catalog)
    return catalog


ESTIMATORS: dict[str, tuple[str, str]] = _build_estimator_catalog()

# Defaults applied at construction time.
_INIT_DEFAULTS: dict[str, dict] = {
    "SVC": {"probability": True},
    "SGDClassifier": {"loss": "modified_huber"},
    "LogisticRegression": {"max_iter": 1000},
    "MLPClassifier": {"max_iter": 500},
    "MLPRegressor": {"max_iter": 500},
    "XGBClassifier": {"tree_method": "hist", "n_jobs": 1},
    "XGBRegressor": {"tree_method": "hist", "n_jobs": 1},
}

_PARALLELIZABLE = {
    "RandomForestClassifier", "RandomForestRegressor",
    "ExtraTreesClassifier", "ExtraTreesRegressor",
    "BaggingClassifier", "BaggingRegressor",
    "XGBClassifier", "XGBRegressor",
}


# ── Helpers ──────────────────────────────────────────────────────────────

def _xgboost_runtime_is_unstable() -> bool:
    """Guard known-bad macOS/Python combos that segfault in XGBoost."""
    return sys.platform == "darwin" and sys.version_info >= (3, 14)


def _apply_runtime_stability_guard(
    estimator_name: str,
    hyperparameters: dict,
) -> tuple[str, dict, str | None]:
    """Swap unstable estimators for stable sklearn equivalents."""
    fallback_map = {
        "XGBClassifier": "HistGradientBoostingClassifier",
        "XGBRegressor": "HistGradientBoostingRegressor",
    }
    fallback = fallback_map.get(estimator_name)
    if fallback is None or not _xgboost_runtime_is_unstable():
        return estimator_name, hyperparameters, None

    translated: dict = {}
    param_map = {
        "n_estimators": "max_iter",
        "learning_rate": "learning_rate",
        "max_depth": "max_depth",
        "reg_lambda": "l2_regularization",
        "lambda": "l2_regularization",
    }
    for src, dst in param_map.items():
        if src in hyperparameters:
            translated[dst] = hyperparameters[src]

    note = (
        f"Requested {estimator_name}, but using {fallback} because XGBoost is unstable "
        f"on macOS with Python {sys.version_info.major}.{sys.version_info.minor} in this environment."
    )
    return fallback, translated, note

def _resolve_estimator(name: str, overrides: dict | None = None):
    """Import and instantiate an sklearn estimator by name."""
    if name not in ESTIMATORS:
        raise ValueError(f"Unknown estimator '{name}'. Available: {sorted(ESTIMATORS)}")
    module_path, class_name = ESTIMATORS[name]
    cls = getattr(importlib.import_module(module_path), class_name)
    init_kw = {**_INIT_DEFAULTS.get(name, {})}
    if name in _PARALLELIZABLE:
        init_kw.setdefault("n_jobs", 1)
    init_kw.update(overrides or {})
    return cls(**init_kw)


def _build_preprocessor(X: pd.DataFrame, categorical_cols: list[str]) -> ColumnTransformer:
    """ColumnTransformer: impute + scale numerics, impute + encode categoricals."""
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


def _prepare_split_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Coerce numeric columns to ndarray-backed floats for sklearn compatibility."""
    X = df[feature_columns].copy()
    for col in X.columns:
        if col in categorical_cols or X[col].dtype == object or not pd.api.types.is_numeric_dtype(X[col]):
            continue
        X[col] = np.asarray(X[col], dtype=np.float64)
    return X


def _select_scoring(is_clf: bool, y: pd.Series) -> str:
    """Pick the best CV scoring metric based on data characteristics."""
    if not is_clf:
        return "r2"
    n_classes = y.nunique()
    minority_ratio = y.value_counts().min() / len(y)
    if n_classes == 2 and minority_ratio < 0.1:
        return "average_precision"
    if n_classes == 2:
        return "roc_auc"
    if n_classes > 2 and minority_ratio < 0.15:
        return "f1_weighted"
    return "accuracy"


def _resolve_cv(
    is_clf: bool,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> int | KFold:
    """Fold count or splitter for RandomizedSearchCV.

    Classifiers default to stratified CV in sklearn, which requires
    n_splits <= each class's count. When the rarest class is too small,
    fall back to unstratified KFold so tuning can still run.
    """
    n = len(y)
    cv_folds = max(2, min(cv_folds, n))

    if not is_clf:
        return max(2, min(cv_folds, n))

    min_per = int(y.value_counts().min())
    if min_per >= 2:
        return max(2, min(cv_folds, min_per))

    n_splits = max(2, min(cv_folds, n - 1))
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


# ── Main entry point ─────────────────────────────────────────────────────

_SUBSAMPLE_SEARCH = 20_000
_MAX_TOTAL_FITS = 24
_DEFAULT_SEARCH_ITER = 8
_DEFAULT_CV_FOLDS = 3
_LARGE_DATASET_SEARCH_ITERS = 6
_FOCUSED_SEARCH_ITERS = 4

_SLOW_ESTIMATORS = {
    "GradientBoostingClassifier", "GradientBoostingRegressor",
    "MLPClassifier", "MLPRegressor",
    "SVC", "SVR", "NuSVC", "NuSVR",
    "KNeighborsClassifier", "KNeighborsRegressor",
    "XGBClassifier", "XGBRegressor",
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
    requested_estimator_name = params.get("estimator")
    estimator_name = requested_estimator_name
    if not estimator_name:
        return f"TRAINING FAILED\nError: 'estimator' is required. Available: {sorted(ESTIMATORS)}"

    dataset_ref = params.get("train_dataset_ref")
    target_column = params.get("target_column")
    for required, label in [(dataset_ref, "train_dataset_ref"), (target_column, "target_column")]:
        if not required:
            return f"TRAINING FAILED\nError: '{label}' is required."

    raw_hyperparameters = dict(params.get("hyperparameters", {}) or {})
    estimator_name, raw_hyperparameters, stability_note = _apply_runtime_stability_guard(
        estimator_name,
        raw_hyperparameters,
    )
    model_name = params.get("model_name", f"{estimator_name.lower()}_model")

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

    categorical_cols = params.get("categorical_columns") or df[feature_columns].select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c in feature_columns]

    # Materialize pandas/Arrow/extension dtypes as ndarray-backed columns so sklearn
    # does not treat inputs as foreign array namespaces.
    X = _prepare_split_frame(df, feature_columns, categorical_cols)
    y = pd.Series(np.asarray(df[target_column]), index=df.index, name=target_column)
    n_rows = len(X)

    val_ref = params.get("val_dataset_ref")
    X_val = None
    y_val = None
    if val_ref:
        val_df = get_registered_dataset(val_ref)
        if val_df is None:
            return f"TRAINING FAILED\nError: Validation dataset '{val_ref}' not found."
        missing = [c for c in feature_columns if c not in val_df.columns]
        if missing:
            return f"TRAINING FAILED\nError: Validation dataset missing feature columns: {missing}"
        if target_column not in val_df.columns:
            return f"TRAINING FAILED\nError: Validation target '{target_column}' not in columns: {list(val_df.columns)}"
        X_val = _prepare_split_frame(val_df, feature_columns, categorical_cols)
        y_val = pd.Series(np.asarray(val_df[target_column]), index=val_df.index, name=target_column)

    print(f"[sklearn_generic] {estimator_name} | {n_rows} rows × {len(feature_columns)} features")
    if stability_note:
        print(f"[sklearn_generic] {stability_note}")

    # ── Build pipeline ───────────────────────────────────────────────────
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

    expected_task = params.get("expected_task_type")
    if expected_task == "classification" and not is_classifier(estimator):
        return (
            f"TRAINING FAILED\nError: task_type is classification but '{estimator_name}' "
            f"is not a classifier. Choose a classifier or fix the stated task type."
        )
    if expected_task == "regression" and not is_regressor(estimator):
        return (
            f"TRAINING FAILED\nError: task_type is regression but '{estimator_name}' "
            f"is not a regressor. Choose a regressor or fix the stated task type."
        )

    is_clf = is_classifier(estimator)
    task_type = "classification" if is_clf else "regression"
    step_name = "model"

    pipeline = Pipeline([
        ("preprocessor", _build_preprocessor(X, categorical_cols)),
        (step_name, estimator),
    ])

    # ── Auto-tune or direct fit ──────────────────────────────────────────
    explicit_auto_tune = params.get("auto_tune")
    auto_tune = (
        bool(explicit_auto_tune)
        if explicit_auto_tune is not None
        else not (fixed_params and not search_overrides)
    )
    n_search_iter = params.get("n_search_iter", _DEFAULT_SEARCH_ITER)
    cv_folds = max(2, min(params.get("cv_folds", _DEFAULT_CV_FOLDS), len(y)))
    best_params: dict = {}
    cv_score: float | None = None
    tuning_cv_splits: int | None = None

    base_space = SEARCH_SPACES.get(estimator_name, {}).copy()
    base_space.update(search_overrides)

    # Don't search over params that were explicitly fixed by the user
    for k in fixed_params:
        base_space.pop(k, None)

    # ── Adaptive settings ────────────────────────────────────────────────
    is_slow = estimator_name in _SLOW_ESTIMATORS
    if is_slow:
        cv_folds = min(cv_folds, 3)
        n_search_iter = min(n_search_iter, _LARGE_DATASET_SEARCH_ITERS)
    if n_rows > _SUBSAMPLE_SEARCH:
        cv_folds = min(cv_folds, 3)
        n_search_iter = min(n_search_iter, _LARGE_DATASET_SEARCH_ITERS)
    if search_overrides:
        n_search_iter = min(n_search_iter, _FOCUSED_SEARCH_ITERS)

    if not auto_tune and fixed_params and not search_overrides:
        print(
            "[sklearn_generic] auto_tune disabled by default because exact hyperparameters "
            "were provided; doing a direct fit."
        )

    scoring = _select_scoring(is_clf, y)

    if auto_tune and base_space:
        space = {f"{step_name}__{k}": v for k, v in base_space.items()}
        actual_iter = n_search_iter

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

        rs = params.get("random_state", 42)
        cv_resolved = _resolve_cv(is_clf, y_search, cv_folds, rs)
        n_cv_splits = cv_resolved if isinstance(cv_resolved, int) else cv_resolved.n_splits

        if actual_iter * n_cv_splits > _MAX_TOTAL_FITS:
            actual_iter = max(2, _MAX_TOTAL_FITS // n_cv_splits)

        cv_note = ""
        if is_clf and isinstance(cv_resolved, KFold):
            cv_note = " (unstratified KFold; smallest class < 2 — stratified CV impossible)"
        print(f"[sklearn_generic] RandomizedSearchCV: {actual_iter} iters × {n_cv_splits}-fold CV{cv_note}"
              f" ({scoring}) = {actual_iter * n_cv_splits} fits")

        search = RandomizedSearchCV(
            pipeline, space,
            n_iter=actual_iter,
            scoring=scoring,
            cv=cv_resolved,
            n_jobs=1,
            random_state=rs,
            error_score="raise",
            verbose=1,
        )
        try:
            search.fit(X_search, y_search)
        except Exception as e:
            return f"TRAINING FAILED (during auto-tune)\nError: {e}"
        best_params = search.best_params_
        cv_score = search.best_score_
        tuning_cv_splits = n_cv_splits

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
        f"Scoring: {scoring}",
    ]
    if stability_note:
        lines.append(
            f"Runtime fallback: requested {requested_estimator_name}, trained {estimator_name}"
        )

    if is_clf:
        train_acc = float(accuracy_score(y, y_pred))
        metrics["train_accuracy"] = train_acc
        lines.append(f"Train Accuracy: {train_acc:.4f}")

        roc = classification_roc_auc(pipeline, X, y)
        if roc is not None:
            metrics["train_roc_auc"] = roc
            lines.append(f"Train ROC-AUC: {roc:.4f}")
        else:
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

    if X_val is not None and y_val is not None:
        y_val_pred = pipeline.predict(X_val)
        lines.append("")
        lines.append("VALIDATION METRICS")
        if is_clf:
            val_acc = float(accuracy_score(y_val, y_val_pred))
            metrics["val_accuracy"] = val_acc
            lines.append(f"Val Accuracy: {val_acc:.4f}")

            val_roc = classification_roc_auc(pipeline, X_val, y_val)
            if val_roc is not None:
                metrics["val_roc_auc"] = val_roc
                lines.append(f"Val ROC-AUC: {val_roc:.4f}")
            else:
                lines.append("Val ROC-AUC: N/A")
        else:
            val_r2 = float(r2_score(y_val, y_val_pred))
            val_mae = float(mean_absolute_error(y_val, y_val_pred))
            val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
            metrics.update({"val_r2": val_r2, "val_mae": val_mae, "val_rmse": val_rmse})
            lines.extend([
                f"Val R2: {val_r2:.4f}",
                f"Val MAE: {val_mae:.4f}",
                f"Val RMSE: {val_rmse:.4f}",
            ])

    if cv_score is not None:
        cv_shown = tuning_cv_splits if tuning_cv_splits is not None else cv_folds
        lines.extend(["", f"Cross-validation score ({cv_shown}-fold, {scoring}): {cv_score:.4f}"])

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
        # Raw dataframe columns (pipeline input), not one-hot / transformed names
        feature_names=feature_columns,
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
