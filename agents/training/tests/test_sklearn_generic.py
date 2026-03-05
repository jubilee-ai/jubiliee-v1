"""
Tests for the sklearn_generic training skill.

Run: python -m agents.training.tests.test_sklearn_generic
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

_ROOT = Path(__file__).parents[3]
for _p in [
    str(_ROOT / "tools" / "data-tools"),
    str(_ROOT / "tools" / "models-tools" / "training"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import register_dataset, get_registered_dataset
from model_storage import list_models, delete_model, load_model


def _make_classification_dataset(n_samples=200, n_features=8, ref="test_clf"):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=5, n_redundant=1, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["target"] = y
    register_dataset(ref, df)
    print(f"  Registered classification dataset '{ref}': {df.shape}")
    return ref


def _make_regression_dataset(n_samples=200, n_features=8, ref="test_reg"):
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features,
        n_informative=5, noise=10, random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["target"] = y
    register_dataset(ref, df)
    print(f"  Registered regression dataset '{ref}': {df.shape}")
    return ref


def _make_mixed_dataset(ref="test_mixed"):
    """Dataset with both numeric and categorical columns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "income": np.random.normal(50000, 15000, n),
        "debt_ratio": np.random.uniform(0, 1, n),
        "region": np.random.choice(["north", "south", "east", "west"], n),
        "employed": np.random.choice(["yes", "no"], n),
        "target": np.random.randint(0, 2, n),
    })
    register_dataset(ref, df)
    print(f"  Registered mixed dataset '{ref}': {df.shape}")
    return ref


def _make_nan_dataset(ref="test_nan"):
    """Dataset with missing values in both numeric and categorical columns."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, n).astype(float),
        "income": np.random.normal(50000, 15000, n),
        "score": np.random.uniform(0, 100, n),
        "region": np.random.choice(["north", "south", "east", "west"], n),
        "target": np.random.randint(0, 2, n),
    })
    df.loc[df.sample(frac=0.1, random_state=1).index, "age"] = np.nan
    df.loc[df.sample(frac=0.15, random_state=2).index, "income"] = np.nan
    df.loc[df.sample(frac=0.08, random_state=3).index, "region"] = None
    register_dataset(ref, df)
    nan_count = df.isna().sum().sum()
    print(f"  Registered NaN dataset '{ref}': {df.shape}, {nan_count} missing values")
    return ref


def _cleanup_model(name):
    try:
        delete_model(name)
    except Exception:
        pass


def _run_skill(params):
    """Import and run the sklearn_generic skill."""
    from agents.training.skills.sklearn_generic.train import run
    return run(params)


# ═══════════════════════════════════════════════════════════════════════
# Test 1: Basic classification with multiple estimators
# ═══════════════════════════════════════════════════════════════════════

def test_classification_estimators():
    print("\n" + "=" * 70)
    print("TEST 1: Classification — multiple estimators")
    print("=" * 70)

    ref = _make_classification_dataset(ref="test_clf_multi")
    estimators = [
        "LogisticRegression",
        "RandomForestClassifier",
        "GradientBoostingClassifier",
        "SVC",
        "KNeighborsClassifier",
        "DecisionTreeClassifier",
        "GaussianNB",
    ]

    results = {}
    for est in estimators:
        model_name = f"test_{est.lower()}"
        print(f"\n  ── Training {est} ──")
        t0 = time.time()

        output = _run_skill({
            "estimator": est,
            "model_name": model_name,
            "train_dataset_ref": ref,
            "target_column": "target",
            "n_search_iter": 5,
            "cv_folds": 3,
        })

        elapsed = time.time() - t0
        success = "TRAINING COMPLETE" in output
        acc_line = [l for l in output.split("\n") if "Train Accuracy" in l]
        acc = acc_line[0].split(": ")[1] if acc_line else "N/A"
        cv_line = [l for l in output.split("\n") if "Cross-validation" in l]
        cv = cv_line[0].split(": ")[1] if cv_line else "N/A"

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {est}: accuracy={acc}, cv={cv}, time={elapsed:.1f}s")
        if not success:
            print(f"  OUTPUT: {output[:200]}")

        results[est] = success
        _cleanup_model(model_name)

    passed = sum(results.values())
    print(f"\n  Classification: {passed}/{len(estimators)} passed")
    assert all(results.values()), f"Failed: {[k for k, v in results.items() if not v]}"


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Regression with multiple estimators
# ═══════════════════════════════════════════════════════════════════════

def test_regression_estimators():
    print("\n" + "=" * 70)
    print("TEST 2: Regression — multiple estimators")
    print("=" * 70)

    ref = _make_regression_dataset(ref="test_reg_multi")
    estimators = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
        "SVR",
        "KNeighborsRegressor",
    ]

    results = {}
    for est in estimators:
        model_name = f"test_{est.lower()}"
        print(f"\n  ── Training {est} ──")
        t0 = time.time()

        output = _run_skill({
            "estimator": est,
            "model_name": model_name,
            "train_dataset_ref": ref,
            "target_column": "target",
            "n_search_iter": 5,
            "cv_folds": 3,
        })

        elapsed = time.time() - t0
        success = "TRAINING COMPLETE" in output
        r2_line = [l for l in output.split("\n") if "Train R2" in l]
        r2 = r2_line[0].split(": ")[1] if r2_line else "N/A"

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {est}: R²={r2}, time={elapsed:.1f}s")
        if not success:
            print(f"  OUTPUT: {output[:200]}")

        results[est] = success
        _cleanup_model(model_name)

    passed = sum(results.values())
    print(f"\n  Regression: {passed}/{len(estimators)} passed")
    assert all(results.values()), f"Failed: {[k for k, v in results.items() if not v]}"


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Hyperparameters — fixed values vs list values
# ═══════════════════════════════════════════════════════════════════════

def test_hyperparameter_handling():
    print("\n" + "=" * 70)
    print("TEST 3: Hyperparameter handling (fixed vs list)")
    print("=" * 70)

    ref = _make_classification_dataset(ref="test_hp")

    # Test A: fixed params only (no auto-tune for these)
    print("\n  ── 3a: Fixed hyperparameters ──")
    output = _run_skill({
        "estimator": "LogisticRegression",
        "model_name": "test_hp_fixed",
        "train_dataset_ref": ref,
        "target_column": "target",
        "hyperparameters": {"class_weight": "balanced", "C": 0.1},
    })
    assert "TRAINING COMPLETE" in output, f"Fixed params failed: {output[:200]}"
    assert "class_weight" in output, "Fixed param 'class_weight' not in output"
    print(f"  [PASS] Fixed params: class_weight=balanced, C=0.1")
    _cleanup_model("test_hp_fixed")

    # Test B: list params (search space overrides)
    print("\n  ── 3b: List hyperparameters (search space) ──")
    output = _run_skill({
        "estimator": "LogisticRegression",
        "model_name": "test_hp_list",
        "train_dataset_ref": ref,
        "target_column": "target",
        "hyperparameters": {
            "class_weight": ["balanced", None],
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "n_search_iter": 5,
        "cv_folds": 3,
    })
    assert "TRAINING COMPLETE" in output, f"List params failed: {output[:200]}"
    assert "Cross-validation" in output, "No CV score in output"
    print(f"  [PASS] List params merged into search space")
    _cleanup_model("test_hp_list")

    # Test C: mixed fixed + list
    print("\n  ── 3c: Mixed fixed + list hyperparameters ──")
    output = _run_skill({
        "estimator": "LogisticRegression",
        "model_name": "test_hp_mixed",
        "train_dataset_ref": ref,
        "target_column": "target",
        "hyperparameters": {
            "class_weight": "balanced",
            "C": [0.01, 0.1, 1.0, 10.0],
        },
        "n_search_iter": 4,
        "cv_folds": 3,
    })
    assert "TRAINING COMPLETE" in output, f"Mixed params failed: {output[:200]}"
    assert "class_weight: balanced" in output, "Fixed param not preserved"
    print(f"  [PASS] Mixed: class_weight fixed, C searched")
    _cleanup_model("test_hp_mixed")

    # Test D: auto_tune=False with explicit params
    print("\n  ── 3d: auto_tune=False ──")
    output = _run_skill({
        "estimator": "Ridge",
        "model_name": "test_hp_notune",
        "train_dataset_ref": _make_regression_dataset(ref="test_hp_reg"),
        "target_column": "target",
        "auto_tune": False,
        "hyperparameters": {"alpha": 10.0},
    })
    assert "TRAINING COMPLETE" in output, f"No-tune failed: {output[:200]}"
    assert "Cross-validation" not in output, "CV should not appear with auto_tune=False"
    print(f"  [PASS] auto_tune=False, direct fit with alpha=10.0")
    _cleanup_model("test_hp_notune")

    print(f"\n  Hyperparameter handling: 4/4 passed")


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Mixed data types (numeric + categorical)
# ═══════════════════════════════════════════════════════════════════════

def test_mixed_data():
    print("\n" + "=" * 70)
    print("TEST 4: Mixed data types (numeric + categorical)")
    print("=" * 70)

    ref = _make_mixed_dataset(ref="test_mixed_data")

    estimators = ["LogisticRegression", "RandomForestClassifier", "SVC"]
    results = {}
    for est in estimators:
        model_name = f"test_mixed_{est.lower()}"
        output = _run_skill({
            "estimator": est,
            "model_name": model_name,
            "train_dataset_ref": ref,
            "target_column": "target",
            "n_search_iter": 3,
            "cv_folds": 3,
        })
        success = "TRAINING COMPLETE" in output
        status = "PASS" if success else "FAIL"
        has_cat = "cat__" in output or "num__" in output
        print(f"  [{status}] {est}: categorical encoding={'detected' if has_cat else 'not detected'}")
        if not success:
            print(f"  OUTPUT: {output[:300]}")
        results[est] = success
        _cleanup_model(model_name)

    passed = sum(results.values())
    print(f"\n  Mixed data: {passed}/{len(estimators)} passed")
    assert all(results.values()), f"Failed: {[k for k, v in results.items() if not v]}"


# ═══════════════════════════════════════════════════════════════════════
# Test 5: Model registration and prediction
# ═══════════════════════════════════════════════════════════════════════

def test_model_registration_and_predict():
    print("\n" + "=" * 70)
    print("TEST 5: Model registration & prediction")
    print("=" * 70)

    ref = _make_classification_dataset(ref="test_predict")

    output = _run_skill({
        "estimator": "RandomForestClassifier",
        "model_name": "test_predict_rf",
        "train_dataset_ref": ref,
        "target_column": "target",
        "auto_tune": False,
        "hyperparameters": {"n_estimators": 50, "max_depth": 5},
    })
    assert "MODEL REGISTERED: test_predict_rf" in output

    models = [m for m in list_models() if m["model_name"] == "test_predict_rf"]
    assert len(models) == 1, f"Model not found in registry"
    info = models[0]
    print(f"  Registry entry: model_type={info['model_type']}, features={len(info['feature_names'])}")
    assert info["model_type"] == "sklearn_RandomForestClassifier"
    assert info["target_column"] == "target"
    assert len(info["feature_names"]) > 0

    model = load_model("test_predict_rf")
    df = get_registered_dataset(ref)
    X = df.drop(columns=["target"])
    preds = model.predict(X)
    print(f"  Predictions: shape={preds.shape}, unique={np.unique(preds)}")
    assert len(preds) == len(df)

    probas = model.predict_proba(X)
    print(f"  Probabilities: shape={probas.shape}")
    assert probas.shape == (len(df), 2)

    print(f"  [PASS] Model registered, loaded, and predictions verified")
    _cleanup_model("test_predict_rf")


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Skill registry integration
# ═══════════════════════════════════════════════════════════════════════

def test_skill_registry():
    print("\n" + "=" * 70)
    print("TEST 6: Skill registry integration")
    print("=" * 70)

    from agents.training.skill_registry import (
        _discover_skills,
        _load_skill_prompt,
        _run_skill,
        build_alias_map,
    )

    skills = _discover_skills()
    assert "sklearn_generic" in skills
    print(f"  [PASS] Discovery: {len(skills)} skill(s) found, sklearn_generic present")

    prompt = _load_skill_prompt("sklearn_generic")
    assert "estimator" in prompt.lower()
    assert "GradientBoostingClassifier" in prompt
    assert "HistGradientBoostingClassifier" in prompt
    print(f"  [PASS] SKILL.md loaded: {len(prompt)} chars, includes HistGradientBoosting")

    alias_map = build_alias_map()
    test_aliases = [
        "svm", "knn", "gradient boosting", "mlp", "ridge",
        "logistic regression", "random forest",
        "hist gradient boosting", "hgb", "passive aggressive",
    ]
    for alias in test_aliases:
        assert alias in alias_map, f"Alias '{alias}' not found"
        assert alias_map[alias] == "sklearn_generic", f"Alias '{alias}' -> {alias_map[alias]}"
    print(f"  [PASS] Alias map: {len(alias_map)} aliases, all route to sklearn_generic")

    ref = _make_classification_dataset(ref="test_registry_exec")
    result = _run_skill("sklearn_generic", {
        "estimator": "DecisionTreeClassifier",
        "model_name": "test_registry_dt",
        "train_dataset_ref": ref,
        "target_column": "target",
        "auto_tune": False,
        "hyperparameters": {"max_depth": 3},
    })
    assert "TRAINING COMPLETE" in result
    print(f"  [PASS] Execution via _run_skill: DecisionTreeClassifier trained")
    _cleanup_model("test_registry_dt")


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Error handling
# ═══════════════════════════════════════════════════════════════════════

def test_error_handling():
    print("\n" + "=" * 70)
    print("TEST 7: Error handling")
    print("=" * 70)

    ref = _make_classification_dataset(ref="test_errors")

    output = _run_skill({"model_name": "x", "train_dataset_ref": ref, "target_column": "target"})
    assert "TRAINING FAILED" in output and "'estimator' is required" in output
    print(f"  [PASS] Missing estimator: caught")

    output = _run_skill({"estimator": "FakeModel", "model_name": "x", "train_dataset_ref": ref, "target_column": "target"})
    assert "TRAINING FAILED" in output and "Unknown estimator" in output
    print(f"  [PASS] Unknown estimator: caught")

    output = _run_skill({"estimator": "SVC", "model_name": "x", "train_dataset_ref": "nonexistent", "target_column": "target"})
    assert "TRAINING FAILED" in output and "not found" in output
    print(f"  [PASS] Missing dataset: caught")

    output = _run_skill({"estimator": "SVC", "model_name": "x", "train_dataset_ref": ref, "target_column": "nonexistent"})
    assert "TRAINING FAILED" in output and "not in columns" in output
    print(f"  [PASS] Missing target column: caught")

    output = _run_skill({"estimator": "SVC", "model_name": "x"})
    assert "TRAINING FAILED" in output
    print(f"  [PASS] Missing train_dataset_ref: caught")

    print(f"\n  Error handling: 5/5 passed")


# ═══════════════════════════════════════════════════════════════════════
# Test 8: HistGradientBoosting estimators
# ═══════════════════════════════════════════════════════════════════════

def test_hist_gradient_boosting():
    print("\n" + "=" * 70)
    print("TEST 8: HistGradientBoosting estimators")
    print("=" * 70)

    clf_ref = _make_classification_dataset(ref="test_hgb_clf")
    reg_ref = _make_regression_dataset(ref="test_hgb_reg")

    # Classification
    print("\n  ── HistGradientBoostingClassifier ──")
    output = _run_skill({
        "estimator": "HistGradientBoostingClassifier",
        "model_name": "test_hgb_clf",
        "train_dataset_ref": clf_ref,
        "target_column": "target",
        "n_search_iter": 5,
        "cv_folds": 3,
    })
    assert "TRAINING COMPLETE" in output, f"HGB classifier failed: {output[:300]}"
    assert "Cross-validation" in output
    acc_line = [l for l in output.split("\n") if "Train Accuracy" in l]
    acc = acc_line[0].split(": ")[1] if acc_line else "N/A"
    print(f"  [PASS] HistGradientBoostingClassifier: accuracy={acc}")
    _cleanup_model("test_hgb_clf")

    # Regression
    print("\n  ── HistGradientBoostingRegressor ──")
    output = _run_skill({
        "estimator": "HistGradientBoostingRegressor",
        "model_name": "test_hgb_reg",
        "train_dataset_ref": reg_ref,
        "target_column": "target",
        "n_search_iter": 5,
        "cv_folds": 3,
    })
    assert "TRAINING COMPLETE" in output, f"HGB regressor failed: {output[:300]}"
    r2_line = [l for l in output.split("\n") if "Train R2" in l]
    r2 = r2_line[0].split(": ")[1] if r2_line else "N/A"
    print(f"  [PASS] HistGradientBoostingRegressor: R²={r2}")
    _cleanup_model("test_hgb_reg")

    # With fixed hyperparameters
    print("\n  ── HGB with fixed hyperparameters ──")
    output = _run_skill({
        "estimator": "HistGradientBoostingClassifier",
        "model_name": "test_hgb_fixed",
        "train_dataset_ref": clf_ref,
        "target_column": "target",
        "auto_tune": False,
        "hyperparameters": {"max_iter": 50, "max_depth": 5, "learning_rate": 0.1},
    })
    assert "TRAINING COMPLETE" in output
    print(f"  [PASS] HGB with fixed params: max_iter=50, max_depth=5")
    _cleanup_model("test_hgb_fixed")

    print(f"\n  HistGradientBoosting: 3/3 passed")


# ═══════════════════════════════════════════════════════════════════════
# Test 9: NaN handling (SimpleImputer in pipeline)
# ═══════════════════════════════════════════════════════════════════════

def test_nan_handling():
    print("\n" + "=" * 70)
    print("TEST 9: NaN handling (SimpleImputer in pipeline)")
    print("=" * 70)

    ref = _make_nan_dataset(ref="test_nan_data")

    df = get_registered_dataset(ref)
    nan_count = df.isna().sum().sum()
    print(f"  Dataset has {nan_count} missing values")
    assert nan_count > 0, "Test data should have NaN values"

    estimators = [
        "LogisticRegression",
        "RandomForestClassifier",
        "HistGradientBoostingClassifier",
    ]

    results = {}
    for est in estimators:
        model_name = f"test_nan_{est.lower()}"
        print(f"\n  ── {est} with NaN data ──")

        output = _run_skill({
            "estimator": est,
            "model_name": model_name,
            "train_dataset_ref": ref,
            "target_column": "target",
            "auto_tune": False,
        })

        success = "TRAINING COMPLETE" in output
        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {est}: {'completed' if success else 'FAILED'}")
        if not success:
            print(f"  OUTPUT: {output[:300]}")
        results[est] = success
        _cleanup_model(model_name)

    passed = sum(results.values())
    print(f"\n  NaN handling: {passed}/{len(estimators)} passed")
    assert all(results.values()), f"Failed: {[k for k, v in results.items() if not v]}"


# ═══════════════════════════════════════════════════════════════════════
# Test 10: Auto-discovery of estimators
# ═══════════════════════════════════════════════════════════════════════

def test_estimator_discovery():
    print("\n" + "=" * 70)
    print("TEST 10: Auto-discovery of estimators")
    print("=" * 70)

    from agents.training.skills.sklearn_generic.train import ESTIMATORS, _discover_estimators

    catalog = _discover_estimators()
    print(f"  Discovered {len(catalog)} estimators")
    assert len(catalog) > 30, f"Expected >30 estimators, got {len(catalog)}"

    required_estimators = [
        "LogisticRegression", "RandomForestClassifier", "RandomForestRegressor",
        "GradientBoostingClassifier", "GradientBoostingRegressor",
        "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
        "SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor",
        "DecisionTreeClassifier", "DecisionTreeRegressor",
        "Ridge", "Lasso", "ElasticNet",
        "MLPClassifier", "MLPRegressor",
        "GaussianNB", "MultinomialNB",
        "AdaBoostClassifier", "AdaBoostRegressor",
        "BaggingClassifier", "BaggingRegressor",
        "ExtraTreesClassifier", "ExtraTreesRegressor",
        "SGDClassifier", "SGDRegressor",
        "LinearRegression",
    ]

    missing = [e for e in required_estimators if e not in catalog]
    assert not missing, f"Missing estimators: {missing}"
    print(f"  [PASS] All {len(required_estimators)} required estimators present")

    # Meta-estimators should be excluded
    meta_estimators = [
        "VotingClassifier", "VotingRegressor",
        "StackingClassifier", "StackingRegressor",
    ]
    present_meta = [e for e in meta_estimators if e in catalog]
    assert not present_meta, f"Meta-estimators should be excluded: {present_meta}"
    print(f"  [PASS] Meta-estimators correctly excluded")

    # Verify module_path format
    for name, (module, cls_name) in list(catalog.items())[:5]:
        assert module.startswith("sklearn."), f"Bad module for {name}: {module}"
        assert cls_name == name, f"Class name mismatch for {name}: {cls_name}"
    print(f"  [PASS] Module paths correctly formatted")

    # Verify ESTIMATORS matches at module level
    assert ESTIMATORS == catalog
    print(f"  [PASS] Module-level ESTIMATORS matches _discover_estimators()")

    print(f"\n  Estimator discovery: all checks passed")


# ═══════════════════════════════════════════════════════════════════════
# Test 11: Scoring metric selection
# ═══════════════════════════════════════════════════════════════════════

def test_scoring_selection():
    print("\n" + "=" * 70)
    print("TEST 11: Scoring metric selection (_select_scoring)")
    print("=" * 70)

    from agents.training.skills.sklearn_generic.train import _select_scoring

    # Balanced binary → accuracy
    y_balanced = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert _select_scoring(True, y_balanced) == "accuracy"
    print(f"  [PASS] Balanced binary → accuracy")

    # Imbalanced binary → roc_auc
    y_imbalanced = pd.Series([0] * 80 + [1] * 20)
    assert _select_scoring(True, y_imbalanced) == "roc_auc"
    print(f"  [PASS] Imbalanced binary (20% minority) → roc_auc")

    # Balanced multiclass → accuracy
    y_multi_balanced = pd.Series([0] * 33 + [1] * 33 + [2] * 34)
    assert _select_scoring(True, y_multi_balanced) == "accuracy"
    print(f"  [PASS] Balanced multiclass → accuracy")

    # Imbalanced multiclass → f1_weighted
    y_multi_imbalanced = pd.Series([0] * 80 + [1] * 10 + [2] * 10)
    assert _select_scoring(True, y_multi_imbalanced) == "f1_weighted"
    print(f"  [PASS] Imbalanced multiclass (10% minority) → f1_weighted")

    # Regression → r2
    y_reg = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert _select_scoring(False, y_reg) == "r2"
    print(f"  [PASS] Regression → r2")

    print(f"\n  Scoring selection: 5/5 passed")


# ═══════════════════════════════════════════════════════════════════════
# Test 12: scipy.stats distributions in search spaces
# ═══════════════════════════════════════════════════════════════════════

def test_search_space_distributions():
    print("\n" + "=" * 70)
    print("TEST 12: scipy.stats distributions in search spaces")
    print("=" * 70)

    from agents.training.skills.sklearn_generic.train import SEARCH_SPACES

    assert "LogisticRegression" in SEARCH_SPACES
    assert "HistGradientBoostingClassifier" in SEARCH_SPACES
    assert "HistGradientBoostingRegressor" in SEARCH_SPACES
    print(f"  [PASS] Key estimators have search spaces ({len(SEARCH_SPACES)} total)")

    from scipy.stats import rv_continuous, rv_discrete

    lr_space = SEARCH_SPACES["LogisticRegression"]
    c_dist = lr_space["C"]
    assert hasattr(c_dist, "rvs"), "C should be a scipy distribution"
    samples = c_dist.rvs(size=100, random_state=42)
    assert all(s > 0 for s in samples), "C samples should be positive"
    print(f"  [PASS] LogisticRegression.C: scipy distribution, samples in (0, inf)")

    hgb_space = SEARCH_SPACES["HistGradientBoostingClassifier"]
    for param in ["max_iter", "learning_rate", "max_depth", "min_samples_leaf", "l2_regularization"]:
        assert param in hgb_space, f"Missing {param} in HGB space"
    print(f"  [PASS] HistGradientBoostingClassifier: all expected params present")

    # Verify distributions sample correctly
    ref = _make_classification_dataset(n_samples=100, ref="test_dist")
    output = _run_skill({
        "estimator": "LogisticRegression",
        "model_name": "test_dist_lr",
        "train_dataset_ref": ref,
        "target_column": "target",
        "n_search_iter": 5,
        "cv_folds": 2,
    })
    assert "TRAINING COMPLETE" in output
    assert "Cross-validation" in output
    print(f"  [PASS] Training with scipy distributions works end-to-end")
    _cleanup_model("test_dist_lr")

    print(f"\n  Search space distributions: all checks passed")


# ═══════════════════════════════════════════════════════════════════════
# Test 13: Fixed params removed from search space
# ═══════════════════════════════════════════════════════════════════════

def test_fixed_params_excluded_from_search():
    print("\n" + "=" * 70)
    print("TEST 13: Fixed params excluded from search space")
    print("=" * 70)

    ref = _make_classification_dataset(ref="test_fixed_excl")

    # Fix "solver" and search over "C" — solver should not appear in search
    output = _run_skill({
        "estimator": "LogisticRegression",
        "model_name": "test_fixed_excl_lr",
        "train_dataset_ref": ref,
        "target_column": "target",
        "hyperparameters": {
            "solver": "saga",
            "C": [0.01, 1.0, 100.0],
        },
        "n_search_iter": 3,
        "cv_folds": 2,
    })
    assert "TRAINING COMPLETE" in output
    # The fixed param should be reported in hyperparameters
    assert "solver: saga" in output
    print(f"  [PASS] Fixed param 'solver=saga' preserved, not searched")
    _cleanup_model("test_fixed_excl_lr")


# ═══════════════════════════════════════════════════════════════════════
# Test 14: Parallelizable estimators get n_jobs=-1
# ═══════════════════════════════════════════════════════════════════════

def test_parallelizable_estimators():
    print("\n" + "=" * 70)
    print("TEST 14: Parallelizable estimators get n_jobs=-1")
    print("=" * 70)

    from agents.training.skills.sklearn_generic.train import _resolve_estimator, _PARALLELIZABLE

    for name in ["RandomForestClassifier", "ExtraTreesClassifier", "BaggingClassifier"]:
        assert name in _PARALLELIZABLE, f"{name} should be in _PARALLELIZABLE"
        est = _resolve_estimator(name)
        assert est.n_jobs == -1, f"{name}.n_jobs should be -1, got {est.n_jobs}"
        print(f"  [PASS] {name}: n_jobs={est.n_jobs}")

    # Non-parallelizable should not have n_jobs set
    lr = _resolve_estimator("LogisticRegression")
    assert not hasattr(lr, "n_jobs") or lr.n_jobs is None or lr.n_jobs == 1
    print(f"  [PASS] LogisticRegression: n_jobs not forced")

    print(f"\n  Parallelizable estimators: all checks passed")


# ═══════════════════════════════════════════════════════════════════════
# Test 15: Real data — insurance.csv regression (predict charges)
# ═══════════════════════════════════════════════════════════════════════

_INSURANCE_CSV = Path(__file__).parents[3] / "datasets" / "csv" / "insurance.csv"


def _load_insurance(ref="test_insurance"):
    """Load the real insurance.csv dataset."""
    df = pd.read_csv(_INSURANCE_CSV)
    register_dataset(ref, df)
    print(f"  Registered insurance dataset '{ref}': {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Dtypes: {dict(df.dtypes)}")
    return ref


def test_real_data_regression():
    print("\n" + "=" * 70)
    print("TEST 15: Real data — insurance.csv regression (predict charges)")
    print("=" * 70)

    if not _INSURANCE_CSV.exists():
        print("  [SKIP] insurance.csv not found")
        return

    ref = _load_insurance(ref="test_ins_reg")

    estimators = [
        ("Ridge", {}),
        ("RandomForestRegressor", {"n_estimators": 100}),
        ("HistGradientBoostingRegressor", {}),
        ("GradientBoostingRegressor", {"n_estimators": 100}),
    ]

    results = {}
    for est, fixed_hp in estimators:
        model_name = f"test_ins_{est.lower()}"
        print(f"\n  ── {est} on insurance.csv (charges) ──")
        t0 = time.time()

        output = _run_skill({
            "estimator": est,
            "model_name": model_name,
            "train_dataset_ref": ref,
            "target_column": "charges",
            "n_search_iter": 5,
            "cv_folds": 3,
            "hyperparameters": fixed_hp,
        })

        elapsed = time.time() - t0
        success = "TRAINING COMPLETE" in output
        r2_line = [l for l in output.split("\n") if "Train R2" in l]
        r2 = r2_line[0].split(": ")[1] if r2_line else "N/A"
        cv_line = [l for l in output.split("\n") if "Cross-validation" in l]
        cv = cv_line[0].split(": ")[-1] if cv_line else "N/A"
        scoring_line = [l for l in output.split("\n") if "Scoring" in l]
        scoring = scoring_line[0].split(": ")[1] if scoring_line else "N/A"

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {est}: R²={r2}, cv={cv}, scoring={scoring}, time={elapsed:.1f}s")
        if not success:
            print(f"  OUTPUT: {output[:300]}")

        results[est] = success
        _cleanup_model(model_name)

    passed = sum(results.values())
    print(f"\n  Real data regression: {passed}/{len(estimators)} passed")
    assert all(results.values()), f"Failed: {[k for k, v in results.items() if not v]}"


# ═══════════════════════════════════════════════════════════════════════
# Test 16: Real data — insurance.csv classification (predict smoker)
# ═══════════════════════════════════════════════════════════════════════

def test_real_data_classification():
    print("\n" + "=" * 70)
    print("TEST 16: Real data — insurance.csv classification (predict smoker)")
    print("=" * 70)

    if not _INSURANCE_CSV.exists():
        print("  [SKIP] insurance.csv not found")
        return

    df = pd.read_csv(_INSURANCE_CSV)
    df["smoker_binary"] = (df["smoker"] == "yes").astype(int)
    df = df.drop(columns=["smoker"])
    ref = "test_ins_clf"
    register_dataset(ref, df)
    smoker_counts = df["smoker_binary"].value_counts().to_dict()
    print(f"  Registered insurance classification '{ref}': {df.shape}")
    print(f"  Class distribution: {smoker_counts} (imbalanced: ~20% smokers)")

    estimators = [
        ("LogisticRegression", {"class_weight": "balanced"}),
        ("RandomForestClassifier", {"n_estimators": 100}),
        ("HistGradientBoostingClassifier", {}),
        ("SVC", {}),
    ]

    results = {}
    for est, fixed_hp in estimators:
        model_name = f"test_ins_clf_{est.lower()}"
        print(f"\n  ── {est} on insurance.csv (smoker) ──")
        t0 = time.time()

        output = _run_skill({
            "estimator": est,
            "model_name": model_name,
            "train_dataset_ref": ref,
            "target_column": "smoker_binary",
            "n_search_iter": 5,
            "cv_folds": 3,
            "hyperparameters": fixed_hp,
        })

        elapsed = time.time() - t0
        success = "TRAINING COMPLETE" in output
        acc_line = [l for l in output.split("\n") if "Train Accuracy" in l]
        acc = acc_line[0].split(": ")[1] if acc_line else "N/A"
        scoring_line = [l for l in output.split("\n") if "Scoring" in l]
        scoring = scoring_line[0].split(": ")[1] if scoring_line else "N/A"

        status = "PASS" if success else "FAIL"
        print(f"  [{status}] {est}: accuracy={acc}, scoring={scoring}, time={elapsed:.1f}s")
        if not success:
            print(f"  OUTPUT: {output[:300]}")

        results[est] = success
        _cleanup_model(model_name)

    passed = sum(results.values())
    print(f"\n  Real data classification: {passed}/{len(estimators)} passed")
    assert all(results.values()), f"Failed: {[k for k, v in results.items() if not v]}"


# ═══════════════════════════════════════════════════════════════════════
# Test 17: Real data — insurance.csv full predict pipeline
# ═══════════════════════════════════════════════════════════════════════

def test_real_data_predict_pipeline():
    print("\n" + "=" * 70)
    print("TEST 17: Real data — insurance.csv full predict pipeline")
    print("=" * 70)

    if not _INSURANCE_CSV.exists():
        print("  [SKIP] insurance.csv not found")
        return

    df = pd.read_csv(_INSURANCE_CSV)
    train_df = df.iloc[:1000]
    test_df = df.iloc[1000:]
    register_dataset("ins_pipe_train", train_df)
    register_dataset("ins_pipe_test", test_df)
    print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

    # Train a regression model
    output = _run_skill({
        "estimator": "HistGradientBoostingRegressor",
        "model_name": "ins_pipe_hgb",
        "train_dataset_ref": "ins_pipe_train",
        "target_column": "charges",
        "n_search_iter": 8,
        "cv_folds": 3,
    })
    assert "TRAINING COMPLETE" in output, f"Training failed: {output[:300]}"

    # Load model and predict on held-out test set
    model = load_model("ins_pipe_hgb")
    X_test = test_df.drop(columns=["charges"])
    y_test = test_df["charges"]
    preds = model.predict(X_test)

    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    print(f"  [PASS] Train complete, model loaded")
    print(f"  Test R²: {r2:.4f}")
    print(f"  Test MAE: ${mae:,.2f}")
    print(f"  Predictions: min=${preds.min():,.0f}, max=${preds.max():,.0f}, mean=${preds.mean():,.0f}")
    print(f"  Actual:      min=${y_test.min():,.0f}, max=${y_test.max():,.0f}, mean=${y_test.mean():,.0f}")

    assert r2 > 0.5, f"R² too low on real data: {r2:.4f}"
    assert mae < 10000, f"MAE too high: ${mae:,.2f}"
    print(f"  [PASS] Real data pipeline: R²={r2:.4f}, MAE=${mae:,.2f}")
    _cleanup_model("ins_pipe_hgb")


# ═══════════════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("SKLEARN GENERIC SKILL — TEST SUITE")
    print("=" * 70)

    tests = [
        ("Classification estimators", test_classification_estimators),
        ("Regression estimators", test_regression_estimators),
        ("Hyperparameter handling", test_hyperparameter_handling),
        ("Mixed data types", test_mixed_data),
        ("Model registration & predict", test_model_registration_and_predict),
        ("Skill registry integration", test_skill_registry),
        ("Error handling", test_error_handling),
        ("HistGradientBoosting", test_hist_gradient_boosting),
        ("NaN handling", test_nan_handling),
        ("Estimator discovery", test_estimator_discovery),
        ("Scoring selection", test_scoring_selection),
        ("Search space distributions", test_search_space_distributions),
        ("Fixed params excluded", test_fixed_params_excluded_from_search),
        ("Parallelizable estimators", test_parallelizable_estimators),
        ("Real data regression", test_real_data_regression),
        ("Real data classification", test_real_data_classification),
        ("Real data predict pipeline", test_real_data_predict_pipeline),
    ]

    t0 = time.time()
    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            failed.append((name, str(e)))
            import traceback
            print(f"\n  [FAIL] {name}:")
            traceback.print_exc()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"RESULTS: {len(passed)}/{len(tests)} passed in {elapsed:.1f}s")
    print("=" * 70)
    if passed:
        for name in passed:
            print(f"  [PASS] {name}")
    if failed:
        for name, err in failed:
            print(f"  [FAIL] {name}: {err}")
    print("=" * 70)

    sys.exit(1 if failed else 0)
