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
        # Verify categorical columns were detected and encoded
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

    # Verify model is in registry
    models = [m for m in list_models() if m["model_name"] == "test_predict_rf"]
    assert len(models) == 1, f"Model not found in registry"
    info = models[0]
    print(f"  Registry entry: model_type={info['model_type']}, features={len(info['feature_names'])}")
    assert info["model_type"] == "sklearn_RandomForestClassifier"
    assert info["target_column"] == "target"
    assert len(info["feature_names"]) > 0

    # Load and predict
    model = load_model("test_predict_rf")
    df = get_registered_dataset(ref)
    X = df.drop(columns=["target"])
    preds = model.predict(X)
    print(f"  Predictions: shape={preds.shape}, unique={np.unique(preds)}")
    assert len(preds) == len(df)

    # Predict proba (RF supports it)
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
        get_all_skill_names,
    )

    # Discovery
    skills = _discover_skills()
    assert "sklearn_generic" in skills
    print(f"  [PASS] Discovery: {len(skills)} skill(s) found, sklearn_generic present")

    # Prompt loading
    prompt = _load_skill_prompt("sklearn_generic")
    assert "estimator" in prompt.lower()
    assert "GradientBoostingClassifier" in prompt
    print(f"  [PASS] SKILL.md loaded: {len(prompt)} chars")

    # Alias map
    alias_map = build_alias_map()
    test_aliases = ["svm", "knn", "gradient boosting", "mlp", "ridge", "logistic regression", "random forest"]
    for alias in test_aliases:
        assert alias in alias_map, f"Alias '{alias}' not found"
        assert alias_map[alias] == "sklearn_generic", f"Alias '{alias}' -> {alias_map[alias]}"
    print(f"  [PASS] Alias map: {len(alias_map)} aliases, all route to sklearn_generic")

    # Execution through registry
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

    # Missing estimator
    output = _run_skill({"model_name": "x", "train_dataset_ref": ref, "target_column": "target"})
    assert "TRAINING FAILED" in output and "'estimator' is required" in output
    print(f"  [PASS] Missing estimator: caught")

    # Unknown estimator
    output = _run_skill({"estimator": "FakeModel", "model_name": "x", "train_dataset_ref": ref, "target_column": "target"})
    assert "TRAINING FAILED" in output and "Unknown estimator" in output
    print(f"  [PASS] Unknown estimator: caught")

    # Missing dataset
    output = _run_skill({"estimator": "SVC", "model_name": "x", "train_dataset_ref": "nonexistent", "target_column": "target"})
    assert "TRAINING FAILED" in output and "not found" in output
    print(f"  [PASS] Missing dataset: caught")

    # Missing target column
    output = _run_skill({"estimator": "SVC", "model_name": "x", "train_dataset_ref": ref, "target_column": "nonexistent"})
    assert "TRAINING FAILED" in output and "not in columns" in output
    print(f"  [PASS] Missing target column: caught")

    # Missing required param
    output = _run_skill({"estimator": "SVC", "model_name": "x"})
    assert "TRAINING FAILED" in output
    print(f"  [PASS] Missing train_dataset_ref: caught")

    print(f"\n  Error handling: 5/5 passed")


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
    ]

    t0 = time.time()
    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except Exception as e:
            failed.append((name, str(e)))
            print(f"\n  [FAIL] {name}: {e}")

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
