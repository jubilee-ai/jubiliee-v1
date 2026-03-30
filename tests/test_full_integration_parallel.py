"""
Full integration test: Loan_default.csv (255k rows, 18 cols) through the
complete agent_simple pipeline WITH the new parallel features:
  - Feature experiment grid (parallel scouts)
  - batch_train_with_skill (parallel training)
  - experiment_result/feature_rankings passed to training agent

This test exercises the REAL pipeline (LLM calls, real sklearn training).
Run: python3 tests/test_full_integration_parallel.py 2>&1 | tee /tmp/integration.log
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "data-tools"))
sys.path.insert(0, str(ROOT / "tools" / "models-tools" / "training"))

import pandas as pd
from utils import clear_registry, register_dataset


def run_full_integration():
    t0 = time.time()
    print("=" * 80)
    print("FULL INTEGRATION TEST: Loan Default (255k rows)")
    print("=" * 80)

    clear_registry()

    # Load a 5k sample for speed while keeping enough rows for meaningful training
    print("\n[1] Loading dataset...")
    df = pd.read_csv(ROOT / "datasets" / "csv" / "Loan_default.csv", nrows=5000)
    register_dataset("loan_default_raw", df, register_sql=False)
    print(f"    Registered 'loan_default_raw': {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Default rate: {df['Default'].mean():.1%}")
    print(f"    Dtypes: {dict(df.dtypes.value_counts())}")

    print("\n[2] Creating agent...")
    from agents.training.agent_simple import create_simple_training_agent

    agent, state = create_simple_training_agent(
        goal="Predict loan default risk. Target column is 'Default'. Use the dataset 'loan_default_raw'.",
        linked_datasets=["loan_default_raw"],
        user_model_preference="supervised",
        model="openai:gpt-5.1",
        hitl=False,
    )
    print("    Agent created.")

    print("\n[3] Invoking agent (full LLM pipeline)...")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": (
            "Train a model to predict loan default. The dataset 'loan_default_raw' is already loaded "
            "with 5000 rows and 18 columns. Target column is 'Default' (binary 0/1). "
            "Columns include Age, Income, LoanAmount, CreditScore, MonthsEmployed, "
            "NumCreditLines, InterestRate, LoanTerm, DTIRatio, Education, EmploymentType, "
            "MaritalStatus, HasMortgage, HasDependents, LoanPurpose, HasCoSigner."
        )}]}
    )
    elapsed = time.time() - t0

    print(f"\n[4] Agent finished in {elapsed:.1f}s")

    # --- Verify state ---
    print("\n" + "=" * 80)
    print("STATE INSPECTION")
    print("=" * 80)

    checks = []

    # Feature spec
    fs = state.get("feature_spec") or {}
    n_features = len(fs.get("features", []))
    print(f"\n  feature_spec: {n_features} features")
    checks.append(("feature_spec has ≥5 features", n_features >= 5))

    # Experiment result
    exp = state.get("experiment_result")
    print(f"  experiment_result: {exp}")
    checks.append(("experiment_result exists", exp is not None))
    if exp:
        checks.append(("total_scouts > 0", exp.get("total_scouts", 0) > 0))
        checks.append(("best_variant_name set", bool(exp.get("best_variant_name"))))
        print(f"    best_variant: {exp.get('best_variant_name')}")
        print(f"    total_scouts: {exp.get('total_scouts')}")
        print(f"    best_metric: {exp.get('best_metric')}")
        print(f"    signal_features: {exp.get('signal_features', [])[:5]}")
        print(f"    dropped_features: {exp.get('dropped_features', [])[:5]}")
        print(f"    wall_time: {exp.get('wall_time_seconds')}s")

    # Feature rankings
    rankings = state.get("feature_rankings")
    print(f"  feature_rankings: {dict(list((rankings or {}).items())[:5])}")
    checks.append(("feature_rankings exists", rankings is not None and len(rankings) > 0))

    # Experiment grid summary
    grid = state.get("experiment_grid_summary")
    print(f"  experiment_grid_summary: {len(grid or [])} entries")

    # Training metrics
    metrics = state.get("training_metrics") or {}
    print(f"\n  training_metrics:")
    print(f"    success: {metrics.get('success')}")
    print(f"    model_name: {metrics.get('model_name')}")
    print(f"    num_iterations: {metrics.get('num_iterations')}")
    print(f"    val_accuracy: {metrics.get('val_accuracy')}")
    print(f"    val_roc_auc: {metrics.get('val_roc_auc')}")
    print(f"    test_accuracy: {metrics.get('test_accuracy')}")
    print(f"    test_roc_auc: {metrics.get('test_roc_auc')}")
    checks.append(("training succeeded", metrics.get("success") is True))
    checks.append(("val_roc_auc > 0.5", (metrics.get("val_roc_auc") or 0) > 0.5))

    # Iterations
    iters = metrics.get("iterations", [])
    print(f"    iterations ({len(iters)}):")
    for it in iters:
        name = it.get("model_name", "?")
        ok = it.get("success", False)
        tool = it.get("tool", it.get("tool_used", "?"))
        hp = it.get("hyperparams", {})
        hp_short = ", ".join(f"{k}={v}" for k, v in list(hp.items())[:3])
        roc = it.get("roc_auc") or it.get("val_roc_auc")
        acc = it.get("val_accuracy")
        print(f"      [{('OK' if ok else 'FAIL')}] {name} ({tool}) — roc_auc={roc}, acc={acc}, hp={hp_short}")
    checks.append(("≥1 training iteration", len(iters) >= 1))

    # Feature importances
    fi = metrics.get("feature_importances", {})
    print(f"    feature_importances: {dict(list(fi.items())[:5])}")

    # Audit trace
    audit = state.get("audit_trace", [])
    step_names = [a.get("step") for a in audit]
    print(f"\n  audit_trace steps: {step_names}")
    checks.append(("experiment_runner in audit", "feature_experiment_runner" in step_names))

    # Report
    report_path = state.get("report_path")
    print(f"  report_path: {report_path}")
    if report_path and Path(report_path).exists():
        with open(report_path) as f:
            report = json.load(f)
        has_exp = report.get("feature_experiment") is not None
        print(f"  report has feature_experiment: {has_exp}")
        checks.append(("report includes feature_experiment", has_exp))
        # Check report has regression metrics too
        tm = report.get("training_results", {})
        val_m = tm.get("validation_metrics", {})
        test_m = tm.get("test_metrics", {})
        print(f"  report validation_metrics: {val_m}")
        print(f"  report test_metrics: {test_m}")
    else:
        checks.append(("report file exists", False))

    # --- Summary ---
    print("\n" + "=" * 80)
    print("CHECKLIST")
    print("=" * 80)
    all_ok = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_ok = False
        print(f"  [{status}] {label}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Result: {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 80)

    return all_ok, state, elapsed


if __name__ == "__main__":
    ok, state, elapsed = run_full_integration()
    sys.exit(0 if ok else 1)
