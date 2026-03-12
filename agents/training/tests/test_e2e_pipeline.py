"""
E2E pipeline tests for agent_simple.py.

Runs 3 diverse scenarios through the full pipeline with real LLM calls.
Uses external sources (Kaggle/HuggingFace) to test the complete flow.
"""

import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
for p in [str(ROOT / "tools" / "data-tools"), str(ROOT / "agents" / "data-retrieval")]:
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
import utils as _u

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "agents") not in sys.path:
    sys.path.append(str(ROOT / "agents"))


def run_e2e_test(label, goal, use_external=True):
    from agents.training.agent_simple import create_simple_training_agent

    print(f"\n{'*'*70}")
    print(f"E2E TEST: {label}")
    print(f"Goal: {goal}")
    print(f"External sources: {use_external}")
    print(f"{'*'*70}")

    agent, state = create_simple_training_agent(
        goal=goal, hitl=False, use_external_sources=use_external)

    start = time.time()
    result = agent.invoke({"messages": [{"role": "user", "content": goal}]})
    elapsed = time.time() - start

    msgs = result.get("messages", [])
    tool_calls = []
    for m in msgs:
        if hasattr(m, "tool_calls") and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls.append(tc.get("name", "?"))

    print(f"\n--- Tool Sequence ({len(tool_calls)} calls) ---")
    for i, tc in enumerate(tool_calls, 1):
        print(f"  {i}. {tc}")
    print(f"\nRuntime: {elapsed:.0f}s")

    metrics = state.get("training_metrics") or {}
    label_def = state.get("label_definition") or {}
    ds_ref = state.get("collected_dataset_ref")
    ds_source = state.get("data_source")
    target_transform = label_def.get("target_transform")

    ds_df = _u.get_registered_dataset(ds_ref) if ds_ref else None
    ds_rows = len(ds_df) if ds_df is not None else "?"
    ds_cols = len(ds_df.columns) if ds_df is not None else "?"
    ds_null_pct = f"{ds_df.isnull().sum().sum() / max(ds_df.size, 1) * 100:.1f}%" if ds_df is not None else "?"

    print(f"\n--- Dataset ---")
    print(f"  Ref: {ds_ref}")
    print(f"  Source: {ds_source}")
    print(f"  Shape: {ds_rows} rows × {ds_cols} cols")
    print(f"  Null %: {ds_null_pct}")
    if ds_df is not None:
        print(f"  Columns: {list(ds_df.columns[:15])}{'...' if len(ds_df.columns) > 15 else ''}")

    print(f"\n--- Label / Split ---")
    print(f"  Target: {label_def.get('target_column', '?')}")
    print(f"  Transform: {target_transform}")
    print(f"  Split: {label_def.get('split_strategy', '?')}")

    features = (state.get("feature_spec") or {}).get("features", [])
    print(f"\n--- Features ---")
    print(f"  Planned: {len(features)}")
    if features:
        print(f"  Sample: {[f.get('name','?') for f in features[:5]]}")

    print(f"\n--- Training Results ---")
    print(f"  Success: {metrics.get('success')}")
    print(f"  Model: {metrics.get('model_type', '?')}")
    print(f"  Model name: {metrics.get('model_name', '?')}")
    print(f"  Iterations: {metrics.get('num_iterations', 0)}")

    for key, lbl in [
        ("val_accuracy", "Val Accuracy"), ("val_roc_auc", "Val ROC-AUC"),
        ("test_accuracy", "Test Accuracy"), ("test_roc_auc", "Test ROC-AUC"),
        ("val_r2", "Val R²"), ("test_r2", "Test R²"),
        ("val_rmse", "Val RMSE"), ("test_rmse", "Test RMSE"),
        ("val_mae", "Val MAE"), ("test_mae", "Test MAE"),
    ]:
        v = metrics.get(key)
        if v is not None:
            print(f"  {lbl:18s} {v:.4f}")

    feat_imp = metrics.get("feature_importances", {})
    if feat_imp:
        sorted_imp = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n--- Feature Importances (top 5 of {len(feat_imp)}) ---")
        for name, imp in sorted_imp[:5]:
            print(f"  {name:30s} {imp:.4f}")
        near_zero = [n for n, v in sorted_imp if abs(v) < 0.01]
        if near_zero:
            print(f"  Near-zero features: {near_zero[:5]}")

    print(f"\n--- Optimization Flags ---")
    print(f"  Feature redo requested: {metrics.get('feature_redo_requested', False)}")
    print(f"  Feature redo reason: {state.get('feature_redo_reason', 'N/A')}")
    print(f"  Training iteration: {state.get('training_iteration', 0)}")
    print(f"  Report path: {state.get('report_path', 'N/A')}")

    audit = state.get("audit_trace", [])
    print(f"\n--- Audit Trace ({len(audit)} entries) ---")
    for a in audit:
        step = a.get("step", "?")
        action = a.get("action", a.get("model_name", ""))
        print(f"  [{step}] {action}")

    if msgs:
        final = msgs[-1].content if hasattr(msgs[-1], "content") else str(msgs[-1])
        print(f"\n--- Agent Final Message (first 500 chars) ---")
        print(f"  {final[:500]}")

    return {
        "label": label,
        "rows": ds_rows,
        "cols": ds_cols,
        "null_pct": ds_null_pct,
        "target": label_def.get("target_column", "?"),
        "target_transform": target_transform,
        "n_features": len(features),
        "model": metrics.get("model_type", "?"),
        "success": metrics.get("success"),
        "n_tools": len(tool_calls),
        "elapsed": elapsed,
        "val_roc": metrics.get("val_roc_auc"),
        "val_r2": metrics.get("val_r2"),
        "val_acc": metrics.get("val_accuracy"),
        "test_roc": metrics.get("test_roc_auc"),
        "test_r2": metrics.get("test_r2"),
        "test_acc": metrics.get("test_accuracy"),
        "feature_redo": metrics.get("feature_redo_requested", False),
        "feat_imp_count": len(feat_imp),
        "training_iter": state.get("training_iteration", 0),
    }


def print_summary(results):
    print(f"\n\n{'#'*70}")
    print(f"OVERALL E2E RESULTS SUMMARY")
    print(f"{'#'*70}")
    print(f"{'Task':<40} {'Rows':>7} {'Feats':>5} {'Model':>14} {'Key Metric':>16} {'Iters':>5} {'Time':>5}")
    print("-" * 100)
    for r in results:
        metric = ""
        if r["val_roc"] is not None:
            metric = f"ROC={r['val_roc']:.4f}"
        elif r["val_r2"] is not None:
            metric = f"R²={r['val_r2']:.4f}"
        elif r["val_acc"] is not None:
            metric = f"Acc={r['val_acc']:.4f}"
        model = str(r["model"])[:14]
        print(f"{r['label']:<40} {str(r['rows']):>7} {r['n_features']:>5} {model:>14} {metric:>16} {r['training_iter']:>5} {r['elapsed']:>4.0f}s")

    print(f"\n--- Improvement Features Verification ---")
    for r in results:
        checks = []
        if r["success"]:
            checks.append("TRAINED")
        else:
            checks.append("FAILED")
        if r["feat_imp_count"] > 0:
            checks.append(f"importances({r['feat_imp_count']})")
        if r["feature_redo"]:
            checks.append("quality_gate_triggered")
        if r["target_transform"]:
            checks.append(f"transform={r['target_transform']}")
        if r["training_iter"] > 1:
            checks.append(f"re-trained({r['training_iter']}x)")
        print(f"  {r['label']}: {', '.join(checks)}")

    all_ok = all(r["success"] for r in results)
    print(f"\n{'='*70}")
    print(f"VERDICT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    tests = [
        {
            "label": "1. Telecom churn (classification)",
            "goal": "Predict customer churn based on telecom account features like tenure, monthly charges, and contract type",
        },
        {
            "label": "2. House price (regression)",
            "goal": "Predict house sale prices based on features like square footage, bedrooms, bathrooms, lot size, year built, and neighborhood",
        },
        {
            "label": "3. Heart disease (classification)",
            "goal": "Predict whether a patient has heart disease based on clinical features like age, cholesterol, blood pressure, chest pain type, and max heart rate",
        },
    ]

    all_results = []
    for test in tests:
        try:
            r = run_e2e_test(test["label"], test["goal"])
            all_results.append(r)
        except Exception as e:
            print(f"\n[ERROR] Test '{test['label']}' crashed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "label": test["label"], "rows": "?", "cols": "?", "null_pct": "?",
                "target": "?", "target_transform": None, "n_features": 0,
                "model": "?", "success": False, "n_tools": 0, "elapsed": 0,
                "val_roc": None, "val_r2": None, "val_acc": None,
                "test_roc": None, "test_r2": None, "test_acc": None,
                "feature_redo": False, "feat_imp_count": 0, "training_iter": 0,
            })

    print_summary(all_results)
