"""
Unit tests for all new helpers and improvements.
No LLM calls — runs quickly.
"""

import sys
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
import agent as _a

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "agents") not in sys.path:
    sys.path.append(str(ROOT / "agents"))

from agents.training.core.state import create_initial_state
from agents.training.steps.data_collection import (
    _infer_target_column,
    _score_dataset,
    _signal_strength,
    _validate,
    data_collection,
)
from agents.training.steps.label_and_split import _detect_target_transform
from agents.training.agent_simple import SYSTEM_PROMPT, _infer_task_type

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} — {detail}")


print("=" * 70)
print("UNIT TESTS — ALL NEW IMPROVEMENTS")
print("=" * 70)

# ─── 1. _infer_target_column ───────────────────────────────────────
print("\n── _infer_target_column ──")
df1 = pd.DataFrame({"tenure": [1], "churn": [0], "price": [100]})
check("goal-word match 'churn'",
      _infer_target_column(df1, "predict customer churn") == "churn")
check("goal-word match 'price'",
      _infer_target_column(df1, "predict house price") == "price")
check("keyword fallback (churn)",
      _infer_target_column(df1, "classify subscribers") == "churn")
check("returns None when nothing matches",
      _infer_target_column(pd.DataFrame({"a": [1], "b": [2]}), "predict xyz") is None)
check("'target' keyword hits",
      _infer_target_column(pd.DataFrame({"x": [1], "target": [2]}), "something") == "target")
check("'label' keyword hits",
      _infer_target_column(pd.DataFrame({"x": [1], "label": [2]}), "something") == "label")

# ─── 2. _signal_strength ──────────────────────────────────────────
print("\n── _signal_strength ──")
np.random.seed(42)
n = 200
df_strong = pd.DataFrame({
    "f1": np.arange(n),
    "f2": np.arange(n) * 2 + np.random.normal(0, 5, n),
    "target": np.arange(n) * 3 + np.random.normal(0, 10, n),
})
sig_strong = _signal_strength(df_strong, "target")
check("strong signal > 0.5", sig_strong > 0.5, f"got {sig_strong:.3f}")

df_weak = pd.DataFrame({
    "f1": np.random.rand(n),
    "f2": np.random.rand(n),
    "target": np.random.rand(n),
})
sig_weak = _signal_strength(df_weak, "target")
check("random signal < 0.2", sig_weak < 0.2, f"got {sig_weak:.3f}")
check("missing column → 0", _signal_strength(df_weak, "nope") == 0.0)
check("non-numeric target → 0",
      _signal_strength(pd.DataFrame({"f": range(50), "t": ["a", "b"] * 25}), "t") == 0.0)
check("empty df → 0",
      _signal_strength(pd.DataFrame(), "t") == 0.0)

# ─── 3. _score_dataset ────────────────────────────────────────────
print("\n── _score_dataset ──")
_u.register_dataset("u_strong", df_strong)
_u.register_dataset("u_weak", df_weak)
s1 = _score_dataset("u_strong", "predict target")
s2 = _score_dataset("u_weak", "predict target")
check("strong > weak (same size)",
      s1 > s2, f"strong={s1:.0f}, weak={s2:.0f}")
check("missing ref → -1", _score_dataset("nonexistent_xyz_123") == -1.0)

# String-ID columns excluded
df_ids = pd.DataFrame({
    "customer_id": [f"c_{i}" for i in range(100)],
    "churn": [0] * 80 + [1] * 20,
    "tenure": range(100),
})
_u.register_dataset("u_ids", df_ids)
s_ids = _score_dataset("u_ids", "predict churn")
check("string-ID excluded (positive score)", s_ids > 0, f"got {s_ids:.0f}")

# Constant columns penalised
df_const = pd.DataFrame({"a": [5] * 50, "b": range(50), "target": range(50)})
_u.register_dataset("u_const", df_const)
df_noconst = pd.DataFrame({"a": range(50), "b": range(50), "target": range(50)})
_u.register_dataset("u_noconst", df_noconst)
check("constant col penalised",
      _score_dataset("u_noconst", "predict target") > _score_dataset("u_const", "predict target"),
      f"{_score_dataset('u_noconst', 'predict target'):.0f} vs {_score_dataset('u_const', 'predict target'):.0f}")

# Bigger dataset scores higher
df_big = pd.DataFrame({"a": range(500), "b": range(500), "target": range(500)})
_u.register_dataset("u_big", df_big)
check("bigger > smaller (same quality)",
      _score_dataset("u_big", "predict target") > _score_dataset("u_noconst", "predict target"))

# ─── 4. _validate ─────────────────────────────────────────────────
print("\n── _validate ──")
_u.register_dataset("v_good", pd.DataFrame({"a": range(50), "b": range(50)}))
check("valid dataset → empty", _validate("v_good") == [])
_u.register_dataset("v_small", pd.DataFrame({"a": [1]}))
check("too few rows flagged", len(_validate("v_small")) > 0)
check("missing ref flagged", len(_validate("nonexistent_xyz")) > 0)
_u.register_dataset("v_dup", pd.DataFrame({"a": [1] * 20, "b": [2] * 20}))
check("all-identical rows flagged", len(_validate("v_dup")) > 0)
_u.register_dataset("v_null", pd.DataFrame({
    "a": range(20), "b": [None] * 20,
}))
check("all-null column flagged", len(_validate("v_null")) > 0)

# ─── 5. _detect_target_transform ──────────────────────────────────
print("\n── _detect_target_transform ──")
df_skew = pd.DataFrame({
    "price": np.random.lognormal(12, 1, 200),
    "sqft": range(200),
})
check("lognormal positive → log1p",
      _detect_target_transform(df_skew, "price", "predict house price") == "log1p")
df_normal = pd.DataFrame({"val": np.random.normal(100, 10, 200), "x": range(200)})
check("normal → None",
      _detect_target_transform(df_normal, "val", "predict value") is None)
check("classification → None",
      _detect_target_transform(df_skew, "price", "classify fraud") is None)
df_neg = pd.DataFrame({"y": np.random.normal(0, 100, 200), "x": range(200)})
check("negative values → None",
      _detect_target_transform(df_neg, "y", "predict outcome") is None)
df_small = pd.DataFrame({"y": [1, 2, 3, 4, 5], "x": [1, 2, 3, 4, 5]})
check("tiny dataset → None (too few rows)",
      _detect_target_transform(df_small, "y", "predict y") is None)

# ─── 6. _infer_task_type ──────────────────────────────────────────
print("\n── _infer_task_type ──")
check("default → classification",
      _infer_task_type("predict churn", "supervised") == "classification")
check("price → regression",
      _infer_task_type("predict house price", "supervised") == "regression")
check("regression model → regression",
      _infer_task_type("predict outcome", "regression") == "regression")
check("logistic regression → classification",
      _infer_task_type("predict churn", "logistic regression") == "classification")
check("salary → regression",
      _infer_task_type("predict employee salary", "supervised") == "regression")
check("cost → regression",
      _infer_task_type("estimate delivery cost", "supervised") == "regression")
check("survival → classification",
      _infer_task_type("predict survival", "supervised") == "classification")

# ─── 7. System Prompt checks ─────────────────────────────────────
print("\n── System Prompt ──")
check("data_collection redo option present",
      "Unsuitable dataset" in SYSTEM_PROMPT)
check("no curator leakage",
      "curator" not in SYSTEM_PROMPT.lower())
check("4-iteration limit",
      "4 total training iterations" in SYSTEM_PROMPT)
check("all 9 steps present",
      all(s in SYSTEM_PROMPT for s in [
          "data_collection", "select_model", "cleaning",
          "label_split_definition", "feature_selection_specification",
          "feature_engineering_executor", "training_approval",
          "training", "generate_report"]))

# ─── 8. data_collection integration ──────────────────────────────
print("\n── data_collection (mocked) ──")

# Local wins if stronger
ref_l, ref_e = "dc_local", "dc_ext"
_u.register_dataset(ref_l, df_strong.copy())
_u.register_dataset(ref_e, df_weak.copy())
mock_local = _a.DataRetrievalResult(
    dataset_ref=ref_l, rows=200, columns=list(df_strong.columns),
    source="sql_warehouse", description="strong",
)
state = create_initial_state("predict target", use_external_sources=True)
with mock.patch("agents.training.steps.data_collection.retrieve_data", return_value=mock_local), \
     mock.patch("agents.training.steps.data_collection._try_curator",
                return_value=(ref_e, "kaggle", {"step": "data_collection", "action": "test"})):
    r = data_collection(state)
check("strong local beats weak external",
      r["collected_dataset_ref"] == ref_l, f"got {r['collected_dataset_ref']}")
check("data_source is 'local'", r["data_source"] == "local")

# External wins if stronger
_u.register_dataset("dc_ext2", df_strong.copy())
_u.register_dataset("dc_local2", df_weak.copy())
mock_local2 = _a.DataRetrievalResult(
    dataset_ref="dc_local2", rows=200, columns=list(df_weak.columns),
    source="sql_warehouse", description="weak",
)
state2 = create_initial_state("predict target", use_external_sources=True)
with mock.patch("agents.training.steps.data_collection.retrieve_data", return_value=mock_local2), \
     mock.patch("agents.training.steps.data_collection._try_curator",
                return_value=("dc_ext2", "kaggle", {"step": "data_collection", "action": "test"})):
    r2 = data_collection(state2)
check("weak local loses to strong external",
      r2["collected_dataset_ref"] == "dc_ext2", f"got {r2['collected_dataset_ref']}")
check("data_source is 'kaggle'", r2["data_source"] == "kaggle")

# External disabled → curator never called
state3 = create_initial_state("predict target", use_external_sources=False)
with mock.patch("agents.training.steps.data_collection.retrieve_data", return_value=mock_local), \
     mock.patch("agents.training.steps.data_collection._try_curator") as m_c:
    _ = data_collection(state3)
    check("external=False → curator never called", not m_c.called)

# No goal → error
check("no-goal → error",
      data_collection(create_initial_state(""))["error"] is not None)

# Both fail → graceful error
state4 = create_initial_state("predict target", use_external_sources=True)
with mock.patch("agents.training.steps.data_collection.retrieve_data",
                return_value={"error": "no data"}), \
     mock.patch("agents.training.steps.data_collection._try_curator",
                return_value=(None, "", {"step": "data_collection", "action": "fail"})):
    r4 = data_collection(state4)
    check("both fail → graceful error",
          r4["collected_dataset_ref"] is None and r4["error"] is not None)

# Linked dataset shortcut
_u.register_dataset("linked_ds", pd.DataFrame({"a": range(50), "b": range(50)}))
state5 = create_initial_state("predict target")
state5["linked_datasets"] = ["linked_ds"]
with mock.patch("agents.training.steps.data_collection.retrieve_data") as m_r:
    r5 = data_collection(state5)
    check("linked dataset skips retrieval",
          r5["collected_dataset_ref"] == "linked_ds" and not m_r.called)

# ─── 9. Quality gates (logic check) ──────────────────────────────
print("\n── Quality gates (logic) ──")
def simulate_gate(task_type, val_r2=None, val_roc_auc=None, val_accuracy=None, iteration=1):
    """Replicate the quality gate logic from agent_simple.py."""
    feature_redo_requested = False
    result = {
        "success": True,
        "val_r2": val_r2,
        "val_roc_auc": val_roc_auc,
        "val_accuracy": val_accuracy,
    }
    if task_type == "regression":
        if val_r2 is not None and val_r2 < 0.05 and iteration <= 2:
            feature_redo_requested = True
    else:
        if val_roc_auc is not None and val_roc_auc < 0.55 and iteration <= 2:
            feature_redo_requested = True
        elif val_accuracy is not None and val_accuracy < 0.55 and val_roc_auc is None and iteration <= 2:
            feature_redo_requested = True
    return feature_redo_requested

check("regression R²<0.05 → redo",
      simulate_gate("regression", val_r2=0.01))
check("regression R²>0.05 → no redo",
      not simulate_gate("regression", val_r2=0.3))
check("regression R²<0.05 but iter 3 → no redo",
      not simulate_gate("regression", val_r2=0.01, iteration=3))
check("classification ROC<0.55 → redo",
      simulate_gate("classification", val_roc_auc=0.50))
check("classification ROC>0.55 → no redo",
      not simulate_gate("classification", val_roc_auc=0.75))
check("classification acc<0.55, no ROC → redo",
      simulate_gate("classification", val_accuracy=0.45))
check("classification acc<0.55 but ROC present → ROC wins",
      not simulate_gate("classification", val_roc_auc=0.70, val_accuracy=0.45))

# ─── Done ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TOTAL: {passed} passed, {failed} failed")
print(f"{'='*70}")
exit(0 if failed == 0 else 1)
