"""
Unit tests for new features: task_type refactor, unsupervised NN support,
PyTorchPredictor.transform(), compute_unsupervised_metrics.

No LLM calls — runs quickly. Avoids deep import chains.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
for p in [
    str(ROOT / "tools" / "data-tools"),
    str(ROOT / "tools" / "models-tools" / "training"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


# ─── task_type: _derive_task_type (no deep imports) ─────────────────

print("--- task_type: _derive_task_type ---")

from agents.training.steps.select_model import _derive_task_type

check("unsupervised → unsupervised",
      _derive_task_type("unsupervised", "segment customers") == "unsupervised")
check("supervised + price → regression",
      _derive_task_type("supervised", "predict price") == "regression")
check("supervised + churn → classification",
      _derive_task_type("supervised", "predict churn") == "classification")
check("neural_networks + forecast → regression",
      _derive_task_type("neural_networks", "forecast revenue") == "regression")
check("neural_networks + classify → classification",
      _derive_task_type("neural_networks", "classify images") == "classification")
check("supervised + salary → regression",
      _derive_task_type("supervised", "predict salary") == "regression")
check("unsupervised ignores goal keywords",
      _derive_task_type("unsupervised", "predict price") == "unsupervised")


# ─── task_type: create_initial_state ────────────────────────────────

print("\n--- task_type: create_initial_state ---")

from agents.training.core.state import create_initial_state

state = create_initial_state("cluster customers", user_model_preference="unsupervised")
check("has task_type key", "task_type" in state)
check("task_type is None initially", state["task_type"] is None)
check("has selected_model key", "selected_model" in state)
check("use_external_sources defaults to False", state["use_external_sources"] is False)

state2 = create_initial_state("train", use_external_sources=True)
check("use_external_sources can be set to True", state2["use_external_sources"] is True)


# ─── compute_unsupervised_metrics ──────────────────────────────────

print("\n--- compute_unsupervised_metrics ---")

from agents.training.skills.neural_networks.train import _make_compute_unsupervised_metrics

compute_unsup = _make_compute_unsupervised_metrics()

X_cluster = np.array([[0, 0], [0, 1], [1, 0], [10, 10], [10, 11], [11, 10]])
labels_cluster = np.array([0, 0, 0, 1, 1, 1])
m = compute_unsup(X_cluster, labels=labels_cluster)
check("silhouette_score present", "silhouette_score" in m)
check("silhouette_score > 0.5", m.get("silhouette_score", 0) > 0.5,
      f"got {m.get('silhouette_score')}")
check("davies_bouldin present", "davies_bouldin" in m)
check("davies_bouldin > 0", m.get("davies_bouldin", 0) > 0)

X_recon = np.array([[1.0, 2.0], [3.0, 4.0]])
X_original = np.array([[1.1, 2.1], [3.1, 4.1]])
m2 = compute_unsup(X_cluster, reconstruction=X_recon, original=X_original)
check("reconstruction_loss present", "reconstruction_loss" in m2)
check("reconstruction_loss < 0.05", m2.get("reconstruction_loss", 1) < 0.05,
      f"got {m2.get('reconstruction_loss')}")

m3 = compute_unsup(X_cluster)
check("no metrics without labels or reconstruction", len(m3) == 0)

single_label = np.array([0, 0, 0, 0])
m4 = compute_unsup(X_cluster[:4], labels=single_label)
check("single-label → no silhouette", "silhouette_score" not in m4)

both = compute_unsup(X_cluster, labels=labels_cluster, reconstruction=X_recon, original=X_original)
check("both clustering + reconstruction metrics returned",
      "silhouette_score" in both and "reconstruction_loss" in both)


# ─── encode_labels no-op ──────────────────────────────────────────

print("\n--- encode_labels no-op ---")

from agents.training.skills.neural_networks.train import _make_encode_labels

le_ref = []
encode_labels = _make_encode_labels(le_ref)

y_none, n_classes_none = encode_labels(None)
check("encode_labels(None) → None", y_none is None)
check("encode_labels(None) → 0 classes", n_classes_none == 0)

le_ref2 = []
el2 = _make_encode_labels(le_ref2)
y_str, n_cls = el2(np.array(["cat", "dog", "cat", "bird"]))
check("encode_labels with strings → encoded", y_str is not None)
check("3 unique strings → 3 classes", n_cls == 3)
check("label encoder stored", len(le_ref2) == 1)

le_ref3 = []
el3 = _make_encode_labels(le_ref3)
y_int, n_cls_int = el3(np.array([0, 1, 2, 0, 1]))
check("0-based ints pass through", np.array_equal(y_int, [0, 1, 2, 0, 1]))
check("3 unique ints → 3 classes", n_cls_int == 3)


# ─── PyTorchPredictor transform ──────────────────────────────────

print("\n--- PyTorchPredictor ---")

try:
    import torch
    import torch.nn as nn
    from pytorch_predictor import PyTorchPredictor

    encoder = nn.Linear(4, 2)
    decoder = nn.Linear(2, 4)
    full_model = nn.Sequential(encoder, decoder)

    pred = PyTorchPredictor(
        model=full_model, preprocessor=None,
        task_type="unsupervised", encoder=encoder,
    )

    X_test = np.random.randn(5, 4).astype(np.float32)

    emb = pred.transform(X_test)
    check("transform shape (5, 2)", emb.shape == (5, 2), f"got {emb.shape}")

    recon = pred.reconstruct(X_test)
    check("reconstruct shape (5, 4)", recon.shape == (5, 4), f"got {recon.shape}")

    out = pred.predict(X_test)
    check("predict returns ndarray", isinstance(out, np.ndarray))
    check("predict shape (5, 4) for unsupervised", out.shape == (5, 4), f"got {out.shape}")

    pred_no_enc = PyTorchPredictor(
        model=full_model, preprocessor=None, task_type="unsupervised",
    )
    emb2 = pred_no_enc.transform(X_test)
    check("transform w/o encoder → full model shape (5, 4)", emb2.shape == (5, 4))

    pred_cls = PyTorchPredictor(
        model=nn.Sequential(nn.Linear(4, 3)), preprocessor=None,
        task_type="classification", n_classes=3,
    )
    cls_pred = pred_cls.predict(X_test)
    check("classification predict → int labels",
          cls_pred.dtype in (np.int64, np.int32))

    pred_reg = PyTorchPredictor(
        model=nn.Sequential(nn.Linear(4, 1)), preprocessor=None,
        task_type="regression",
    )
    reg_pred = pred_reg.predict(X_test)
    check("regression predict → float values",
          np.issubdtype(reg_pred.dtype, np.floating))

    check("classes_ returns array for classification",
          pred_cls.classes_ is not None and len(pred_cls.classes_) == 3)
    check("classes_ returns None for unsupervised",
          pred.classes_ is None)

except ImportError as e:
    print(f"  (skipping PyTorchPredictor tests — {e})")


# ─── TrainingIteration / TrainingResult unsupervised fields ──────────

print("\n--- TrainingIteration / TrainingResult ---")

from agents.training.steps.training import (
    TrainingIteration,
    TrainingResult,
    _primary_metric,
    _find_best_iteration,
    _format_best_metric,
)

it_unsup = TrainingIteration(
    model_name="km_v1", tool_used="unsupervised/KMeans",
    success=True, silhouette_score=0.65, davies_bouldin=0.8,
)
check("silhouette_score=0.65", it_unsup.silhouette_score == 0.65)
check("davies_bouldin=0.8", it_unsup.davies_bouldin == 0.8)
check("_primary_metric unsupervised → 0.65", _primary_metric(it_unsup, "unsupervised") == 0.65)

it_no_sil = TrainingIteration(
    model_name="ae_v1", tool_used="neural_networks/AE",
    success=True, reconstruction_loss=0.01,
)
check("_primary_metric unsupervised no silhouette → -inf",
      _primary_metric(it_no_sil, "unsupervised") == -float("inf"))

it_cls = TrainingIteration(
    model_name="rf_v1", tool_used="supervised/RF",
    success=True, val_roc_auc=0.85, val_accuracy=0.80,
)
check("_primary_metric classification → ROC-AUC", _primary_metric(it_cls, "classification") == 0.85)

it_reg = TrainingIteration(
    model_name="gbm_v1", tool_used="supervised/GBM",
    success=True, val_r2=0.92,
)
check("_primary_metric regression → val_r2", _primary_metric(it_reg, "regression") == 0.92)

iters = [
    {"model_name": "km1", "success": True, "silhouette_score": 0.5, "davies_bouldin": 1.2},
    {"model_name": "km2", "success": True, "silhouette_score": 0.7, "davies_bouldin": 0.8},
    {"model_name": "km3", "success": False},
]
best = _find_best_iteration(iters, "unsupervised")
check("_find_best_iteration → km2 (highest silhouette)",
      best is not None and best["model_name"] == "km2")

iters_tied = [
    {"model_name": "a", "success": True, "silhouette_score": 0.5, "davies_bouldin": 1.0},
    {"model_name": "b", "success": True, "silhouette_score": 0.5, "davies_bouldin": 0.5},
]
best_tied = _find_best_iteration(iters_tied, "unsupervised")
check("_find_best_iteration tiebreaks on lower davies_bouldin → b",
      best_tied is not None and best_tied["model_name"] == "b")

iters_cls = [
    {"model_name": "a", "success": True, "val_roc_auc": 0.80},
    {"model_name": "b", "success": True, "val_roc_auc": 0.90},
]
check("_find_best_iteration classification → b",
      _find_best_iteration(iters_cls, "classification")["model_name"] == "b")

fmt = _format_best_metric({"silhouette_score": 0.65, "davies_bouldin": 0.8}, "unsupervised")
check("format unsupervised has Silhouette", "Silhouette" in fmt)
check("format unsupervised has Davies-Bouldin", "Davies-Bouldin" in fmt)

fmt_reg = _format_best_metric({"val_r2": 0.92}, "regression")
check("format regression has R²", "R²" in fmt_reg)

tr = TrainingResult(
    success=True, best_model_name="km_best", model_type="KMeans",
    num_iterations=3, summary="done",
    silhouette_score=0.7, davies_bouldin=0.8, inertia=150.0,
)
check("TrainingResult.silhouette_score", tr.silhouette_score == 0.7)
check("TrainingResult.davies_bouldin", tr.davies_bouldin == 0.8)
check("TrainingResult.inertia", tr.inertia == 150.0)
check("TrainingResult.reconstruction_loss default None", tr.reconstruction_loss is None)


# ─── Done ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"TOTAL: {passed} passed, {failed} failed")
print(f"{'='*70}")
exit(0 if failed == 0 else 1)
