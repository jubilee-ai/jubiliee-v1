"""Neural network training skill — minimal code-execution sandbox.

The agent writes full Python training code (including its own imports).
The sandbox provides only infrastructure helpers and the params dict.
"""

import importlib.util
import io
import re
import sys
import traceback
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

_ROOT = Path(__file__).parents[4]
for _p in [
    str(_ROOT / "tools" / "models-tools" / "training"),
    str(_ROOT / "tools" / "data-tools"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model_storage import generate_model_path, register_model
from pytorch_predictor import PyTorchPredictor
from utils import get_registered_dataset


# ── Infrastructure helpers exposed to the sandbox ────────────────────────

def _make_load_dataset():
    def load_dataset(ref: str) -> pd.DataFrame:
        df = get_registered_dataset(ref)
        if df is None:
            raise ValueError(f"Dataset '{ref}' not found in registry.")
        return df
    return load_dataset


def _make_save_model(label_encoder_ref: list):
    """Create save_model helper. label_encoder_ref is a mutable list holding the label encoder."""
    def save_model(
        model,
        name: str = None,
        metrics: dict = None,
        preprocessor=None,
        feature_names=None,
        target_column=None,
        task_type="classification",
        n_classes=None,
        description=None,
        hyperparameters=None,
        training_samples=None,
        encoder=None,
        *,
        model_name: str = None,
    ):
        resolved_name = name or model_name or "nn_model"
        if metrics is None:
            metrics = {}

        if preprocessor is None and task_type != "unsupervised":
            print(
                "[save_model] WARNING: preprocessor is None. "
                "Pass the fitted preprocessor from preprocess() to save_model()."
            )
        le = label_encoder_ref[0] if label_encoder_ref else None
        wrapper = PyTorchPredictor(
            model=model,
            preprocessor=preprocessor,
            task_type=task_type,
            n_classes=n_classes,
            label_encoder=le,
            encoder=encoder,
        )

        save_path = generate_model_path(resolved_name)
        with open(save_path, "wb") as f:
            cloudpickle.dump(wrapper, f)

        classes = []
        if n_classes and task_type not in ("regression", "unsupervised"):
            if le is not None:
                classes = [str(c) for c in le.classes_]
            else:
                classes = [str(i) for i in range(n_classes)]

        register_model(
            model_name=resolved_name,
            model_path=save_path,
            model_type="pytorch_nn",
            description=description or f"PyTorch neural network: {resolved_name}",
            metrics=metrics,
            feature_names=feature_names or [],
            target_column=target_column or "",
            hyperparameters=hyperparameters or {},
            training_samples=training_samples or 0,
            classes=classes,
        )
        print(f"MODEL REGISTERED: {resolved_name}")
        return resolved_name
    return save_model


def _make_compute_metrics():
    def compute_metrics(y_true, y_pred, y_proba=None, task_type="classification") -> dict:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        result = {}
        if task_type == "regression":
            result["r2"] = float(r2_score(y_true, y_pred))
            result["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            result["mae"] = float(mean_absolute_error(y_true, y_pred))
        else:
            result["accuracy"] = float(accuracy_score(y_true, y_pred))
            if y_proba is not None:
                try:
                    y_proba = np.asarray(y_proba)
                    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        result["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                    elif y_proba.ndim == 2:
                        result["roc_auc"] = float(
                            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                        )
                except (ValueError, TypeError):
                    pass
        return result
    return compute_metrics


def _make_compute_unsupervised_metrics():
    def compute_unsupervised_metrics(
        X, labels=None, reconstruction=None, original=None
    ) -> dict:
        """Compute unsupervised metrics for clustering and autoencoders.

        Args:
            X: Input data (numpy array or torch tensor).
            labels: Cluster assignments (for clustering metrics).
            reconstruction: Reconstructed output from autoencoder.
            original: Original input for reconstruction loss comparison.

        Returns dict with available metrics: silhouette_score, davies_bouldin,
        reconstruction_loss.
        """
        X_np = np.asarray(X) if not isinstance(X, np.ndarray) else X
        result = {}

        if labels is not None:
            labels_np = np.asarray(labels)
            n_labels = len(set(labels_np))
            if 2 <= n_labels < len(labels_np):
                try:
                    result["silhouette_score"] = float(silhouette_score(X_np, labels_np))
                except ValueError:
                    pass
                try:
                    result["davies_bouldin"] = float(davies_bouldin_score(X_np, labels_np))
                except ValueError:
                    pass

        if reconstruction is not None and original is not None:
            recon_np = np.asarray(reconstruction)
            orig_np = np.asarray(original)
            result["reconstruction_loss"] = float(np.mean((orig_np - recon_np) ** 2))

        return result
    return compute_unsupervised_metrics


def _make_extract_params():
    _script = Path(__file__).parent.parent / "scripts" / "extract_params.py"
    spec = importlib.util.spec_from_file_location("extract_params", str(_script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def extract_params(class_name: str, module_hint: str | None = None) -> dict:
        return mod.extract_estimator_params(class_name, module_hint=module_hint)
    return extract_params


def _make_encode_labels(label_encoder_ref: list):
    """Create encode_labels helper that stores the fitted LabelEncoder for save_model."""
    def encode_labels(y=None):
        """Encode labels to 0-based contiguous integer codes.

        Returns (encoded_y, n_classes). The label mapping is automatically
        stored and attached to the saved model so predictions are decoded
        back to original labels at inference time.

        For continuous float targets (regression), passes through unchanged.
        For unsupervised tasks, call with y=None to get (None, 0).

        Usage:
            y_encoded, n_classes = encode_labels(y)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long)
        """
        if y is None:
            return None, 0

        y_arr = np.asarray(y)
        unique_vals = np.unique(y_arr[~pd.isna(y_arr)])
        n_classes = len(unique_vals)

        # Continuous float targets with many unique values → regression, pass through
        if np.issubdtype(y_arr.dtype, np.floating) and n_classes > 20:
            print(f"[encode_labels] Regression target ({n_classes} unique values) — passing through")
            return y_arr, n_classes

        # Already 0-based contiguous ints (0,1,...,n-1)
        if np.issubdtype(y_arr.dtype, np.integer):
            int_vals = sorted(unique_vals.astype(int))
            if int_vals == list(range(n_classes)):
                return y_arr.astype(int), n_classes

        # Otherwise use LabelEncoder to remap to 0..n-1
        le = LabelEncoder()
        encoded = le.fit_transform(y_arr)
        label_encoder_ref.clear()
        label_encoder_ref.append(le)
        n_classes = len(le.classes_)
        if n_classes <= 20:
            print(f"[encode_labels] Encoded {n_classes} classes: {dict(zip(le.classes_, range(n_classes)))}")
        else:
            preview = list(le.classes_[:5])
            print(f"[encode_labels] Encoded {n_classes} classes (first 5: {preview})")
        return encoded, n_classes
    return encode_labels


def _make_preprocess():
    def preprocess(X_df, categorical_cols=None, preprocessor=None):
        """Fit/transform data using sklearn ColumnTransformer.

        Returns (X_array, preprocessor). For val/test data, pass the
        fitted preprocessor from the training call.

        If the data is already fully numeric (no object/category columns),
        this just scales the features. Pass preprocessor=None to fit new,
        or pass a fitted preprocessor for val/test.
        """
        if categorical_cols is None:
            categorical_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = [c for c in X_df.columns if c not in categorical_cols]

        if preprocessor is not None:
            return preprocessor.transform(X_df).astype(np.float32), preprocessor

        transformers = []
        if numeric_cols:
            transformers.append(("num", SklearnPipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric_cols))
        if categorical_cols:
            transformers.append(("cat", SklearnPipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols))

        ct = ColumnTransformer(transformers=transformers, remainder="passthrough")
        X_out = ct.fit_transform(X_df).astype(np.float32)
        return X_out, ct
    return preprocess


def _make_to_tensor():
    def to_tensor(array):
        import torch
        return torch.tensor(np.asarray(array, dtype=np.float32), dtype=torch.float32)
    return to_tensor


# ── Default fallback training ────────────────────────────────────────────

_DEFAULT_CODE = '''
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

df = load_dataset(params["train_dataset_ref"])
target_col = params["target_column"]
model_name = params.get("model_name", "nn_default_v1")

X_df = df.drop(columns=[target_col])
y_raw = df[target_col].values

y_encoded, n_classes = encode_labels(y_raw)
is_regression = n_classes > 10

cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
X_np, preprocessor = preprocess(X_df, categorical_cols=cat_cols)
n_features = X_np.shape[1]

y_target = y_raw.astype(np.float32) if is_regression else y_encoded
X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_target, test_size=0.2, random_state=42)
X_tr_t, X_val_t = to_tensor(X_tr), to_tensor(X_val)

if is_regression:
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    criterion = nn.MSELoss()
    out_dim = 1
else:
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    out_dim = n_classes

model = nn.Sequential(
    nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, out_dim),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

best_val_loss = float("inf")
best_state = None
patience_counter = 0

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(X_tr_t)
    loss = criterion(out, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        val_loss = criterion(val_out, y_val_t).item()
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= 20:
        print(f"Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}: train_loss={loss.item():.4f} val_loss={val_loss:.4f}")

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    val_preds = model(X_val_t)
    if is_regression:
        y_pred = val_preds.squeeze().numpy()
        metrics = compute_metrics(y_val, y_pred, task_type="regression")
    else:
        y_pred = val_preds.argmax(dim=1).numpy()
        y_proba = torch.softmax(val_preds, dim=1).numpy()
        metrics = compute_metrics(y_val, y_pred, y_proba=y_proba, task_type="classification")

task_type = "regression" if is_regression else "classification"
print(f"Validation metrics: {metrics}")
save_model(model, name=model_name, metrics=metrics, preprocessor=preprocessor,
           feature_names=X_df.columns.tolist(), target_column=target_col,
           task_type=task_type, n_classes=n_classes if not is_regression else None,
           training_samples=len(X_tr))
'''


# ── Output filtering ─────────────────────────────────────────────────────

_METRIC_KEYWORDS = {
    "metric", "r2", "rmse", "mae", "accuracy", "roc_auc", "f1",
    "auc", "precision", "recall", "loss",
    "silhouette", "davies_bouldin", "inertia", "reconstruction",
}
_KEEP_KEYWORDS = {
    "error", "traceback", "exception", "warning",
    "model registered", "model saved", "save_model",
    "early stopping", "encode_labels",
    "experiment", "keep", "discard", "baseline",
}


_EPOCH_RE = re.compile(r"epoch\s+(\d+)", re.IGNORECASE)


def _filter_sandbox_output(raw: str) -> str:
    """Return only metrics, errors, and key training events from sandbox output."""
    lines = raw.split("\n")
    kept: list[str] = []
    in_error_block = False

    for line in lines:
        lower = line.lower().strip()
        if not lower:
            if in_error_block:
                kept.append(line)
            continue

        if "execution error" in lower or "traceback" in lower:
            in_error_block = True
            kept.append(line)
            continue
        if in_error_block:
            kept.append(line)
            continue

        if any(kw in lower for kw in _METRIC_KEYWORDS | _KEEP_KEYWORDS):
            kept.append(line)
            continue

        if lower.startswith("epoch "):
            m = _EPOCH_RE.match(lower)
            if m:
                epoch_num = int(m.group(1))
                if epoch_num <= 1 or epoch_num % 50 == 0:
                    kept.append(line)
            continue

    result = "\n".join(kept).strip()
    if not result:
        result = "(Training code ran but produced no metric output.)"
    return result


# ── Main entry point ─────────────────────────────────────────────────────

def run(params: dict) -> str:
    """Execute agent-written training code in a minimal sandbox.

    The sandbox provides only infrastructure helpers and the params dict.
    The agent's code includes its own imports (torch, numpy, etc.).
    """
    code = params.get("code")
    if not code:
        code = _DEFAULT_CODE

    label_encoder_ref: list = []

    sandbox_globals = {
        "load_dataset": _make_load_dataset(),
        "save_model": _make_save_model(label_encoder_ref),
        "compute_metrics": _make_compute_metrics(),
        "compute_unsupervised_metrics": _make_compute_unsupervised_metrics(),
        "extract_params": _make_extract_params(),
        "encode_labels": _make_encode_labels(label_encoder_ref),
        "preprocess": _make_preprocess(),
        "to_tensor": _make_to_tensor(),
        "params": params,
        "__builtins__": __builtins__,
    }

    captured = io.StringIO()
    old_stdout = sys.stdout
    try:
        sys.stdout = captured
        exec(code, sandbox_globals)
    except Exception:
        tb = traceback.format_exc()
        captured.write(f"\n\nEXECUTION ERROR:\n{tb}")
    finally:
        sys.stdout = old_stdout

    raw_output = captured.getvalue()
    if not raw_output.strip():
        return "(Code executed successfully but produced no output.)"

    return _filter_sandbox_output(raw_output)
