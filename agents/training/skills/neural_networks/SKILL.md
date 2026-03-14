---
name: neural_networks
description: Train PyTorch neural networks via dynamic code execution. The agent writes and runs training code in a sandboxed environment with infrastructure helpers.
---

# Neural Networks — PyTorch Code Execution Skill

This skill trains PyTorch neural networks through **dynamic code execution**. You write Python training code that runs in a sandbox with pre-loaded infrastructure helpers. Your code includes its own imports — you can use any installed library.

## Experiment Protocol

Follow these rules for every training session. They are adapted from [autoresearch](https://github.com/karpathy/autoresearch).

### 1. Baseline First

Your first run is **always** the default configuration with no modifications. This establishes the metric to beat. Do not skip this step. Do not "improve" the baseline before running it.

### 2. One Change at a Time

Each experiment after the baseline modifies **exactly one thing**: architecture OR hyperparameter OR training strategy. Never change multiple things simultaneously — you won't know what caused the improvement or regression.

### 3. Keep or Discard

After each run, compare the validation metric to the current best:

- **Metric improved?** → **KEEP**. This is the new baseline. All future experiments start from this config.
- **Equal or worse?** → **DISCARD**. Revert to the previous best config. Do not hedge.

Be binary and decisive. Do not average results or keep "promising" losers.

### 4. Build on Successes

Each new experiment starts from the **current best configuration**, not from scratch. Improvements compound.

### 5. Simplicity Criterion

All else equal, **simpler is better**. A tiny metric gain that adds ugly complexity is not worth it. Removing complexity for equal results IS a win. A 0.001 improvement from deleting code? Definitely keep.

### 6. Never Stop Early

If standard approaches plateau, think harder:
- Combine elements from previous near-misses
- Try radical architecture changes
- Try different optimizers or LR schedules
- Re-examine the data preprocessing

Do not ask whether to continue. Keep iterating until you are satisfied with the result or have exhausted reasonable ideas.

### 7. Crash Handling

If a run crashes with a trivial fix (typo, shape mismatch, wrong dtype), fix and re-run. If the idea is fundamentally broken (OOM on any reasonable batch size, numerical instability that can't be fixed), log "crash" as status and move on.

### 8. Structured Experiment Log

Every run must print a cumulative results summary. Track all experiments:

```
EXPERIMENT LOG:
model_name      | val_metric | status  | description
nn_baseline_v1  | 0.8234     | keep    | 2-layer (128,64), AdamW lr=1e-3
nn_dropout_v2   | 0.8456     | keep    | +Dropout(0.3), BatchNorm
nn_deeper_v3    | 0.8401     | discard | 4 layers, overfit
nn_gelu_v4      | 0.8512     | keep    | GELU activation, cosine LR
nn_tuned_v5     | 0.8589     | keep    | patience=20, best at epoch 47
```

This prevents repeating failed configurations and shows progress.

---

## Execution Environment

### How It Works

You write Python code and pass it via:

```python
train_with_skill(
    skill_name="neural_networks",
    params={
        "code": "... your Python training code ...",
        "train_dataset_ref": "...",
        "target_column": "...",
        "model_name": "nn_v1",
    }
)
```

The sandbox executes your code and returns everything you `print()`.

### Infrastructure Helpers (Pre-loaded)

These functions are available in your code without importing them:

| Helper | Signature | What it does |
|--------|-----------|-------------|
| `load_dataset` | `load_dataset(ref) → DataFrame` | Load a registered dataset by reference name |
| `encode_labels` | `encode_labels(y) → (encoded_y, n_classes)` | **For classification**: converts string labels to integer codes. The mapping is auto-saved with the model. **For regression**: returns values as-is. Use `n_classes > 10` to detect regression vs classification. |
| `save_model` | `save_model(model, name, metrics, preprocessor=..., feature_names=None, target_column=None, task_type="classification", n_classes=None)` | Save PyTorch model and register it. **ALWAYS pass the fitted `preprocessor` from `preprocess()`** — without it, the model cannot process raw data at inference. Custom nn.Module subclasses are supported. |
| `compute_metrics` | `compute_metrics(y_true, y_pred, y_proba=None, task_type="classification") → dict` | Compute standard metrics (accuracy/roc_auc or r2/rmse/mae). Pass integer-encoded y values. |
| `extract_params` | `extract_params(class_name, module_hint=None) → dict` | Discover constructor params for any class |
| `preprocess` | `preprocess(X_df, categorical_cols=None, preprocessor=None) → (np.ndarray, preprocessor)` | Fit/transform with sklearn ColumnTransformer. Pass fitted `preprocessor` for val/test. |
| `to_tensor` | `to_tensor(array) → torch.FloatTensor` | Convert numpy array to float tensor |

The `params` dict is also available with context about the training task.

### Your Code Includes Its Own Imports

The sandbox does NOT pre-import any framework. Your code starts with whatever imports it needs:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
```

This means you can use **any installed library** without changes to the sandbox.

### Data Pipeline (CRITICAL)

This is **tabular data**, not images or text. The data may already be feature-engineered (one-hot encoded, scaled) — check the sample data to decide if `preprocess()` is needed. Follow this exact pattern:

```python
df = load_dataset(params["train_dataset_ref"])       # pandas DataFrame
target_col = params["target_column"]
X_df = df.drop(columns=[target_col])                  # DataFrame without target
y_raw = df[target_col].values                         # may be strings!

# For classification: encode_labels converts string labels to ints
# For regression: encode_labels returns values as-is (n_classes = n_unique)
y_encoded, n_classes = encode_labels(y_raw)
is_regression = n_classes > 10  # heuristic: many unique values → regression

# Preprocess features (handles scaling + one-hot encoding)
cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
X_np, preprocessor = preprocess(X_df, categorical_cols=cat_cols)

# Convert to tensors
X_tensor = to_tensor(X_np)
y_tensor = torch.tensor(y_encoded, dtype=torch.long)  # for classification
```

For train/val split, use `sklearn.model_selection.train_test_split` on the numpy arrays AFTER preprocessing:

```python
X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_encoded, test_size=0.2, random_state=42)
X_tr_t, X_val_t = to_tensor(X_tr), to_tensor(X_val)
```

### Batching Strategy

- **< 50,000 rows**: Use full-batch training (load everything into tensors, no DataLoader needed).
- **50,000 - 500,000 rows**: Use mini-batch training with `TensorDataset` + `DataLoader`. Mini-batch SGD generalizes better and reduces memory usage.
- **> 500,000 rows**: Always use mini-batch. Consider batch size 1024-4096.

Mini-batch pattern (for datasets > 50K rows):

```python
from torch.utils.data import TensorDataset, DataLoader

X_tr_t, X_val_t = to_tensor(X_tr), to_tensor(X_val)
y_tr_t = torch.tensor(y_tr, dtype=torch.long)

train_ds = TensorDataset(X_tr_t, y_tr_t)
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)

for epoch in range(200):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)  # full-batch eval is fine
        val_loss = criterion(val_out, y_val_t).item()
```

**IMPORTANT**: Only use `TensorDataset` + `DataLoader` on **already-converted tensors**, never on DataFrames.

### Common Mistakes (AVOID THESE)

- **DO NOT** call `save_model(...)` without `preprocessor=preprocessor`. The model WILL produce wrong predictions at inference without it. This is the #1 source of bugs.
- **DO NOT** manually map labels with `df[col].map({...})` — use `encode_labels(y)` instead. It handles strings, stores the mapping, and attaches it to the saved model automatically.
- **DO NOT** use `torch.tensor(y, dtype=torch.long)` on string arrays — encode first with `encode_labels()`.
- **DO NOT** use `DataLoader` or `default_collate` directly on DataFrames — convert to tensors first.
- **DO NOT** try to concatenate DataFrames with torch collate functions.
- **DO NOT** skip the `preprocess()` call — it handles scaling and one-hot encoding.
- **DO NOT** forget to pass the fitted `preprocessor` when processing validation/test data.
- **DO NOT** pass raw DataFrames to the model — always convert to tensors first.
- **You CAN define custom `nn.Module` subclasses** — the sandbox handles pickling correctly via cloudpickle.

---

## Dynamic Discovery

Before writing training code, use `extract_params()` to discover available parameters:

```python
# What does nn.Linear accept?
print(extract_params("Linear", module_hint="torch.nn"))

# What does AdamW accept? (lr, weight_decay, betas, eps, amsgrad, ...)
print(extract_params("AdamW", module_hint="torch.optim"))

# What does ReduceLROnPlateau accept?
print(extract_params("ReduceLROnPlateau", module_hint="torch.optim.lr_scheduler"))

# What does BatchNorm1d accept?
print(extract_params("BatchNorm1d", module_hint="torch.nn"))
```

Use this to inform hyperparameter choices rather than guessing.

---

## Code Templates

### Template 1: Basic Training

```python
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

df = load_dataset(params["train_dataset_ref"])
target_col = params["target_column"]
model_name = params.get("model_name", "nn_baseline_v1")

X_df = df.drop(columns=[target_col])
y_raw = df[target_col].values

y_encoded, n_classes = encode_labels(y_raw)
is_regression = n_classes > 10

cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
X_np, preprocessor = preprocess(X_df, categorical_cols=cat_cols)
n_features = X_np.shape[1]

X_train = to_tensor(X_np)
if is_regression:
    y_train = torch.tensor(y_raw.astype(np.float32), dtype=torch.float32).unsqueeze(1)
    criterion = nn.MSELoss()
    out_dim = 1
else:
    y_train = torch.tensor(y_encoded, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()
    out_dim = n_classes

model = nn.Sequential(
    nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(64, out_dim),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    preds = model(X_train)
    if is_regression:
        y_pred = preds.squeeze().numpy()
        metrics = compute_metrics(y_encoded, y_pred, task_type="regression")
    else:
        y_pred = preds.argmax(dim=1).numpy()
        y_proba = torch.softmax(preds, dim=1).numpy()
        metrics = compute_metrics(y_encoded, y_pred, y_proba=y_proba, task_type="classification")

print(f"Training metrics: {metrics}")
save_model(model, model_name, metrics, preprocessor=preprocessor,
           feature_names=X_df.columns.tolist(), target_column=target_col,
           task_type="regression" if is_regression else "classification",
           n_classes=n_classes if not is_regression else None)
print(f"Model saved: {model_name}")
```

### Template 2: Epoch-by-Epoch with Early Stopping

```python
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

df = load_dataset(params["train_dataset_ref"])
target_col = params["target_column"]
model_name = params.get("model_name", "nn_earlystop_v1")

X_df = df.drop(columns=[target_col])
y_raw = df[target_col].values

y_encoded, n_classes = encode_labels(y_raw)
is_regression = n_classes > 10

cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
X_np, preprocessor = preprocess(X_df, categorical_cols=cat_cols)

y_target = y_raw.astype(np.float32) if is_regression else y_encoded
X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_target, test_size=0.2, random_state=42)
X_tr, X_val = to_tensor(X_tr), to_tensor(X_val)

if is_regression:
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    criterion = nn.MSELoss()
else:
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

n_features = X_tr.shape[1]
model = nn.Sequential(
    nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(64, 1 if is_regression else n_classes),
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

best_val_loss = float("inf")
best_state = None
patience, patience_counter = 20, 0

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(X_tr)
    loss = criterion(out, y_tr_t)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val)
        val_loss = criterion(val_out, y_val_t).item()

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: train_loss={loss.item():.4f} val_loss={val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    val_preds = model(X_val)
    if is_regression:
        y_pred = val_preds.squeeze().numpy()
        metrics = compute_metrics(y_val, y_pred, task_type="regression")
    else:
        y_pred = val_preds.argmax(dim=1).numpy()
        y_proba = torch.softmax(val_preds, dim=1).numpy()
        metrics = compute_metrics(y_val, y_pred, y_proba=y_proba, task_type="classification")

print(f"Validation metrics: {metrics}")
save_model(model, model_name, metrics, preprocessor=preprocessor,
           feature_names=X_df.columns.tolist(), target_column=target_col,
           task_type="regression" if is_regression else "classification",
           n_classes=n_classes if not is_regression else None)
```

### Template 3: Architecture Search

```python
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

df = load_dataset(params["train_dataset_ref"])
target_col = params["target_column"]

X_df = df.drop(columns=[target_col])
y_raw = df[target_col].values

y_encoded, n_classes = encode_labels(y_raw)
is_regression = n_classes > 10

cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
X_np, preprocessor = preprocess(X_df, categorical_cols=cat_cols)
n_features = X_np.shape[1]

X_tr, X_val, y_tr, y_val = train_test_split(X_np, y_encoded, test_size=0.2, random_state=42)
X_tr_t, X_val_t = to_tensor(X_tr), to_tensor(X_val)
if is_regression:
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    criterion = nn.MSELoss()
else:
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    criterion = nn.CrossEntropyLoss()

architectures = [
    ("shallow", [64]),
    ("medium", [128, 64]),
    ("wide", [256, 128]),
    ("deep", [128, 64, 32]),
    ("wide_deep", [256, 128, 64]),
]

results = []
for arch_name, hidden_sizes in architectures:
    layers = []
    prev = n_features
    for h in hidden_sizes:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
        prev = h
    layers.append(nn.Linear(prev, 1 if is_regression else n_classes))
    model = nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_out = model(X_val_t)
        if is_regression:
            y_pred = val_out.squeeze().numpy()
            m = compute_metrics(y_val, y_pred, task_type="regression")
            score = m.get("r2", 0)
        else:
            y_pred = val_out.argmax(dim=1).numpy()
            y_proba = torch.softmax(val_out, dim=1).numpy()
            m = compute_metrics(y_val, y_pred, y_proba=y_proba, task_type="classification")
            score = m.get("roc_auc", m.get("accuracy", 0))

    results.append((arch_name, hidden_sizes, score, model, m))
    print(f"  {arch_name} {hidden_sizes}: score={score:.4f}")

results.sort(key=lambda x: x[2], reverse=True)
best_name, best_hidden, best_score, best_model, best_metrics = results[0]
print(f"\nBest architecture: {best_name} {best_hidden} (score={best_score:.4f})")

model_name = params.get("model_name", f"nn_arch_{best_name}")
save_model(best_model, model_name, best_metrics, preprocessor=preprocessor,
           feature_names=X_df.columns.tolist(), target_column=target_col,
           task_type="regression" if is_regression else "classification")
```

---

## Architecture Decision Guide

### Dataset Size Heuristics

| Rows | Recommended Architecture |
|------|-------------------------|
| < 1,000 | 1-2 layers, 32-64 units, high dropout (0.3-0.5) |
| 1,000 - 10,000 | 2-3 layers, 64-128 units, moderate dropout (0.2-0.3) |
| 10,000 - 100,000 | 2-4 layers, 128-256 units, BatchNorm + dropout (0.1-0.3) |
| > 100,000 | 3-5 layers, 256-512 units, BatchNorm, lower dropout |

### When to Use Each Component

- **Dropout**: Always, unless dataset is very large. Start at 0.2-0.3.
- **BatchNorm**: When using 2+ hidden layers. Stabilizes training, allows higher LR.
- **Residual connections**: When using 4+ layers. Helps gradient flow.
- **Gradient clipping** (`clip_grad_norm_`): Always recommended. Set to 1.0.

### Activation Selection

- **ReLU**: Default. Simple, fast, works well.
- **GELU**: Modern alternative. Smoother. Often slightly better than ReLU.
- **SiLU (Swish)**: Good for deeper networks. Smooth, non-monotonic.
- **Tanh**: Only for normalized data in [-1, 1] range.

### Optimizer Selection

- **AdamW**: Default. Weight-decoupled L2 regularization. Works well out of the box.
- **SGD + momentum**: For fine-grained control. Useful when AdamW overfits.
- **Adam**: Legacy. Prefer AdamW for better generalization.

---

## Hyperparameter Strategy Matrix

### Diagnosis → Action

| Symptom | Diagnosis | Actions |
|---------|-----------|---------|
| Train metric high, val metric low | Overfitting | Increase dropout, increase weight_decay, reduce layer width/depth, add BatchNorm |
| Both metrics low | Underfitting | Increase layer width/depth, reduce dropout, reduce weight_decay, train longer |
| Loss goes to NaN or explodes | Divergence | Halve learning rate, add gradient clipping, reduce batch size, check data for NaN |
| Val metric flat for many epochs | Plateau | Try cosine LR schedule, switch optimizer, change activation, try different architecture |
| Train loss oscillates wildly | LR too high | Reduce LR by 2-5x, or use LR warmup |
| Very slow convergence | LR too low | Increase LR by 2-5x, or use OneCycleLR |

### What NOT to Do

- Do not change multiple hyperparameters at once
- Do not add complexity without evidence it helps
- Do not skip the baseline
- Do not ignore the experiment log
- Do not train without a validation set
