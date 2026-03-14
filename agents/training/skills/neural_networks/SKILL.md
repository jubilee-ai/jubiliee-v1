---
name: neural_networks
description: Train PyTorch neural networks via dynamic code execution. The agent writes and runs training code in a sandboxed environment with infrastructure helpers.
---

# Neural Networks — PyTorch Code Execution Skill

You write Python training code that runs in a sandbox with pre-loaded helpers. Your code includes its own imports — any installed library is available (torch, sklearn, optuna, numpy, scipy, etc.).

## Execution

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

### Pre-loaded Helpers

| Helper | Signature | Purpose |
|--------|-----------|---------|
| `load_dataset` | `(ref) → DataFrame` | Load dataset by reference |
| `encode_labels` | `(y) → (encoded_y, n_classes)` | Encode string labels to ints (classification) or pass-through (regression). Mapping auto-saved with model. |
| `preprocess` | `(X_df, categorical_cols=None, preprocessor=None) → (np.ndarray, preprocessor)` | Fit/transform with ColumnTransformer. Pass fitted preprocessor for val/test. |
| `save_model` | `(model, name=..., metrics=..., preprocessor=..., feature_names=None, target_column=None, task_type=..., n_classes=None) → name` | Save and register model. Returns the model name. **Always pass `preprocessor`**. Custom `nn.Module` subclasses supported. |
| `compute_metrics` | `(y_true, y_pred, y_proba=None, task_type=...) → dict` | Accuracy/ROC-AUC (classification) or R²/RMSE/MAE (regression) |
| `to_tensor` | `(array) → FloatTensor` | Numpy to tensor |
| `extract_params` | `(class_name, module_hint=None) → dict` | Discover constructor params for any class |

The `params` dict is also available in your code.

### Critical Rules

- **Always** pass `preprocessor=preprocessor` to `save_model`. Without it, inference fails.
- **Always** use `encode_labels(y)` — never manually map labels.
- **Always** convert to tensors before using `DataLoader` — never pass DataFrames.
- **Detect task type** robustly: check `params.get("task_type")` first, then use dtype and unique-value ratio — don't rely solely on `n_classes > 10`.

## Experiment Protocol

1. **Baseline first.** Run the default config with no modifications. This is the metric to beat.
2. **One change at a time.** Each experiment modifies exactly one thing. Never change multiple variables simultaneously.
3. **Keep or discard.** Metric improved → keep as new baseline. Equal or worse → discard, revert. Be binary.
4. **Build on successes.** Each experiment starts from the current best config, not from scratch.
5. **Simplicity wins.** Equal performance with less complexity is a win.
6. **Exhaust the space.** If standard approaches plateau, escalate: different activations → different optimizers → LR schedules → architecture changes → advanced regularization → Optuna search → ensembles.
7. **Log everything.** Print a cumulative experiment table after every run (model name, val metric, keep/discard, description).

## Writing Training Code

This is **tabular data**. Write complete, self-contained Python scripts. Your code should:

1. Load data with `load_dataset`, split features/target, call `encode_labels`
2. Detect task type, preprocess features with `preprocess`
3. Train/val split with `train_test_split`, convert to tensors
4. Build a model, train with early stopping, evaluate on validation set
5. Call `save_model` with the preprocessor and metrics

### Architecture Guidance

**Layer sizing** — scale with dataset size:

| Rows | Layers | Width | Dropout |
|------|--------|-------|---------|
| < 500 | 1 | 32-64 | 0.4-0.5 |
| 500-2K | 1-2 | 64-128 | 0.3-0.4 |
| 2K-10K | 2-3 | 128-256 | 0.2-0.3 |
| 10K-50K | 2-4 | 256-512 | 0.1-0.3 |
| 50K+ | 3-5 | 256-512 | 0.1-0.2 |

**Components to use:**

- **BatchNorm**: Always with 2+ layers. Stabilizes training, allows higher LR.
- **GELU** over ReLU as default activation. SiLU for deeper nets.
- **Gradient clipping** (`clip_grad_norm_(params, 1.0)`): Always.
- **Residual connections**: When using 3+ layers. Define a `ResidualBlock` with `nn.Module`.
- **Entity embeddings**: For high-cardinality categoricals (>10 unique values) — dramatically beats one-hot encoding. Use `nn.Embedding`.
- **Attention** (`nn.MultiheadAttention`): For many interacting features on medium+ datasets.

**Batching**: Full-batch for <50K rows. Mini-batch (`TensorDataset` + `DataLoader`, batch 512-2048) for larger datasets.

### Hyperparameter Guidance

**Optimizers**: Start with `AdamW(lr=1e-3, weight_decay=1e-2)`. Try SGD+momentum if AdamW overfits.

**LR schedules** (always use one for >200 epochs):
- `ReduceLROnPlateau(patience=10, factor=0.5)` — conservative default
- `OneCycleLR(max_lr, epochs, steps_per_epoch=1)` — faster convergence, state-of-the-art
- `CosineAnnealingWarmRestarts(T_0=20, T_mult=2)` — long training, escaping local minima

**Early stopping**: Track val loss, save best state with `deepcopy(model.state_dict())`, patience 20-30 epochs.

**Regularization** (escalate as needed):
- Label smoothing: `CrossEntropyLoss(label_smoothing=0.05-0.1)` — prevents overconfidence
- Mixup: Interpolate training pairs — effective tabular augmentation
- SWA (`AveragedModel` + `SWALR`): Average weights from late training — better generalization

**Automated search**: When manual tuning stalls, use `optuna` with 3-fold cross-validation (`StratifiedKFold` for classification) to search over architecture and hyperparameters jointly.

### Diagnosis → Action

| Symptom | Fix |
|---------|-----|
| Train high, val low (overfitting) | +dropout, +weight_decay, -width/depth, +label smoothing, +mixup |
| Both low (underfitting) | +width/depth, -dropout, -weight_decay, longer training, +residual blocks |
| Loss → NaN | ½ LR, +grad clipping, check data for NaN/inf |
| Val metric flat (plateau) | Try OneCycleLR, switch optimizer, change activation, +SWA |
| High variance across seeds | +ensemble, +SWA, +regularization |
| Categoricals hurting | Use entity embeddings instead of one-hot |
