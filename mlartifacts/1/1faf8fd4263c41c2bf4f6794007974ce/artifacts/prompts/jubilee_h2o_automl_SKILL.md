---
name: h2o_automl
description: When to use H2O-3 tools vs sklearn — you choose per task; this file is guidance only.
---

# H2O-3 and AutoML (tool-driven)

There is **no** `train.py` here. The training agent already has **`h2o_*`** and **`mlflow_*`** tools.
Use this SKILL as a **manual** for when each tool tends to help.

## When H2O tends to win

- Medium/large **tabular** supervised problems where you want **many algorithms + stacking** without hand-writing each one.
- You want a **leaderboard** and to cherry-pick or stack top `model_id`s.
- You need **MOJO export** for deployment via `h2o_to_registered_model`.

## When to stay on sklearn / skills

- Strong **interpretability** requirements (coefficients, simple pipelines).
- Very **small** data where H2O JVM overhead dominates.
- **Unsupervised** or **neural_networks** skill flows — H2O tools are optional extras, not replacements.

## Suggested tool flows (examples only — not mandatory)

1. `h2o_init` → `h2o_import_frame` → `h2o_automl_run` → `h2o_leaderboard` → pick `leader_model_id` → `h2o_to_registered_model` (pass real `feature_columns` list).
2. Same as (1) but replace AutoML with one or more `h2o_train_estimator` calls, then `h2o_stack_models`.
3. Log runs with `mlflow_start_run` / `mlflow_log_*` when `MLFLOW_TRACKING_URI` is configured.

## Shutdown

Call `h2o_shutdown` when finished to free the JVM.
