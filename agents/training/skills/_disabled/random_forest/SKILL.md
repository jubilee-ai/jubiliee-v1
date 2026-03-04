---
name: random_forest
description: Train a Random Forest model for classification or regression. Use for fraud detection, risk scoring, tabular ML tasks needing non-linear relationships and feature interactions with minimal feature engineering.
---

# Random Forest

Ensemble of decision trees for classification or regression. Captures non-linearities and feature interactions with minimal feature engineering. Uses sklearn RandomForestClassifier/RandomForestRegressor with automatic preprocessing pipeline.

## Training Execution

To train this model, call the `train_with_skill` tool with:
- `skill_name`: `"random_forest"`
- `params`: JSON object with the parameters described below (must include `model_name`, `train_dataset_ref`, `target_column`, `task_type`)

## Task Types
- Classification (binary and multiclass)
- Regression

## When to Use
- Tabular classification: fraud detection, risk scoring, churn prediction
- Tabular regression: claim amount prediction, pricing, LTV estimation
- When you need to capture non-linear relationships and feature interactions
- Mixed feature types (numeric + categorical)
- As a strong baseline before trying boosting methods

## When NOT to Use
- Very high-dimensional sparse data (text, images)
- When strict interpretability is required — use logistic regression
- When calibrated probabilities are needed — consider calibration wrapper
- Real-time inference with tight latency constraints

## Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | str | Unique name for this model (e.g., 'rf_v1') |
| train_dataset_ref | str | Reference name of the registered training dataset |
| target_column | str | Name of the target column to predict |
| task_type | str | 'classification' or 'regression' |

## Optional Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| description | str | "" | Human-readable model description |
| feature_columns | list[str] | null | Features to use. null = all except target |
| categorical_columns | list[str] | null | Columns to one-hot encode. null = auto-detect |
| sample_weight_column | str | null | Column with per-row sample weights |
| n_estimators | int | 100 | Number of trees. More = better but diminishing returns after 300-500 |
| max_depth | int | null | Max tree depth. null = unlimited. Set 5-15 to prevent overfitting |
| min_samples_split | int | 2 | Min samples to split a node. Increase for noisy data |
| min_samples_leaf | int | 1 | Min samples at leaf node. Increase for smoother predictions |
| max_features | str/float | "sqrt" | Features per split: 'sqrt', 'log2', or float fraction |
| bootstrap | bool | true | Use bootstrap samples. Enables OOB score |
| class_weight | str | null | 'balanced' or 'balanced_subsample' for imbalanced classification |
| random_state | int | 42 | Random seed |
| n_jobs | int | -1 | Parallel jobs. -1 = all processors |
| test_size | float | 0.2 | Fraction held out for evaluation |

## Hyperparameter Tuning Guide

### Overfitting (train >> val by >0.1)
- Reduce max_depth (try 4-6)
- Increase min_samples_leaf (try 5-10)
- Increase min_samples_split (try 5-10)

### Underfitting (both metrics low)
- Increase n_estimators (try 200-500)
- Set max_depth=null (unlimited)
- Reduce min_samples_leaf to 1

### Imbalanced Data (positive rate < 20%)
- ALWAYS use class_weight='balanced'
- When switching FROM logistic regression with class_weight='balanced', carry it forward

## Output Metrics
- **train_score / test_score**: Accuracy (classification) or R² (regression)
- **roc_auc**: For classification. >0.7 acceptable, >0.8 good
- **feature_importances**: Higher = more predictive
- **oob_score**: Out-of-bag generalization estimate (if bootstrap=True)

## Feature Engineering Guidance

Random forests build decorrelated decision trees. Trees naturally capture non-linear
effects and interactions through recursive splits. The philosophy is MINIMAL
TRANSFORMATION, GENEROUS INCLUSION.

### Encoding Strategy
- ORDINAL encode categoricals — trees split on thresholds efficiently
- Avoid one-hot for features with >10 categories — dilutes importance
- One-hot acceptable only for low-cardinality features (≤5 values)
- For very high cardinality (>50), use GROUP_AGG (mean-target encoding)

### Transformations
- PASSTHROUGH numeric features as-is — trees are SCALE-INVARIANT
- SKIP binning — trees find optimal split points automatically
- KEEP continuous features continuous
- CREATE ratio/difference features ONLY with clear domain meaning
- SKIP log/sqrt transforms — trees handle skewed data natively

### Feature Selection
- Include MORE features rather than fewer — RF handles high-dimensional spaces well
- Keep mildly correlated features (|r| < 0.95)
- Only drop truly redundant (|r| > 0.95) or leaky features
- 15-40 features is typical

### What to Avoid
- DO NOT create many one-hot columns from high-cardinality features
- DO NOT scale or normalize — no benefit for trees
- DO NOT use PCA — destroys interpretability

## Example
```json
{
  "model_name": "rf_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "fraud",
  "task_type": "classification",
  "n_estimators": 200,
  "max_depth": 10,
  "class_weight": "balanced"
}
```
