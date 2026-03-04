---
name: xgboost
description: Train an XGBoost gradient boosted tree model for classification or regression. Use when maximum accuracy on tabular data is critical, with built-in early stopping, loss curve tracking, and regularization.
---

# XGBoost

Gradient boosted trees with early stopping, loss curve tracking, and built-in regularization. Uses XGBClassifier/XGBRegressor with automatic preprocessing pipeline. Often achieves state-of-the-art results on tabular data.

## Training Execution

To train this model, call the `train_with_skill` tool with:
- `skill_name`: `"xgboost"`
- `params`: JSON object with the parameters described below (must include `model_name`, `train_dataset_ref`, `target_column`, `task_type`)

## Task Types
- Classification (binary and multiclass)
- Regression

## When to Use
- Tabular data where accuracy is critical
- When you need the best possible performance
- Complex non-linear relationships
- After trying simpler baselines (logistic regression, random forest)

## When NOT to Use
- When interpretability is critical — use logistic regression
- Very small datasets (<100 samples)
- When training speed is more important than accuracy

## Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | str | Unique name for this model (e.g., 'xgb_v1') |
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
| n_estimators | int | 100 | Max boosting rounds. 100-1000 typical |
| max_depth | int | 6 | Max tree depth. 3-10 typical |
| learning_rate | float | 0.1 | Step size shrinkage (eta). 0.01-0.3 typical |
| min_child_weight | int | 1 | Min sum of instance weight in child. Higher = more conservative |
| gamma | float | 0 | Min loss reduction for split. Higher = more conservative |
| reg_alpha | float | 0 | L1 regularization. Higher = sparser model |
| reg_lambda | float | 1 | L2 regularization. Higher = simpler model |
| subsample | float | 1.0 | Fraction of samples per tree. <1.0 adds regularization |
| colsample_bytree | float | 1.0 | Fraction of features per tree. <1.0 adds regularization |
| scale_pos_weight | float | null | Balance weight for positive class in binary classification |
| monotone_constraints | dict | null | Per-feature monotonic constraints: {name: 1 (increasing), -1 (decreasing)} |
| tree_method | str | "auto" | Tree algorithm: 'auto', 'exact', 'approx', 'hist' |
| early_stopping_rounds | int | 10 | Stop if no improvement for N rounds |
| random_state | int | 42 | Random seed |
| n_jobs | int | -1 | Parallel threads |
| test_size | float | 0.2 | Fraction for test set |
| validation_size | float | 0.1 | Fraction for early stopping validation |

## Hyperparameter Tuning Guide

### Overfitting (train >> val by >0.1)
- Increase reg_alpha (try 0.1, 1.0) for L1 regularization
- Increase reg_lambda (try 2.0, 5.0) for L2 regularization
- Increase gamma (try 0.1, 0.5) for minimum split gain
- Reduce learning_rate (try 0.05) with more n_estimators
- Reduce max_depth (try 3-4)
- Reduce subsample/colsample_bytree (try 0.7-0.9)

### Underfitting (both metrics low)
- Increase max_depth (try 8-10)
- Increase n_estimators (try 500-1000)
- Reduce regularization parameters

### Imbalanced Data (positive rate < 20%)
- ALWAYS use scale_pos_weight = (num_negatives / num_positives)
- E.g., 5% positive → scale_pos_weight = 19.0
- When switching FROM logistic regression/random forest with class_weight='balanced', you MUST use scale_pos_weight

### Learning Rate Strategy
- Lower learning_rate (0.01-0.05) with proportionally more n_estimators often gives better results
- Use early_stopping_rounds to find optimal number of trees automatically

## Output Metrics
- **train_score / val_score / test_score**: Accuracy (classification) or R² (regression)
- **roc_auc**: For classification
- **training_history**: Loss curve per boosting round (train and validation)
- **best_iteration**: Optimal number of trees from early stopping
- **feature_importances**: Higher = more predictive

## Feature Engineering Guidance

XGBoost is a gradient-boosted tree ensemble with built-in regularization and native
missing value handling. Focus on CLEAN DATA and DOMAIN-MEANINGFUL DERIVED FEATURES.

### Encoding Strategy
- ORDINAL encode categoricals — XGBoost splits on ordered values efficiently
- Avoid one-hot for features with >10 categories — use ordinal instead
- One-hot fine for very low cardinality (≤5 values)
- For high-cardinality (>50), use GROUP_AGG for mean-target encodings

### Transformations
- PASSTHROUGH numeric features as-is — tree-based XGBoost is SCALE-INVARIANT
- XGBoost handles missing values NATIVELY — do not impute unless domain-motivated
- CREATE ratio/difference features when domain-meaningful
- SKIP binning — XGBoost uses histogram-based splitting internally
- SKIP log/sqrt transforms unless extreme outliers dominate all splits

### Feature Selection
- Include a GENEROUS set of features — XGBoost has built-in regularization
- Keep correlated features — boosting distributes importance differently than bagging
- 15-50 features is typical
- Only drop clearly leaky or irrelevant features

### What to Avoid
- DO NOT create massive one-hot expansion
- DO NOT remove mildly correlated features
- DO NOT scale or normalize features
- DO NOT impute missing values by default

## Example
```json
{
  "model_name": "xgb_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "fraud",
  "task_type": "classification",
  "n_estimators": 200,
  "max_depth": 6,
  "learning_rate": 0.1,
  "scale_pos_weight": 19.0,
  "early_stopping_rounds": 10
}
```
