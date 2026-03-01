---
name: logistic_regression
description: Train a Logistic Regression classifier for binary/multiclass classification. Use for underwriting, risk assessment, governed decisioning, or as a strong interpretable baseline with coefficient weights.
---

# Logistic Regression

Binary/multiclass classification using sklearn LogisticRegression with automatic preprocessing pipeline (StandardScaler + OneHotEncoder).

## Training Execution

To train this model, call the `train_with_skill` tool with:
- `skill_name`: `"logistic_regression"`
- `params`: JSON object with the parameters described below (must include `model_name`, `train_dataset_ref`, `target_column`)

## Task Types
- Classification (binary and multiclass)

## When to Use
- Binary classification: approve/decline, risk flag, lapse yes/no
- Multiclass classification: risk tiers, customer segments
- When interpretability matters: coefficient weights show feature importance
- As a strong baseline before trying complex models
- For governed decisioning where model transparency is required

## When NOT to Use
- Regression problems (continuous targets)
- Non-linear relationships that can't be captured with interactions
- Very large datasets (>1M rows) — consider SGDClassifier

## Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | str | Unique name for this model (e.g., 'lr_v1', 'loan_default_v1') |
| train_dataset_ref | str | Reference name of the registered training dataset |
| target_column | str | Name of the target column to predict |

## Optional Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| description | str | "" | Human-readable model description |
| feature_columns | list[str] | null | Features to use. null = all except target |
| categorical_columns | list[str] | null | Columns to one-hot encode. null = auto-detect |
| sample_weight_column | str | null | Column with per-row sample weights |
| C | float | 1.0 | Inverse regularization strength. Lower = stronger regularization |
| l1_ratio | float | 0.0 | L1 vs L2 mixing: 0.0=L2 (Ridge), 1.0=L1 (Lasso), between=ElasticNet |
| solver | str | "lbfgs" | Optimizer: 'lbfgs', 'liblinear', 'saga', etc. Use 'saga' for L1/ElasticNet |
| max_iter | int | 100 | Max solver iterations. Increase to 500-1000 if convergence warnings |
| class_weight | str | null | Set 'balanced' for imbalanced classes |
| fit_intercept | bool | true | Whether to fit intercept term |
| random_state | int | 42 | Random seed for reproducibility |
| test_size | float | 0.2 | Fraction held out for evaluation |

## Hyperparameter Tuning Guide

### Overfitting (train >> val by >0.1)
- Reduce C (try 0.1, 0.01 for stronger regularization)
- Try l1_ratio=0.5 with solver='saga' for feature selection

### Underfitting (both metrics low)
- Increase C (try 10, 100 for less regularization)
- Add interaction features in feature engineering

### Imbalanced Data (positive rate < 20%)
- ALWAYS use class_weight='balanced'
- Focus on ROC-AUC and F1 rather than accuracy

### Solver Compatibility
- l1_ratio > 0 and < 1 (ElasticNet): requires solver='saga'
- l1_ratio = 1.0 (pure L1): requires solver='saga' or 'liblinear'
- l1_ratio = 0.0 (pure L2): any solver works, 'lbfgs' is default

## Output Metrics
- **train_accuracy / test_accuracy**: Compare to baseline (majority class rate)
- **test_roc_auc**: Ranking quality. >0.7 acceptable, >0.8 good, >0.9 excellent
- **classification_report**: Precision/recall/F1 per class
- **coefficients**: Positive = increases probability, negative = decreases

## Feature Engineering Guidance

Logistic regression fits a LINEAR decision boundary in log-odds space. It cannot
discover non-linear relationships or interactions — feature engineering must
compensate by creating them explicitly.

### Encoding Strategy
- ONE-HOT encode nominal categoricals with ≤15 unique values (drop_first=true)
- ORDINAL encode only when a clear natural order exists
- For high-cardinality categoricals (>15 values), use GROUP_AGG for mean-target encodings

### Scaling & Transformations
- Passthrough numeric features — pipeline includes StandardScaler
- For heavily skewed features (|skew| > 2), create log expressions
- BIN features with extreme distributions into quantile bins (5-10 bins)

### Feature Interactions (Critical)
- CREATE ratio features for related quantities (debt/income, claims/premium)
- CREATE interaction terms for pairs with domain-motivated relationships
- Keep main-effect features when creating interactions

### Multicollinearity
- DROP one from each correlated pair (|r| > 0.8), target VIF < 5
- Events Per Variable: at least 10-15 events per feature
- Prefer 8-20 well-chosen features

## Example
```json
{
  "model_name": "loan_default_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "default",
  "description": "Predicts loan default for retail customers",
  "C": 0.1,
  "class_weight": "balanced",
  "solver": "lbfgs"
}
```
