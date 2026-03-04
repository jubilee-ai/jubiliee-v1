---
name: naive_bayes
description: Train a Naive Bayes classifier for fast probabilistic classification. Use as a baseline for small datasets, text-like features, or when training speed matters. Supports Gaussian, Multinomial, and Complement variants.
---

# Naive Bayes

Fast probabilistic classifier using sklearn Naive Bayes with automatic preprocessing. Supports Gaussian (continuous features), Multinomial (count data), and Complement (imbalanced data) variants.

## Training Execution

To train this model, call the `train_with_skill` tool with:
- `skill_name`: `"naive_bayes"`
- `params`: JSON object with the parameters described below (must include `model_name`, `train_dataset_ref`, `target_column`)

## Task Types
- Classification only (binary and multiclass)

## When to Use
- Fast baseline classifier before trying more complex models
- Small to medium datasets where training speed matters
- High-dimensional feature spaces
- When probabilistic class predictions are needed
- Imbalanced datasets (use complement variant)
- When training data is limited — NB needs very few samples

## When NOT to Use
- Regression problems — NB is classification only
- When features have strong dependencies/correlations
- When you need well-calibrated probabilities
- Complex non-linear decision boundaries

## Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | str | Unique name for this model (e.g., 'nb_v1') |
| train_dataset_ref | str | Reference name of the registered training dataset |
| target_column | str | Name of the target column to predict |

## Optional Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| description | str | "" | Human-readable model description |
| feature_columns | list[str] | null | Features to use. null = all except target |
| categorical_columns | list[str] | null | Columns to one-hot encode. null = auto-detect |
| sample_weight_column | str | null | Column with per-row sample weights |
| variant | str | "gaussian" | NB variant: 'gaussian', 'multinomial', or 'complement' |
| var_smoothing | float | 1e-9 | (Gaussian) Variance smoothing. Increase if overfitting |
| alpha | float | 1.0 | (Multinomial/Complement) Laplace smoothing parameter |
| fit_prior | bool | true | Learn class priors from data |
| class_prior | list[float] | null | Explicit class priors (must sum to 1.0) |
| test_size | float | 0.2 | Fraction held out for evaluation |

## Variant Selection
- **gaussian** (default): For general tabular data with continuous features. Assumes normal distribution per class.
- **multinomial**: For count/frequency data (word counts, tf-idf, event counts). Requires non-negative features.
- **complement**: Improved multinomial for IMBALANCED datasets. Uses complement class statistics.

## Hyperparameter Tuning Guide

### Gaussian NB
- var_smoothing: Start with 1e-9. If overfitting, increase to 1e-5 or 1e-3

### Multinomial/Complement NB
- alpha: Start with 1.0 (Laplace). If underfitting, try 0.1, 0.01

### Imbalanced Data
- Use variant='complement' — specifically designed for class imbalance
- Or set class_prior to reflect the desired decision boundary

## Output Metrics
- **train_accuracy / test_accuracy**: Compare to majority class rate
- **test_roc_auc**: NB often has decent AUC even if accuracy is moderate
- **classification_report**: Check precision/recall per class
- **class_log_prior**: Log prior probabilities — verify they match class distribution

## Feature Engineering Guidance

Naive Bayes applies Bayes' theorem with conditional independence assumption.
Feature engineering should focus on INDEPENDENCE and DISTRIBUTIONAL FIT.

### Encoding Strategy
- ONE-HOT encode nominal categoricals with ≤15 unique values (drop_first=true)
- ORDINAL encode only with genuinely monotonic relationships
- For high-cardinality categoricals (>15 values), use GROUP_AGG

### Feature Independence (Critical)
- DROP one from each correlated pair (|r| > 0.7) — MORE important for NB than other models
- Correlated features DOUBLE-COUNT evidence, inflating NB's confidence
- Prefer FEWER, MORE INDEPENDENT features
- AVOID interaction terms — NB cannot use them effectively

### Feature Selection
- Aim for 5-20 well-chosen, independent features
- Remove features with near-zero variance
- Focus on features with clear discriminative power between classes

### What to Avoid
- DO NOT include highly correlated features
- DO NOT create polynomial or interaction features
- DO NOT use raw high-cardinality categoricals as one-hot
- DO NOT expect well-calibrated probabilities
- DO NOT use NB for regression tasks

## Example
```json
{
  "model_name": "nb_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "churn",
  "variant": "complement",
  "description": "Complement NB for imbalanced churn prediction"
}
```
