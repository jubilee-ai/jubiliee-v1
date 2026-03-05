---
name: sklearn_generic
description: Train any scikit-learn estimator with automatic hyperparameter tuning via RandomizedSearchCV (with scipy.stats distributions) and a robust preprocessing pipeline (imputation + StandardScaler + OneHotEncoder).
---

# Sklearn Generic

Train any supported scikit-learn estimator through a single unified interface. Estimators are auto-discovered from the installed sklearn version, so new models are available without code changes. Hyperparameters are tuned automatically via cross-validated search using continuous distributions for efficient exploration.

## Training Execution

Call `train_with_skill` with:
- `skill_name`: `"sklearn_generic"`
- `params`: JSON object with the parameters below

## Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `estimator` | string | Any sklearn classifier or regressor class name (see tables below) |
| `model_name` | string | Unique name for the trained model |
| `train_dataset_ref` | string | Reference to the registered training dataset |
| `target_column` | string | Column to predict |

## Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_tune` | bool | `true` | Run RandomizedSearchCV to find best hyperparameters |
| `n_search_iter` | int | `20` | Number of random parameter combinations to try |
| `cv_folds` | int | `5` | Cross-validation folds for tuning |
| `hyperparameters` | dict | `{}` | See hyperparameters section below |
| `feature_columns` | list | all | Subset of columns to use as features |
| `categorical_columns` | list | auto-detect | Columns to one-hot encode |
| `description` | string | auto | Human-readable model description |
| `random_state` | int | `42` | Random seed for reproducibility |

## Hyperparameters

The `hyperparameters` dict accepts two kinds of values:

- **Single value** → fixed on the estimator (not tuned). Example: `"class_weight": "balanced"`
- **List of values** → added to the auto-tune search space. Example: `"C": [0.01, 0.1, 1.0, 10.0]`

You can mix both in the same dict:

```json
{
  "hyperparameters": {
    "class_weight": "balanced",
    "C": [0.01, 0.1, 1.0, 10.0],
    "solver": "lbfgs"
  }
}
```

This fixes `class_weight="balanced"` and `solver="lbfgs"`, then searches over `C` values.

When `auto_tune` is true (default), the skill already has built-in search spaces for each estimator using scipy.stats distributions (loguniform, randint, uniform) for efficient continuous sampling. Any list values in `hyperparameters` are merged with (and override) the built-in search space. Single values bypass tuning entirely for that parameter and are also removed from the search space.

## Available Estimators

Any sklearn classifier or regressor class name works as the `estimator` parameter. The tables below highlight the most commonly useful ones. Estimators with built-in search spaces get automatic tuning; others can still be trained with manual hyperparameters.

### Classification

| Estimator | Best For |
|-----------|----------|
| `LogisticRegression` | Interpretable linear classifier, baseline, regulated environments |
| `SVC` | Small-to-medium datasets, high-dimensional data, non-linear boundaries |
| `LinearSVC` | Large datasets, linear decision boundaries, text classification |
| `KNeighborsClassifier` | Instance-based learning, small datasets, local patterns |
| `DecisionTreeClassifier` | Interpretable models, feature importance |
| `RandomForestClassifier` | Non-linear patterns, feature importance, robust ensemble |
| `GradientBoostingClassifier` | High accuracy on tabular data, structured data |
| `HistGradientBoostingClassifier` | **Fastest tree model**, handles NaN natively, large datasets, best default choice |
| `AdaBoostClassifier` | Boosting weak learners, reducing bias |
| `BaggingClassifier` | Reducing variance, parallel ensemble |
| `ExtraTreesClassifier` | Fast ensemble alternative to Random Forest |
| `MLPClassifier` | Complex non-linear patterns, multi-layer neural network |
| `RidgeClassifier` | Fast linear classifier with L2 regularization |
| `SGDClassifier` | Very large datasets, online learning |
| `GaussianNB` | Fast baseline, continuous features, small datasets |
| `MultinomialNB` | Text classification, count/frequency data |
| `ComplementNB` | Imbalanced text classification |
| `BernoulliNB` | Binary/boolean features |
| `PassiveAggressiveClassifier` | Online learning, large-scale linear classification |

### Regression

| Estimator | Best For |
|-----------|----------|
| `SVR` | Small-to-medium datasets, non-linear relationships |
| `LinearSVR` | Large datasets, linear regression with epsilon-insensitive loss |
| `KNeighborsRegressor` | Local patterns, non-parametric regression |
| `DecisionTreeRegressor` | Interpretable non-linear regression |
| `RandomForestRegressor` | Non-linear regression, robust ensemble |
| `GradientBoostingRegressor` | High accuracy on tabular data |
| `HistGradientBoostingRegressor` | **Fastest tree model**, handles NaN natively, large datasets, best default choice |
| `AdaBoostRegressor` | Boosting weak learners |
| `BaggingRegressor` | Reducing variance |
| `ExtraTreesRegressor` | Fast ensemble regression |
| `MLPRegressor` | Complex non-linear patterns |
| `LinearRegression` | Simple baseline, fully interpretable |
| `Ridge` | Linear regression with L2 regularization (prevents overfitting) |
| `Lasso` | Linear regression with L1 regularization (feature selection) |
| `ElasticNet` | Combined L1+L2 regularization |
| `HuberRegressor` | Robust to outliers |
| `SGDRegressor` | Very large datasets, online learning |
| `PassiveAggressiveRegressor` | Online learning, large-scale linear regression |

## How Auto-Tuning Works

When `auto_tune` is true (default), the skill:
1. Selects a built-in search space for the chosen estimator (using scipy.stats distributions for continuous/integer params)
2. Removes any parameters you explicitly fixed via `hyperparameters`
3. Runs `RandomizedSearchCV` with `n_search_iter` random samples from the distributions
4. Automatically selects the best scoring metric: `roc_auc` for imbalanced binary, `f1_weighted` for imbalanced multiclass, `accuracy` for balanced classification, `r2` for regression
5. Uses k-fold cross-validation on the training data to evaluate each combination
6. Refits the best configuration on all training data
7. Reports the best hyperparameters and CV score

Set `auto_tune` to false and pass explicit `hyperparameters` if you want full control.

## Preprocessing Pipeline

The skill automatically builds a preprocessing pipeline:
- **Numeric columns**: median imputation → StandardScaler
- **Categorical columns**: most-frequent imputation → OneHotEncoder (handle_unknown="ignore")

Missing values are handled automatically — no need to drop or fill NaN before training.

## Examples

Simplest call (auto-tune with built-in search space):

```json
{
  "estimator": "HistGradientBoostingClassifier",
  "model_name": "fraud_detection_hgb_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "is_fraud"
}
```

With fixed + tuned hyperparameters:

```json
{
  "estimator": "LogisticRegression",
  "model_name": "lr_balanced_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "Default",
  "hyperparameters": {
    "class_weight": "balanced",
    "C": [0.01, 0.1, 1.0, 10.0]
  }
}
```

No auto-tune (all values fixed):

```json
{
  "estimator": "Ridge",
  "model_name": "ridge_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "price",
  "auto_tune": false,
  "hyperparameters": {"alpha": 10.0}
}
```
