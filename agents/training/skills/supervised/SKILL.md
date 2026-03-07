---
name: sklearn_generic
description: Train a scikit-learn supervised learning estimator. Use this guide to select the right model for the task.
---

# Sklearn Generic — Supervised Model Selection

This skill trains scikit-learn **supervised learning** estimators (classifiers and regressors). Use this guide to choose the correct model for the user's data and problem.

## How to Choose a Model

Work through these questions in order:

1. **Classification or regression?** Is the target a category (classification) or a continuous number (regression)?
2. **Dataset size.** How many samples? Some models scale poorly beyond ~10K rows.
3. **Interpretability.** Does the user need to explain predictions (e.g. regulated industry)?
4. **Feature types.** Mostly numeric? Text/count data? Mix of categorical and numeric?
5. **Data quality.** Are there outliers? Missing values? Class imbalance?

Use the decision flowcharts below to narrow down the estimator, then confirm with the detailed tables.

---

## Classification

### Quick Decision Flowchart

```
Start
 │
 ├─ Need interpretability? ──► LogisticRegression (linear) or DecisionTreeClassifier
 │
 ├─ Text / count data? ──► MultinomialNB  (try ComplementNB if imbalanced)
 │
 ├─ Binary features? ──► BernoulliNB
 │
 ├─ Very large dataset (>100K rows)?
 │    ├─ Linear boundary ──► SGDClassifier or LinearSVC
 │    └─ Non-linear ──► HistGradientBoostingClassifier
 │
 ├─ Medium dataset (<100K rows)?
 │    ├─ Default best choice ──► HistGradientBoostingClassifier
 │    ├─ Want feature importance ──► RandomForestClassifier or GradientBoostingClassifier
 │    └─ High-dimensional / kernel trick ──► SVC
 │
 └─ Small dataset (<1K rows)? ──► KNeighborsClassifier or GaussianNB
```

### Classifier Reference

| Family | Estimator | When to Use |
|--------|-----------|-------------|
| **Linear** | `LogisticRegression` | Interpretable baseline; works well when features are roughly linearly separable; good in regulated environments |
| | `RidgeClassifier` | Fast linear classifier with L2 regularization; no probability estimates |
| | `SGDClassifier` | Online / very large datasets; linear model trained via stochastic gradient descent |
| | `PassiveAggressiveClassifier` | Online learning; large-scale linear classification |
| **Support Vector** | `SVC` | Small-to-medium datasets; effective in high-dimensional spaces; non-linear boundaries via kernel trick |
| | `LinearSVC` | Large datasets; faster than SVC when a linear kernel suffices; good for text classification |
| **Neighbors** | `KNeighborsClassifier` | Small datasets; instance-based / local patterns; no training phase |
| | `NearestCentroid` | Very fast; works when classes form compact clusters |
| **Naive Bayes** | `GaussianNB` | Fast baseline; continuous features; strong independence assumption |
| | `MultinomialNB` | Text classification with word counts / TF-IDF |
| | `ComplementNB` | Imbalanced text classification; corrects MultinomialNB bias |
| | `BernoulliNB` | Binary / boolean feature vectors |
| | `CategoricalNB` | Categorical features encoded as integers |
| **Tree** | `DecisionTreeClassifier` | Full interpretability; feature importance; prone to overfitting on its own |
| **Ensemble — Bagging** | `RandomForestClassifier` | General-purpose; robust to overfitting; gives feature importance; handles non-linear patterns |
| | `ExtraTreesClassifier` | Faster alternative to Random Forest; more randomized splits reduce variance further |
| | `BaggingClassifier` | Wraps any base estimator in a bagging ensemble to reduce variance |
| **Ensemble — Boosting** | `GradientBoostingClassifier` | High accuracy on tabular data; sequential boosting with gradient descent |
| | `HistGradientBoostingClassifier` | **Recommended default.** Fastest tree-based model; handles NaN natively; scales to large datasets |
| | `AdaBoostClassifier` | Boosts weak learners (typically shallow trees); sensitive to noise |
| **Neural Network** | `MLPClassifier` | Complex non-linear patterns; multi-layer perceptron; requires feature scaling |
| **Discriminant Analysis** | `LinearDiscriminantAnalysis` | Linear boundaries; works well when features are normally distributed per class; also does dimensionality reduction |
| | `QuadraticDiscriminantAnalysis` | Non-linear (quadratic) boundaries; each class has its own covariance matrix |
| **Calibration** | `CalibratedClassifierCV` | Wraps any classifier to produce well-calibrated probability estimates |

---

## Regression

### Quick Decision Flowchart

```
Start
 │
 ├─ Need interpretability? ──► LinearRegression, Ridge, or Lasso
 │
 ├─ Want automatic feature selection? ──► Lasso or ElasticNet
 │
 ├─ Outliers in target? ──► HuberRegressor or RANSACRegressor or TheilSenRegressor
 │
 ├─ Very large dataset (>100K rows)?
 │    ├─ Linear ──► SGDRegressor
 │    └─ Non-linear ──► HistGradientBoostingRegressor
 │
 ├─ Medium dataset (<100K rows)?
 │    ├─ Default best choice ──► HistGradientBoostingRegressor
 │    ├─ Want feature importance ──► RandomForestRegressor or GradientBoostingRegressor
 │    ├─ Few important features ──► Lasso or ElasticNet
 │    └─ Non-linear, small-to-medium ──► SVR
 │
 └─ Small dataset (<1K rows)? ──► KNeighborsRegressor or Ridge
```

### Regressor Reference

| Family | Estimator | When to Use |
|--------|-----------|-------------|
| **Linear** | `LinearRegression` | Simple baseline; fully interpretable; no regularization |
| | `Ridge` | Linear with L2 regularization; prevents overfitting when features are correlated |
| | `Lasso` | Linear with L1 regularization; drives unimportant feature weights to zero (built-in feature selection) |
| | `ElasticNet` | Combined L1 + L2 regularization; useful when multiple correlated features exist |
| | `LarsCV` | Efficient for high-dimensional data; selects features via Least Angle Regression |
| | `SGDRegressor` | Online / very large datasets; linear model trained via stochastic gradient descent |
| | `PassiveAggressiveRegressor` | Online learning; large-scale linear regression |
| **Robust Linear** | `HuberRegressor` | Linear regression robust to outliers in the target |
| | `RANSACRegressor` | Fits model to inliers only; ignores outlier points entirely |
| | `TheilSenRegressor` | Median-based regression; robust to up to ~29% outliers |
| **Bayesian Linear** | `BayesianRidge` | Ridge-like but estimates regularization strength from data; gives uncertainty on predictions |
| | `ARDRegression` | Automatic relevance determination; sparsity-inducing Bayesian linear model |
| **Support Vector** | `SVR` | Small-to-medium datasets; non-linear relationships via kernel trick |
| | `LinearSVR` | Large datasets; linear regression with epsilon-insensitive loss |
| **Neighbors** | `KNeighborsRegressor` | Local patterns; non-parametric; good for small datasets |
| **Tree** | `DecisionTreeRegressor` | Interpretable non-linear regression; prone to overfitting alone |
| **Ensemble — Bagging** | `RandomForestRegressor` | General-purpose non-linear regression; robust ensemble; feature importance |
| | `ExtraTreesRegressor` | Faster, more randomized alternative to Random Forest |
| | `BaggingRegressor` | Wraps any base estimator in a bagging ensemble |
| **Ensemble — Boosting** | `GradientBoostingRegressor` | High accuracy on tabular data; sequential boosting |
| | `HistGradientBoostingRegressor` | **Recommended default.** Fastest tree-based model; handles NaN natively; scales to large datasets |
| | `AdaBoostRegressor` | Boosts weak learners; sensitive to outliers and noise |
| **Neural Network** | `MLPRegressor` | Complex non-linear patterns; multi-layer perceptron; requires feature scaling |
| **Gaussian Process** | `GaussianProcessRegressor` | Gives uncertainty estimates; works well on small datasets; does not scale beyond ~10K samples |
| **Isotonic** | `IsotonicRegression` | Non-parametric; fits a monotonically increasing/decreasing function |
| **Quantile** | `QuantileRegressor` | Predicts conditional quantiles instead of the mean; useful for prediction intervals |

---

## Model Selection Tips

- **Start with `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor`** unless there is a specific reason not to. They handle missing values, mixed feature types, and scale well.
- **Linear models first for interpretability.** `LogisticRegression` (classification) and `Ridge` / `Lasso` (regression) are strong baselines that are easy to explain.
- **Naive Bayes for text.** `MultinomialNB` with TF-IDF features is a fast, competitive text classifier.
- **Check class balance.** For imbalanced targets, tree-based ensembles or linear models with `class_weight="balanced"` tend to work best.
- **Small data (< 1K samples)** favors simpler models: Naive Bayes, KNeighbors, Ridge, or Gaussian Processes.
- **High-dimensional sparse data** (e.g. text) favors `LinearSVC`, `SGDClassifier`, or `MultinomialNB`.
- When in doubt, train 2–3 models from different families and compare validation metrics.

## Example

```json
{
  "estimator": "HistGradientBoostingClassifier",
  "model_name": "churn_predictor_v1",
  "train_dataset_ref": "customer_features_train",
  "target_column": "churned"
}
```

## Workflow

After selecting an estimator using the guide above, follow these steps:

### Step 1 — Train

Call `train_with_skill(skill_name="supervised", params={...})` with:
- `"estimator"`: the sklearn class name (e.g. `"HistGradientBoostingClassifier"`)
- `"model_name"`: a descriptive unique name (e.g. `"hgb_v1"`, `"lr_v2"`)
- `"train_dataset_ref"`: the training dataset reference
- `"target_column"`: the target column name
- `"hyperparameters"`: (optional) override specific hyperparameters

The training tool automatically discovers the estimator's parameters, runs
cross-validated hyperparameter tuning, and returns results **including a list of
all tunable parameters** (with types, defaults, choices, and scale hints).

Use this parameter list to inform your next iteration — for example, fix a
categorical parameter to a specific choice, or override a numeric range.

### Step 2 — Evaluate and iterate

Call `evaluate_model` on the validation set. Analyze the results, then either:
- Tune hyperparameters on the same estimator (go back to Step 1 with adjusted params)
- Switch to a different estimator (go back to Step 1 with a new estimator name)

When satisfied, run `evaluate_model` on the test set with your best model.
