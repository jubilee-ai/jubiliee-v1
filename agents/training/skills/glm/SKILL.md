---
name: glm
description: Train a Generalized Linear Model for insurance pricing and actuarial modeling. Use for claim frequency (Poisson), severity (Gamma), and pure premium (Tweedie) regression with interpretable multiplicative coefficients.
---

# Generalized Linear Model (GLM)

GLM for insurance pricing and actuarial modeling using sklearn. Supports Poisson (claim frequency), Gamma (claim severity), and Tweedie (pure premium) distributions with log link function. Coefficients represent multiplicative effects.

## Training Execution

To train this model, call the `train_with_skill` tool with:
- `skill_name`: `"glm"`
- `params`: JSON object with the parameters described below (must include `model_name`, `train_dataset_ref`, `target_column`, `distribution`)

## Task Types
- Regression (actuarial/insurance pricing)

## When to Use
- **Poisson**: Predicting claim counts (0, 1, 2, 3...). "How many claims will this policy have?"
- **Gamma**: Predicting claim severity (always > 0). "Given a claim, how much will it cost?"
- **Tweedie**: Predicting total claim cost (many zeros, some positive). "What is the expected total loss?"
- Auto insurance premium calculation
- Health insurance claim cost prediction
- Property damage severity modeling
- Loss reserving and IBNR estimation

## When NOT to Use
- Classification tasks
- Non-insurance regression without distributional assumptions
- When interpretability is not required — tree-based models may perform better

## Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | str | Unique name for this model (e.g., 'glm_v1') |
| train_dataset_ref | str | Reference name of the registered training dataset |
| target_column | str | Name of the target column to predict |
| distribution | str | 'poisson' (counts), 'gamma' (positive continuous), or 'tweedie' (zero-inflated positive) |

## Optional Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| description | str | "" | Human-readable model description |
| feature_columns | list[str] | null | Features to use. null = all except target |
| categorical_columns | list[str] | null | Columns to one-hot encode. null = auto-detect |
| exposure_column | str | null | Exposure for rate modeling (e.g., policy duration, years at risk) |
| sample_weight_column | str | null | Column with per-row sample weights |
| tweedie_power | float | 1.5 | Tweedie power parameter. Range (1, 2) for insurance. Only used when distribution='tweedie' |
| alpha | float | 1.0 | L2 regularization strength. 0 = no regularization |
| solver | str | "lbfgs" | 'lbfgs' or 'newton-cholesky' |
| max_iter | int | 100 | Max solver iterations |
| fit_intercept | bool | true | Whether to fit intercept term |
| random_state | int | 42 | Random seed |
| test_size | float | 0.2 | Fraction held out for evaluation |

## Distribution Selection Guide
- **Poisson**: Target is non-negative integers (counts). Use for claim frequency.
- **Gamma**: Target is strictly positive (amounts). Use for claim severity.
- **Tweedie**: Target is non-negative with point mass at zero. Use for total claim cost / pure premium.

## Exposure Modeling
Use exposure_column for rate modeling:
- Policy duration (0.5 = 6 months, 1.0 = full year)
- Years at risk
- Miles driven (usage-based insurance)

## Coefficient Interpretation
GLM coefficients are on log scale (log link function):
- coef = 0.10 → exp(0.10) = 1.105 → 10.5% increase in expected value
- coef = -0.20 → exp(-0.20) = 0.819 → 18.1% decrease in expected value

## Output Metrics
- **train_deviance / test_deviance**: D² score (like R² for GLMs)
- **train_mae / test_mae**: Mean absolute error
- **train_rmse / test_rmse**: Root mean squared error
- **coefficients**: Feature coefficients on log scale with multiplier interpretation
- **intercept**: Base rate = exp(intercept)

## Feature Engineering Guidance

GLMs model the conditional mean through a LINK FUNCTION. Feature engineering must
align with the link and the distributional assumption. GLMs demand INTERPRETABILITY.

### Encoding Strategy
- ONE-HOT encode categoricals with ≤15 unique values (drop_first=true)
- For high-cardinality, BIN or GROUP_AGG to reduce dimensionality
- ORDINAL encode only with genuinely monotonic effects

### Transformations & Link Function Alignment
- With LOG LINK: coefficients are MULTIPLICATIVE (exp(β) = factor change)
- Log-transform predictors when effect is multiplicative on the mean
- CREATE ratio features for rate structures (claims/exposure, cost/unit)
- Use exposure_column as OFFSET for rate modeling — do NOT include as regular feature
- BIN continuous features with complex non-linear effects into 5-10 quantile bins

### Feature Selection & Multicollinearity
- Keep the model INTERPRETABLE — coefficients must be explainable
- DROP one from correlated pairs (|r| > 0.8), target VIF < 5
- Aim for 8-25 carefully chosen features
- GLMs have NO built-in feature selection

### Feature Interactions
- GLMs require EXPLICIT interaction specification (x₁ × x₂)
- Always include main effects with interaction terms
- Limit to 2-3 well-motivated interactions

### What to Avoid
- DO NOT model a RATE as response in Poisson — model COUNT with offset
- DO NOT include too many noisy features — no built-in regularization by default
- DO NOT use raw high-cardinality categoricals
- DO NOT ignore distributional assumptions

## Example (Claim Frequency)
```json
{
  "model_name": "frequency_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "num_claims",
  "distribution": "poisson",
  "exposure_column": "policy_years",
  "description": "Poisson model for auto claim frequency"
}
```

## Example (Pure Premium)
```json
{
  "model_name": "pure_premium_v1",
  "train_dataset_ref": "pipeline_train_features",
  "target_column": "total_claim_cost",
  "distribution": "tweedie",
  "tweedie_power": 1.5,
  "description": "Tweedie model for expected total loss"
}
```
