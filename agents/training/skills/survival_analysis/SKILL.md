---
name: survival_analysis
description: Train a survival analysis model for time-to-event prediction with censoring. Use for policy lapse, claim timing, mortality modeling, and customer lifetime value using Cox PH, Weibull AFT, or Log-Normal AFT models.
---

# Survival Analysis

Time-to-event models with censoring support using the lifelines library. Supports Cox Proportional Hazards (semi-parametric), Weibull AFT, and Log-Normal AFT (parametric) models.

## Training Execution

To train this model, call the `train_with_skill` tool with:
- `skill_name`: `"survival_analysis"`
- `params`: JSON object with the parameters described below (must include `model_name`, `train_dataset_ref`, `duration_column`, `event_column`)

## Task Types
- Survival (time-to-event analysis)

## When to Use
- **Policy Lapse/Churn**: duration = months since policy start, event = lapsed
- **Time to First Claim**: duration = days until first claim, event = claimed
- **Mortality Modeling**: duration = survival time, event = death
- **Customer Lifetime Value**: duration = relationship length, event = churned
- Any scenario with RIGHT-CENSORED observations (subjects not yet experienced the event)

## When NOT to Use
- Standard classification problems without time component
- Regression without censoring
- When you don't have a clear duration and event indicator

## Required Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| model_name | str | Unique name for this model (e.g., 'cox_v1') |
| train_dataset_ref | str | Reference name of the registered training dataset |
| duration_column | str | Column with time until event/censoring (must be positive) |
| event_column | str | Column with event indicator (1=event occurred, 0=censored) |

## Optional Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| description | str | "" | Human-readable model description |
| feature_columns | list[str] | null | Covariates to use. null = all except duration and event |
| model_type | str | "cox" | 'cox' (semi-parametric), 'weibull' (monotonic hazard), 'lognormal' (non-monotonic hazard) |
| penalizer | float | 0.01 | L2 regularization strength. Range: 0.001 to 0.5 |
| l1_ratio | float | 0.0 | Elastic net mixing. 0 = L2 only, 1 = L1 only |
| test_size | float | 0.2 | Fraction for test set |
| random_state | int | 42 | Random seed |

NOTE: Survival analysis uses duration_column and event_column instead of target_column.

## Model Type Selection
- **Cox PH** (default): Semi-parametric, most widely used. Outputs hazard ratios. Assumes proportional hazards.
- **Weibull AFT**: Parametric. Good when hazard is monotonically increasing or decreasing.
- **Log-Normal AFT**: Parametric. Good for non-monotonic hazards (risk increases then decreases).

## Key Metric — Concordance Index (C-index)
- 0.5 = Random prediction (no better than coin flip)
- 0.7-0.8 = Acceptable discrimination
- 0.8+ = Excellent discrimination
- 1.0 = Perfect prediction

## Hazard Ratio Interpretation
- HR = 1.0: No effect on risk
- HR > 1.0: Increases risk (faster event)
- HR < 1.0: Decreases risk (slower event)
- Example: HR = 1.5 means 50% higher risk of event per unit time

## Output Metrics
- **train_concordance / test_concordance**: C-index
- **log_likelihood**: Model fit quality
- **aic / bic**: Model selection criteria (lower = better)
- **coefficients**: With hazard ratios, p-values, and significance

## Feature Engineering Guidance

Survival models estimate time-to-event with CENSORING. Feature engineering must
preserve temporal integrity and respect model assumptions.

### Encoding Strategy
- ONE-HOT encode nominal categoricals with ≤10 values (drop_first=true)
- ORDINAL encode ordered risk categories (e.g., stage I/II/III/IV)
- For high-cardinality, BIN into meaningful groups or use GROUP_AGG

### Transformations
- PASSTHROUGH baseline covariates measured at/before observation start
- CREATE duration features using date_diff between relevant events
- LOG-TRANSFORM heavily skewed continuous covariates
- Consider BINNING continuous covariates into 3-5 quantile groups for non-linear effects

### Temporal Rules (Critical)
- ALWAYS set as_of_constraint for date-derived features
- NEVER include post-baseline data as a predictor (data leakage)
- Duration/time-to-event columns must NOT be used as predictors
- Watch for IMMORTAL TIME BIAS

### Events Per Variable (EPV) Rule
- Require at least 10-15 EVENTS per predictor variable
- E.g., 80 events → use at most 5-8 features
- Censored observations do not count for EPV

### Multicollinearity
- Cox PH is SENSITIVE to multicollinearity
- DROP one from correlated pairs (|r| > 0.7), target VIF < 5

### Feature Selection
- Focus on BASELINE characteristics and static risk factors
- 8-20 features typical
- DROP features that correlate with the event indicator

### What to Avoid
- DO NOT use features encoding whether/when event occurred
- DO NOT use post-baseline measurements as time-fixed covariates
- DO NOT include features measured only on long-surviving subjects
- DO NOT ignore proportional hazards assumption
- DO NOT exceed EPV limit

## Example (Policy Lapse)
```json
{
  "model_name": "lapse_cox_v1",
  "train_dataset_ref": "pipeline_train_features",
  "duration_column": "months_active",
  "event_column": "lapsed",
  "model_type": "cox",
  "description": "Cox model for policy lapse prediction"
}
```
