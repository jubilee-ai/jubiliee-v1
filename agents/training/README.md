# ML Model Training Agent

## Entry Point
```python
from agents.training.agent import invoke_training_agent

result = invoke_training_agent(
    goal="Build a model to predict loan default risk",
    linked_datasets=["my_dataset"],  # Optional pre-registered datasets
    user_model_preference="logistic_regression"  # Optional: glm, logistic_regression, random_forest, survival_analysis, xgboost
)
```

## Input
- **goal**: Goal/objective from user (required)
- **linked_datasets**: Optional list of pre-registered dataset references
- **user_model_preference**: Optional model type (if not provided, LLM selects based on goal)

## Output (TrainingAgentState)
- **audit_trace**: Full lineage trace of all steps
- **explanations**: List of explanations of what was done at each step
- **model_weights_path**: Name of the trained model in registry
- **report_path**: Path to generated JSON report
- **training_metrics**: Validation and test metrics

## Architecture
1. Select model
- Based on the goal
- If user specifies a model than we will use that
- Return an explanation as well if not given
--> User can comment and regenerate here (3 total regens)

2. Data collection agent (tools for sql + connecting to DBs, vector stores, CRMs, web...)
**Tools:** same as data-retrieval agent
- Reuse our data-retrieval agent from ./data-retrieval
- Goes and finds the right data and puts together a dataset
- Uses the datasets in here if provided
- Iterative agent 

3. Cleaning + Standardization agent (Iterative) (dataset_ref, goal) --> updated_dataset_ref
**Tools:** eda_report + data_validation + clean_ops (clip, drop_nulls, fill_null, impute, regex_replace, replace_values) + row_ops (dedupe, filter, limit, sample, sort) + column_ops (add_column, cast, drop, parse_datetime, rename, select) + reshape_ops (pivot, union, unpivot)
- Has access to cleaning transformation tools in tools/data-tools/transformations + eda_report + validation
- Applies a list of transformations to clean the dataset and make it ready for training
    i. Run the eda_report and analysis (+ others if we need)
    ii. LLM reviews it and either:
        1) Calls the necessary transformations from clean_ops, row_ops, col_ops and reshape_ops (less important)
        2) Exits and registers the new dataset --> Ready for feature engineering!
    iii. Apply transformations
    iv. Iterate back to i.

TODO: DECIDE WHETHER SUPERVISED OR NOT (3.5 IS BASED ON THIS)

3.5. Label + Split Definition (One-time, Human-driven) --> SKIPPER IF NOT RELEVANT
- Agent presents the dataset schema and asks user to define:
    1. Target column: "Which column is the prediction target?"
    2. Prediction horizon: "How far into the future are we predicting?" (e.g., 30 days, 90 days)
    3. Grain: "What does one row represent?" (one customer? one policy? one transaction?)
    4. As-of cutoff: "Which column represents the observation/prediction timestamp?"
    5. Split strategy: "How should we split? (random / time-based / entity-based)"
    6. Forbidden columns: "Which columns would NOT be available at prediction time?" (agent lists all columns, user marks forbidden ones)
- **User can choose**:
    - **Manual mode**: Answer each question directly or some
    - **Auto-fill mode**: Agent infers answers based on goal + schema + heuristics, user reviews and confirms/corrects
- If user is unsure on any question, agent can explain trade-offs
- Human confirms final answers → definitions are LOCKED
- **SPLITTING HAPPENS HERE**:
    ```python
    # 1. Define split configuration
    label_def = run_label_split_definition(dataset_ref, goal, ...)
    
    # 2. Compute split indices based on strategy (random/time/entity)
    split_indices = compute_split_indices(df, label_def, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # 3. Apply split and register datasets
    train_df, val_df, test_df = apply_split(df, split_indices)
    register_dataset("my_dataset_train", train_df)
    register_dataset("my_dataset_val", val_df)
    register_dataset("my_dataset_test", test_df)
    ```
- Locked definitions + split dataset refs are passed to all downstream steps (4, 5, 7)
- This prevents data leakage in feature selection and engineering

4. Feature Selection & Specification Agent (train_ref, goal, label_definition) --> feature_spec
- **Inputs from 3.5**: train_ref (TRAINING DATA ONLY), target column, as-of cutoff, forbidden columns list
**Tools:** concentration_analysis + correlation_matrix + data_validation + distribution_analysis + eda_report + feature_diagnostics + group_summary + trend_analysis
- **CRITICAL**: All analysis runs on TRAINING DATA ONLY to prevent leakage
- The goal here is to choose which features to use AND specify how to build them
- Excludes forbidden columns (future-leaking features) identified in 3.5
- We have access to all analysis tools in tools/data-tools/analysis for stat analysis to decide
- Binded to the different tools and iterates until it feels it knows and outputs the feature_spec
    ```python
    result = run_feature_engineering_simple(
        train_ref="my_dataset_train",  # Analysis on train only!
        val_ref="my_dataset_val",       # Passed for info, not analyzed
        test_ref="my_dataset_test",     # Passed for info, not analyzed
        goal=goal,
        target_column=label_def["target_column"],
        grain=label_def["grain"],
        forbidden_columns=label_def["forbidden_columns"],
    )
    feature_spec = result["feature_spec"]
    ```
--> Human in the loop here to see the trace, stats, analysis and verify or comment on the features chosen
--> ACTUALLY USING WHERE WE RUN ALL TOOLS AND OUTPUT (could get worse once too many tools) --> prob good tho

- **Output: feature_spec** (structured contract for step 5):
    ```json
    {
      "features": [
        {
          "name": "claim_count_90d",
          "formula": "count(claims) where claim_date > as_of - 90d",
          "source_tables": ["claims", "policies"],
          "window": "90 days",
          "grain": "policy_id",
          "as_of_constraint": "claim_date < as_of_date",
          "encoding": "numeric"
        },
        {
          "name": "region_encoded",
          "formula": "one_hot(region)",
          "source_tables": ["policies"],
          "window": null,
          "grain": "policy_id",
          "as_of_constraint": null,
          "encoding": "one_hot"
        }
      ]
    }
    ```
- Each feature includes: name, formula/logic, source tables, windows, grain, as-of constraints, encoding strategy
- This spec is reviewable by humans before execution
TODO: Look into multi agent
- Can we spin up subagents to assess what are trying to evaluate and eahc subagent tries to select 1 feature to select it

5. Feature Engineering Executor (feature_spec, train/val/test refs, label_definition) --> Transformed datasets
- **Inputs**: feature_spec from step 4, train_ref/val_ref/test_ref from 3.5, as-of cutoff, grain
- **Tools**: feature_ops, agg_ops, row_ops, column_ops
- **This is a DETERMINISTIC executor** (no LLM reasoning loop):
    i. Parse the feature_spec
    ii. For each feature in spec:
        - **FIT on training data only** (compute bin edges, encoding categories, aggregation stats)
        - **TRANSFORM all three sets** using the fitted parameters
    iii. Apply temporal constraints (respecting as-of dates from spec to avoid leakage)
    iv. Run feature_diagnostics tests to validate
    v. If validation fails → return error to step 4 for spec revision
    vi. If validation passes → register transformed train/val/test datasets
    ```python
    result = execute_feature_spec_split(
        train_ref="my_dataset_train",
        val_ref="my_dataset_val",
        test_ref="my_dataset_test",
        feature_spec=feature_spec,
        target_column=label_def["target_column"],
        grain=label_def["grain"],
    )
    # result["train_ref"] = "my_dataset_train_features"
    # result["val_ref"] = "my_dataset_val_features"
    # result["test_ref"] = "my_dataset_test_features"
    ```
- No iterative LLM decision-making; execution follows the spec exactly
- Failures are spec failures, not execution failures
- **CRITICAL**: Fit-on-train prevents data leakage from val/test into feature computation → blame is clear

6. Human Confirmation Checkpoint (`human_confirmation` in `agent.py`)
- Shows full audit trace of steps 1-5
- Displays: goal, selected model, target column, train/val/test dataset refs
- **Automated runs**: Auto-confirms (for testing)
- **Production**: Would await user input to proceed or abort

7. Iterative Training Subagent (`run_training_agent` in `training.py`)
- **Inputs from 5**: train_ref, val_ref, test_ref (already transformed with features)
- **Inputs from 3.5**: label_definition (target_column, etc.)
- **Tools:** `sklearn_logistic_regression`, `sklearn_random_forest`, `xgboost_train`, `sklearn_glm`, `survival_analysis`, `evaluate_model`, `get_model_info`, `list_trained_models`

```python
from agents.training.training import run_training_agent

result = run_training_agent(
    train_ref="my_dataset_train_features",
    val_ref="my_dataset_val_features", 
    test_ref="my_dataset_test_features",
    target_column="default",
    selected_model="logistic_regression",
    goal="Predict loan default",
    model_name="my_model",
    max_iterations=3,
)
```

**Iteration Loop (max 3 iterations by default):**
1. LLM analyzes data info (class distribution, features) and decides hyperparameters
2. Train model using `train_dataset_ref` parameter (NOT raw data)
3. Evaluate on validation using `evaluate_model` tool
4. LLM analyzes validation metrics:
   - If ROC-AUC >= 0.75 AND accuracy reasonable → proceed to test
   - If poor metrics → iterate with different hyperparameters
5. Final evaluation on test set
6. Output model name (saved in registry) + metrics

**Key Implementation Details:**
- All tools use `dataset_ref` parameters (no raw data passed)
- Models train on ALL provided training data (no internal splits)
- Validation/test evaluation via `evaluate_model` tool
- Each iteration uses unique model name (model_v1, model_v2, model_v3)
- Agent explains reasoning for hyperparameter choices
--> TODO: Down the line we want to think of it as being able to spin up multiple 'training' agents in parallel
--> Essentially running x experiments at the same time and then testing them against each other (e.g. different parameters, types of models, learning rates...). Where each also will reflect on what went wrong and try to course correct.

8. Generate Report (`generate_report` in `agent.py`)
- Generates JSON report saved to `trained_models/{model_name}_report.json`
- Includes: goal, model info, data lineage, label definition, training metrics, full audit trace
- Report accompanies the model weights file for reproducibility

---

## Available Models

| Model | Tool Name | Use Case |
|-------|-----------|----------|
| Logistic Regression | `sklearn_logistic_regression` | Binary/multiclass classification, interpretable |
| Random Forest | `sklearn_random_forest` | Classification/regression, feature importance |
| XGBoost | `xgboost_train` | High-performance tabular data |
| GLM | `sklearn_glm` | Poisson/Gamma/Tweedie regression |
| Survival Analysis | `survival_analysis` | Time-to-event prediction with censoring |

## Test Status

Tests in `tests/test_training_agent.py`:
- ✅ Basic Training Flow (logistic regression)
- ✅ Imbalanced Data Handling (class_weight='balanced')
- ✅ Random Forest Training
- ✅ XGBoost Training
- ✅ Model Saved and Usable
- ✅ Iteration Behavior (poor metrics trigger retry)
- ✅ Full Pipeline Integration

---

NEXT STEP IMPROVEMENT IDEAS:
- All todos above (e.g. taking multiple approaches in parallel)

- Few shot prompts for each tool

- data retrieval on the internet to build datasets
--> Kaggle MCP
--> Google datasets MCP

- Testing agents which try to test one part of decisions and fix it
--> mock in more depth how people do it

- optimize context engineering
--> make sure all the importanr context makes it in at each turn

- ADD MORE TOOLS
--> we're missing a ton of important ones for certain use cases
--> text to sql helps this but can get better

- unsupervised learning and models and optimize process for it
--> keep the system dynamic and still one core system


- neural nets and deep learning

- computer vision

- generating knowledge graphs of what has led to best models
--> This approach worked best for this task...
--> COULD BE HUGE for accuracy and efficient

- SFT

- RL apis

- NLP...


