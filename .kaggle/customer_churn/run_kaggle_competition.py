"""
Kaggle Customer Churn Competition - Full Pipeline

This script:
1. Registers the Kaggle training data
2. Invokes the simple training agent (LLM chooses model + features)
3. Stores the resulting model
4. Applies the same feature engineering to test data and generates predictions
5. Creates a submission.csv file for Kaggle

Evaluation: ROC AUC
Submission: id, Churn (probability)

Run: python .kaggle/customer_churn/run_kaggle_competition.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "data-tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "models-tools" / "training"))

from utils import register_dataset, get_registered_dataset
from model_storage import load_model, get_model_info
from agents.training.agent_simple import create_simple_training_agent
from agents.training.steps.feature_engineering_executor import execute_feature_spec_split


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(__file__).parent
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = DATA_DIR / "submission.csv"

USE_EXISTING_MODEL = None  # e.g., "supervised_1742096123"

GOAL = """
MAXIMIZE ROC AUC for binary customer churn prediction. The target is 'Churn' (Yes/No).
This is a Kaggle competition — every 0.001 improvement in ROC AUC matters. Push for the
highest possible score. Do NOT settle for a "good enough" model.

EVALUATION: Area under the ROC curve. The model MUST output well-calibrated probabilities.
Probability ranking quality is everything — focus on separating churners from non-churners
in predicted probability space.

DATA OVERVIEW (594K training rows, ~22.5% churn rate — moderate class imbalance):
- Demographics: gender, SeniorCitizen, Partner, Dependents
- Account: tenure (months), Contract (Month-to-month / One year / Two year),
  PaperlessBilling, PaymentMethod
- Services: PhoneService, MultipleLines, InternetService, OnlineSecurity,
  OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Charges: MonthlyCharges, TotalCharges (may have blank strings — convert to numeric)

CRITICAL FEATURE ENGINEERING (do ALL of these):
- tenure is the #1 predictor — create polynomial features (tenure^2, tenure^3) and
  interaction terms (tenure * Contract, tenure * InternetService, tenure * MonthlyCharges)
- TotalCharges / tenure = average monthly spend (handle tenure=0 → use MonthlyCharges)
- MonthlyCharges / TotalCharges ratio (recency-weighted spending signal)
- Number of premium services subscribed (count of Yes across OnlineSecurity, OnlineBackup,
  DeviceProtection, TechSupport, StreamingTV, StreamingMovies)
- Binary flags: has_internet, has_phone, has_any_protection (OnlineSecurity OR OnlineBackup
  OR DeviceProtection), has_streaming (StreamingTV OR StreamingMovies)
- Contract risk score: Month-to-month=2, One year=1, Two year=0 — multiply by tenure inverse
- Payment method risk: Electronic check is high-churn — create is_electronic_check flag
- Tenure buckets (0-6, 6-12, 12-24, 24-48, 48-72) for non-linear capture
- SeniorCitizen * MonthlyCharges interaction
- Encode 'No internet service' and 'No phone service' as 'No' (they mean the same thing
  for churn prediction)

MODEL STRATEGY — aim for ROC AUC > 0.92:
1. Use HistGradientBoostingClassifier or XGBoost as the primary model — these dominate
   tabular classification on Kaggle
2. Tune aggressively: learning_rate (0.01-0.1), max_depth (3-8), min_samples_leaf (20-100),
   max_iter/n_estimators (300-1000), l2_regularization (0.01-1.0)
3. Use class_weight='balanced' or scale_pos_weight to handle the 22.5% churn imbalance
4. Cross-validate with stratified k-fold to ensure stable AUC estimates
5. If time permits, try a second model (LogisticRegression with engineered features) and
   see if an ensemble or blending improves AUC

HYPERPARAMETER TUNING GUIDANCE:
- For gradient boosting: start with learning_rate=0.05, max_depth=5, then narrow
- More iterations with lower learning rate generally improves AUC
- Regularization prevents overfitting on this large dataset
- Do NOT use default hyperparameters — they are almost never optimal for competition scores
""".strip()


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_step(step: str, msg: str = ""):
    print(f"\n[{step}] {msg}")


def preview_dataframe(df: pd.DataFrame, name: str, max_rows: int = 3):
    print(f"\n  {name}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    print(f"  Dtypes summary: {df.dtypes.value_counts().to_dict()}")
    if max_rows > 0:
        print(f"  First {max_rows} rows:")
        print(df.head(max_rows).to_string(max_cols=8))


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_kaggle_pipeline(use_existing_model: str = None):
    """Run the full Kaggle Customer Churn competition pipeline."""

    print_header("KAGGLE CUSTOMER CHURN COMPETITION")
    print("  Goal: Predict probability of customer churn")
    print("  Metric: ROC AUC")

    model_name = use_existing_model
    state = None

    if use_existing_model:
        print_header("USING EXISTING MODEL")
        print(f"  Model: {use_existing_model}")

        model_info = get_model_info(model_name)
        if not model_info:
            print(f"Model '{model_name}' not found in registry!")
            return None

        print(f"  Type: {model_info.get('model_type')}")
        print(f"  Target: {model_info.get('target_column')}")
        print(f"  Features: {len(model_info.get('feature_names', []))}")

    else:
        # -----------------------------------------------------------------
        # STEP 1: Load and Register Training Data
        # -----------------------------------------------------------------
        print_header("STEP 1: Load and Register Training Data")

        if not TRAIN_FILE.exists():
            print(f"Training file not found: {TRAIN_FILE}")
            return None

        df_train = pd.read_csv(TRAIN_FILE)
        print_step("LOADED", f"Training data from {TRAIN_FILE}")
        preview_dataframe(df_train, "Training Data")

        # Convert Churn Yes/No → 1/0
        print_step("TRANSFORM", "Converting Churn Yes/No to 1/0")
        df_train["Churn"] = df_train["Churn"].map({"Yes": 1, "No": 0})
        print(f"  Churn distribution:\n{df_train['Churn'].value_counts().to_string()}")
        print(f"  Churn rate: {df_train['Churn'].mean():.2%}")

        # Convert TotalCharges to numeric (may have blanks)
        df_train["TotalCharges"] = pd.to_numeric(df_train["TotalCharges"], errors="coerce")

        # Drop the id column for training
        df_train = df_train.drop(columns=["id"])

        register_dataset("customer_churn_train", df_train)
        print_step("REGISTERED", "Dataset registered as 'customer_churn_train'")

        # -----------------------------------------------------------------
        # STEP 2: Invoke Training Agent
        # -----------------------------------------------------------------
        print_header("STEP 2: Invoke Training Agent")

        print_step("GOAL", f"  {GOAL[:200]}...")
        print_step("INVOKE", "Calling create_simple_training_agent()...")
        print("\n" + "-" * 80)

        agent, state = create_simple_training_agent(
            goal=GOAL,
            linked_datasets=["customer_churn_train"],
            user_model_preference=None,
            hitl=False,
        )

        result = agent.invoke(
            {"messages": [{"role": "user", "content": GOAL}]},
        )

        print("-" * 80)

        # -----------------------------------------------------------------
        # STEP 3: Review Results
        # -----------------------------------------------------------------
        print_header("STEP 3: Training Results")

        print_step("SUMMARY", "")
        print(f"  Selected Model: {state.get('selected_model')}")
        print(f"  Model Explanation: {state.get('model_explanation', '')[:150]}...")
        print(f"  Collected Dataset: {state.get('collected_dataset_ref')}")
        print(f"  Cleaned Dataset: {state.get('cleaned_dataset_ref')}")
        print(f"  Train Dataset: {state.get('transformed_train_ref')}")
        print(f"  Model Weights: {state.get('model_weights_path')}")
        print(f"  Report Path: {state.get('report_path')}")

        training_metrics = state.get("training_metrics", {})
        if training_metrics:
            print_step("METRICS", "")
            print(f"  Success: {training_metrics.get('success')}")
            print(f"  Model Name: {training_metrics.get('model_name')}")
            print(f"  Iterations: {training_metrics.get('num_iterations', 0)}")
            for key in ["val_accuracy", "val_roc_auc", "test_accuracy", "test_roc_auc"]:
                v = training_metrics.get(key)
                if v is not None:
                    print(f"  {key}: {v:.4f}")

        if state.get("error"):
            print(f"\nError: {state.get('error')}")
            return state

        model_name = state.get("model_weights_path")
        if not model_name:
            print("\nNo model was trained!")
            return state

    # -----------------------------------------------------------------
    # STEP 4: Prepare Test Data with Same Feature Engineering
    # -----------------------------------------------------------------
    print_header("STEP 4: Prepare Test Data")

    if not TEST_FILE.exists():
        print(f"Test file not found: {TEST_FILE}")
        return state

    df_test_raw = pd.read_csv(TEST_FILE)
    print_step("LOADED", f"Test data from {TEST_FILE}")
    preview_dataframe(df_test_raw, "Test Data (Raw)")

    test_ids = df_test_raw["id"].copy()

    # Clean test data the same way as training
    df_test = df_test_raw.drop(columns=["id"]).copy()
    df_test["TotalCharges"] = pd.to_numeric(df_test["TotalCharges"], errors="coerce")

    # Load the trained model
    print_step("LOAD MODEL", f"Loading model '{model_name}'")
    try:
        model = load_model(model_name)
        model_info = get_model_info(model_name)
        print(f"  Model type: {model_info.get('model_type')}")
        print(f"  Features: {len(model_info.get('feature_names', []))}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return state

    feature_names = model_info.get("feature_names", [])

    # Try to apply the same feature engineering via feature_spec if available
    if state and state.get("feature_spec"):
        print_step("FEATURE ENGINEERING", "Applying same feature spec to test data")
        feature_spec = state["feature_spec"]
        label_def = state.get("label_definition", {})
        target_column = label_def.get("target_column", "Churn")

        # Add a dummy target column so the executor can keep it
        df_test[target_column] = 0

        test_fe_ref = "kaggle_test_fe_input"
        register_dataset(test_fe_ref, df_test)

        # Use the training data as train_ref (needed for fit-on-train ops)
        train_ref_for_fe = state.get("train_dataset_ref") or state.get("cleaned_dataset_ref")
        fe_result = execute_feature_spec_split(
            train_ref=train_ref_for_fe,
            val_ref=None,
            test_ref=test_fe_ref,
            feature_spec=feature_spec,
            target_column=target_column,
            grain=label_def.get("grain", ""),
            as_of_cutoff=label_def.get("as_of_cutoff"),
        )

        transformed_test_ref = fe_result.get("test_ref")
        if transformed_test_ref:
            df_test_transformed = get_registered_dataset(transformed_test_ref)
            if df_test_transformed is not None:
                # Drop the dummy target
                df_test_transformed = df_test_transformed.drop(
                    columns=[target_column], errors="ignore"
                )
                X_test = df_test_transformed
                print(f"  Feature-engineered test shape: {X_test.shape}")
            else:
                print("  Warning: Could not load transformed test data, falling back to raw")
                X_test = df_test.drop(columns=[target_column], errors="ignore")
        else:
            print("  Warning: Feature engineering returned no test ref, falling back to raw")
            X_test = df_test.drop(columns=["Churn"], errors="ignore")
    else:
        print_step("FEATURE ENGINEERING", "No feature spec in state, using raw columns")
        # Strip prefixes from feature_names to get expected raw columns
        raw_cols = set()
        for f in feature_names:
            if f.startswith("num__"):
                raw_cols.add(f[5:])
            elif f.startswith("cat__"):
                raw_cols.add(f[5:])
            else:
                raw_cols.add(f)
        # Keep only columns that exist in test data
        available = [c for c in raw_cols if c in df_test.columns]
        X_test = df_test[available]
        print(f"  Using {len(available)} raw columns")

    print(f"  Final test shape: {X_test.shape}")

    # -----------------------------------------------------------------
    # STEP 5: Generate Predictions (probabilities)
    # -----------------------------------------------------------------
    print_header("STEP 5: Generate Predictions")

    print_step("PREDICT", "Generating churn probabilities...")

    try:
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_test)
            # Get probability of the positive class (Churn=1)
            if probas.shape[1] == 2:
                predictions = probas[:, 1]
            else:
                predictions = probas[:, -1]
            print(f"  Generated {len(predictions)} probability predictions")
            print(f"  Probability range: {predictions.min():.4f} to {predictions.max():.4f}")
            print(f"  Mean predicted churn probability: {predictions.mean():.4f}")
        else:
            predictions = model.predict(X_test)
            print(f"  Warning: Model doesn't support predict_proba, using predict()")
            print(f"  Generated {len(predictions)} predictions")
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return state

    # -----------------------------------------------------------------
    # STEP 6: Create Submission File
    # -----------------------------------------------------------------
    print_header("STEP 6: Create Submission File")

    submission = pd.DataFrame({
        "id": test_ids,
        "Churn": predictions,
    })

    submission.to_csv(SUBMISSION_FILE, index=False)
    print_step("SAVED", f"Submission saved to {SUBMISSION_FILE}")

    preview_dataframe(submission, "Submission Preview", max_rows=5)

    print(f"\n  File: {SUBMISSION_FILE}")
    print(f"  Rows: {len(submission)}")
    print(f"  Ready to upload to Kaggle!")

    # -----------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------
    print_header("PIPELINE COMPLETE")

    print(f"""
  Summary:
     - Test samples: {len(df_test_raw)}
     - Model: {model_name}
     - Submission: {SUBMISSION_FILE}

  Next Steps:
     1. Upload {SUBMISSION_FILE.name} to the Kaggle competition
     2. Check your ROC AUC score!
""")

    return state


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Kaggle Customer Churn competition pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Only load data, don't train")
    parser.add_argument("--model", type=str, help="Use existing model instead of training new")
    args = parser.parse_args()

    if args.dry_run:
        print_header("DRY RUN - Data Loading Only")

        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)

        preview_dataframe(df_train, "Training Data")
        preview_dataframe(df_test, "Test Data")

        print(f"\n  Churn distribution:")
        print(df_train["Churn"].value_counts())
    else:
        result = run_kaggle_pipeline(use_existing_model=args.model)

        if result is None:
            print("\nPipeline failed!")
        elif isinstance(result, dict) and result.get("training_metrics", {}).get("success"):
            print("\nPipeline completed successfully!")
        else:
            print("\nPipeline completed - check output above")
