"""
Kaggle Customer Churn Competition - End-to-End Pipeline

Uses the simple training agent for the full ML pipeline, then applies
the same cleaning + feature engineering to Kaggle test data for submission.

Run: python3 .kaggle/customer_churn/run_kaggle_competition.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "data-tools"))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "models-tools" / "training"))

from utils import get_registered_dataset, register_dataset
from agents.training.agent_simple import invoke_simple_training_agent
from agents.training.steps.cleaning_simple import run_cleaning_simple
from agents.training.steps.feature_engineering_executor import execute_feature_spec
from model_storage import load_model, get_model_info

DATA_DIR = Path(__file__).parent
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = DATA_DIR / "submission.csv"
ARTIFACTS_FILE = DATA_DIR / "pipeline_artifacts.json"

GOAL = """
Predict customer churn (binary classification). Target: Churn (Yes/No).
Your goal is to build the ABSOLUTE BEST model possible — target validation AUC-ROC > 0.99.
Be relentless. Do NOT settle for "good enough." Push every iteration to squeeze out more performance.

TASK: Binary classification. Target: Churn (Yes/No — encode Yes=1, No=0 before training).
METRIC: Area under the ROC curve (AUC-ROC). Maximize it aggressively.

DATASET OVERVIEW (594K training rows, 20 features):
- Numeric: tenure (months), MonthlyCharges, TotalCharges, SeniorCitizen (0/1)
- Categorical (Yes/No): Partner, Dependents, PhoneService, PaperlessBilling
- Categorical (Yes/No/No phone or internet service): MultipleLines, OnlineSecurity,
  OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- Categorical (multi-level): gender (Male/Female), InternetService (DSL/Fiber optic/No),
  Contract (Month-to-month/One year/Two year), PaymentMethod (4 values)
- ID column: id (FORBIDDEN — never use as a feature)

CRITICAL FEATURE ENGINEERING — build ALL of these, they are proven high-impact for churn:
- tenure_group: bin tenure into 0-12, 13-24, 25-36, 37-48, 49-60, 61-72 month buckets
- AvgMonthlyCharge = TotalCharges / (tenure + 1) — average spend per month
- ChargeRatio = MonthlyCharges / (TotalCharges + 1) — recent vs historical spend
- tenure * MonthlyCharges interaction (commitment × spend)
- tenure * Contract interaction (how long they've been on what contract type)
- NumServices: count of active services (OnlineSecurity, OnlineBackup, DeviceProtection,
  TechSupport, StreamingTV, StreamingMovies — count "Yes" values)
- HasNoInternetService: flag for customers with InternetService == "No"
- IsMonthToMonth: binary flag for Contract == "Month-to-month" (strongest churn predictor)
- IsElectronicCheck: binary flag for PaymentMethod == "Electronic check"
- tenure_squared: tenure^2 to capture nonlinear retention curve
- HighSpender: MonthlyCharges > 70 (above-median flag)
- NewCustomer: tenure <= 6 (high-risk early-lifecycle flag)
- Ordinal encode Contract: Month-to-month=0, One year=1, Two year=2
- Ordinal encode InternetService: No=0, DSL=1, Fiber optic=2
- One-hot encode PaymentMethod, gender
- For all Yes/No categoricals, encode Yes=1, No=0
- For "No phone service"/"No internet service" values, encode as 0

IMPORTANT — the Churn target is a STRING ("Yes"/"No"). Convert to 1/0 integer before training.
TotalCharges may have blank/space entries that fail numeric conversion — coerce and fill with 0.

MODEL STRATEGY — be aggressive:
- Try HistGradientBoostingClassifier AND RandomForestClassifier AND LogisticRegression
- Use extensive hyperparameter search: max_iter/n_estimators up to 500-1000, learning_rate
  sweep from 0.01 to 0.3, max_depth 3-12, regularization sweeps
- Run the MAXIMUM number of training iterations allowed
- After each iteration, if AUC < 0.85, redo features with more interactions
- Consider class_weight='balanced' or oversampling since churn rate is ~22%
- Consider stacking or blending if multiple models are close in performance

DO NOT STOP until you have exhausted all improvement opportunities.
""".strip()


def save_artifacts(state: dict):
    """Persist pipeline artifacts so a trained model can be reused without retraining."""
    artifacts = {
        "model_name": state.get("model_weights_path"),
        "feature_spec": state.get("feature_spec"),
        "label_definition": state.get("label_definition"),
        "selected_model": state.get("selected_model"),
        "training_metrics": state.get("training_metrics"),
    }
    with open(ARTIFACTS_FILE, "w") as f:
        json.dump(artifacts, f, indent=2, default=str)
    print(f"  Artifacts saved to {ARTIFACTS_FILE}")


def load_artifacts() -> dict:
    """Load previously saved pipeline artifacts."""
    with open(ARTIFACTS_FILE) as f:
        return json.load(f)


def prepare_test_data(feature_spec: dict, label_def: dict, selected_model: str) -> pd.DataFrame:
    """Clean and feature-engineer the Kaggle test data using the training pipeline's spec."""
    df_test = pd.read_csv(TEST_FILE)
    test_ids = df_test["id"].copy()

    df_test["TotalCharges"] = pd.to_numeric(df_test["TotalCharges"], errors="coerce").fillna(0.0)

    register_dataset("kaggle_test_raw", df_test)

    target_col = label_def.get("target_column", "Churn")

    print("  Cleaning test data...")
    cleaning_result = run_cleaning_simple(
        dataset_ref="kaggle_test_raw",
        goal=GOAL,
        target_col=target_col,
        task_type="classification",
        selected_model=selected_model,
    )
    cleaned_ref = cleaning_result["cleaned_ref"]

    if feature_spec:
        print("  Applying feature engineering...")
        fe_result = execute_feature_spec(
            dataset_ref=cleaned_ref,
            feature_spec=feature_spec,
            target_column=target_col,
            grain=label_def.get("grain", ""),
        )
        predict_ref = fe_result["transformed_dataset_ref"]
        errors = fe_result.get("errors", [])
        if errors:
            print(f"  Feature engineering warnings: {errors}")
    else:
        predict_ref = cleaned_ref

    df_predict = get_registered_dataset(predict_ref)
    X_test = df_predict.drop(columns=[target_col, "id"], errors="ignore")

    return test_ids, X_test


def align_features(X_test: pd.DataFrame, model) -> pd.DataFrame:
    """Align test DataFrame columns to match what the model's preprocessor expects."""
    try:
        expected = list(model.feature_names_in_)
    except AttributeError:
        preprocessor = model.named_steps.get("preprocessor")
        if preprocessor and hasattr(preprocessor, "feature_names_in_"):
            expected = list(preprocessor.feature_names_in_)
        else:
            return X_test

    missing = set(expected) - set(X_test.columns)
    extra = set(X_test.columns) - set(expected)

    if missing:
        print(f"  Adding {len(missing)} missing columns (zeros): {sorted(missing)}")
        for col in missing:
            X_test[col] = 0

    if extra:
        print(f"  Dropping {len(extra)} extra columns: {sorted(extra)}")

    return X_test[expected]


def generate_submission(
    model_name: str,
    feature_spec: dict,
    label_def: dict,
    selected_model: str,
):
    """Load model, transform test data, predict probabilities, and write submission.csv."""
    print(f"\n--- Generating submission with model '{model_name}' ---")

    model = load_model(model_name)
    model_info = get_model_info(model_name)
    print(f"  Model type: {model_info.get('model_type')}")
    print(f"  Features: {len(model_info.get('feature_names', []))}")

    test_ids, X_test = prepare_test_data(feature_spec, label_def, selected_model)
    print(f"  Test shape after feature engineering: {X_test.shape}")

    X_test = align_features(X_test, model)
    print(f"  Test shape after alignment: {X_test.shape}")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        # Find the column for the positive class (1 or "Yes")
        classes = model.classes_
        pos_idx = list(classes).index(1) if 1 in classes else list(classes).index("Yes")
        predictions = probabilities[:, pos_idx]
        print(f"  Using predict_proba (classes: {list(classes)}, positive index: {pos_idx})")
    else:
        predictions = model.predict(X_test).astype(float)
        print("  Warning: model lacks predict_proba, using hard predictions")

    predictions = np.clip(predictions, 0.0, 1.0)

    submission = pd.DataFrame({"id": test_ids, "Churn": predictions})
    submission.to_csv(SUBMISSION_FILE, index=False)

    print(f"\n  Submission: {SUBMISSION_FILE}")
    print(f"  Rows: {len(submission)}")
    print(f"  Probability range: {predictions.min():.4f} - {predictions.max():.4f}")
    print(f"  Mean probability: {predictions.mean():.4f}")
    print(f"  Predicted churn rate (>0.5): {(predictions > 0.5).mean():.2%}")


def run_full_pipeline():
    """Train a model and generate a Kaggle submission."""
    df_train = pd.read_csv(TRAIN_FILE)

    df_train["TotalCharges"] = pd.to_numeric(df_train["TotalCharges"], errors="coerce").fillna(0.0)
    df_train["Churn"] = df_train["Churn"].map({"Yes": 1, "No": 0})

    register_dataset("customer_churn_train", df_train)
    print(f"Registered training data: {df_train.shape[0]} rows x {df_train.shape[1]} cols")
    churn_rate = df_train["Churn"].mean()
    print(f"Churn distribution: {churn_rate:.2%} positive ({df_train['Churn'].sum():,} / {len(df_train):,})")

    print("\n--- Invoking training agent ---")
    state = invoke_simple_training_agent(
        goal=GOAL,
        linked_datasets=["customer_churn_train"],
    )

    model_name = state.get("model_weights_path")
    if not model_name:
        print(f"\nTraining failed: {state.get('error')}")
        return

    metrics = state.get("training_metrics", {})
    print(f"\n--- Training complete ---")
    print(f"  Model: {model_name} ({state.get('selected_model')})")
    print(f"  Iterations: {metrics.get('num_iterations', 0)}")
    for key in ["val_roc_auc", "val_accuracy", "test_roc_auc", "test_accuracy"]:
        val = metrics.get(key)
        if val is not None:
            print(f"  {key}: {val:.4f}")

    save_artifacts(state)

    generate_submission(
        model_name=model_name,
        feature_spec=state.get("feature_spec"),
        label_def=state.get("label_definition", {}),
        selected_model=state.get("selected_model", "supervised"),
    )


def run_with_existing_model(model_name: str):
    """Generate a submission using a previously trained model and saved artifacts."""
    info = get_model_info(model_name)
    if not info:
        print(f"Model '{model_name}' not found in registry.")
        return

    if not ARTIFACTS_FILE.exists():
        print(f"No pipeline artifacts found at {ARTIFACTS_FILE}.")
        print("Run without --model first to train and save artifacts.")
        return

    artifacts = load_artifacts()
    print(f"Loaded artifacts for model '{artifacts.get('model_name')}'")

    generate_submission(
        model_name=model_name,
        feature_spec=artifacts.get("feature_spec"),
        label_def=artifacts.get("label_definition", {}),
        selected_model=artifacts.get("selected_model", "supervised"),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle Customer Churn pipeline")
    parser.add_argument("--model", type=str, help="Reuse an existing trained model")
    parser.add_argument("--dry-run", action="store_true", help="Only preview data")
    args = parser.parse_args()

    if args.dry_run:
        df = pd.read_csv(TRAIN_FILE)
        print(f"Train: {df.shape}")
        print(df["Churn"].value_counts())
        print(f"\nChurn rate: {(df['Churn'] == 'Yes').mean():.2%}")
        df_t = pd.read_csv(TEST_FILE)
        print(f"\nTest: {df_t.shape}")
    elif args.model:
        run_with_existing_model(args.model)
    else:
        run_full_pipeline()
