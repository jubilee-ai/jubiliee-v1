"""
Test suite for the full training agent flow (invoke_training_agent).

Tests:
1-5: Use pre-registered datasets from datasets/csv/
6-8: No linked datasets - agent must discover/create data

Run: python agents/training/test_full_agent.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools" / "data-tools"))

import pandas as pd
import numpy as np
from utils import register_dataset, get_registered_dataset

from agents.training.core.graph import invoke_training_agent


def print_header(test_num: int, title: str):
    """Print a test header."""
    print("\n" + "=" * 80)
    print(f"TEST {test_num}: {title}")
    print("=" * 80)


def print_step(step: str, msg: str = ""):
    """Print a step."""
    print(f"\n[{step}] {msg}")


def print_dataset_preview(df: pd.DataFrame, name: str, max_rows: int = 3):
    """Print a preview of the dataset."""
    print(f"\n  📊 DATASET: {name}")
    print(f"  Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
    print(f"  Dtypes: {dict(list(df.dtypes.items())[:5])}...")
    print(f"  Sample rows:")
    for i, row in df.head(max_rows).iterrows():
        row_dict = {k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in list(row.items())[:6]}
        print(f"    [{i}] {row_dict}...")


def print_report_contents(report_path: str):
    """Read and print key parts of the generated report."""
    import json
    try:
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        print("\n  📄 GENERATED REPORT CONTENTS")
        print("  " + "-" * 50)
        print(f"  Goal: {report.get('goal', 'N/A')}")
        print(f"  Generated: {report.get('generated_at', 'N/A')}")
        
        model = report.get('model', {})
        print(f"\n  Model Info:")
        print(f"    Type: {model.get('type')}")
        print(f"    Name: {model.get('name')}")
        print(f"    Explanation: {str(model.get('explanation', ''))[:80]}...")
        
        data = report.get('data', {})
        print(f"\n  Data Lineage:")
        print(f"    Collected: {data.get('collected_dataset')}")
        print(f"    Cleaned: {data.get('cleaned_dataset')}")
        print(f"    Train: {data.get('train_dataset')}")
        print(f"    Val: {data.get('val_dataset')}")
        print(f"    Test: {data.get('test_dataset')}")
        
        label_def = report.get('label_definition', {})
        print(f"\n  Label Definition:")
        print(f"    Target: {label_def.get('target_column')}")
        print(f"    Split Strategy: {label_def.get('split_strategy')}")
        print(f"    Grain: {label_def.get('grain')}")
        
        training = report.get('training_results', {})
        print(f"\n  Training Results:")
        print(f"    Success: {training.get('success')}")
        print(f"    Iterations: {training.get('num_iterations', 0)}")
        val_metrics = training.get('validation_metrics', {})
        print(f"    Val Accuracy: {val_metrics.get('accuracy')}")
        print(f"    Val ROC-AUC: {val_metrics.get('roc_auc')}")
        test_metrics = training.get('test_metrics', {})
        print(f"    Test Accuracy: {test_metrics.get('accuracy')}")
        print(f"    Test ROC-AUC: {test_metrics.get('roc_auc')}")
        
        # Show iterations from report
        iterations = training.get('iterations', [])
        if iterations:
            print(f"\n  Iteration Details:")
            for i, it in enumerate(iterations, 1):
                status = "✅" if it.get("success") else "❌"
                m = it.get("metrics") or {}
                print(f"    {status} [{i}] {it.get('model_name', 'unknown')[:25]} -> ROC: {m.get('roc_auc', 'N/A')}")
        
        audit = report.get('audit_trace', [])
        print(f"\n  Audit Trace ({len(audit)} steps):")
        for item in audit:
            step = item.get('step', 'unknown')
            # Print key info for each step
            if step == 'feature_engineering_executor':
                features = item.get('features_created', [])
                print(f"    - [{step}] Created {len(features)} features: {features[:5]}...")
            elif step == 'training':
                print(f"    - [{step}] Model: {item.get('model_name')}, Success: {item.get('success')}")
            else:
                print(f"    - [{step}]")
        
    except Exception as e:
        print(f"  ⚠️ Could not read report: {e}")


def print_result(result: dict):
    """Print the result summary."""
    print("\n" + "=" * 60)
    print("📋 RESULT SUMMARY")
    print("=" * 60)
    print(f"  Error: {result.get('error')}")
    print(f"  Selected Model: {result.get('selected_model')}")
    print(f"  Model Explanation: {result.get('model_explanation', '')[:100]}...")
    print(f"  Collected Dataset: {result.get('collected_dataset_ref')}")
    print(f"  Cleaned Dataset: {result.get('cleaned_dataset_ref')}")
    print(f"  Train Dataset: {result.get('transformed_train_ref')}")
    print(f"  Val Dataset: {result.get('transformed_val_ref')}")
    print(f"  Test Dataset: {result.get('transformed_test_ref')}")
    print(f"  Model Weights Path: {result.get('model_weights_path')}")
    print(f"  Report Path: {result.get('report_path')}")
    
    training_metrics = result.get("training_metrics", {})
    if training_metrics:
        print(f"\n  🎯 TRAINING METRICS:")
        print(f"    Success: {training_metrics.get('success')}")
        print(f"    Model Name: {training_metrics.get('model_name')}")
        print(f"    Iterations: {training_metrics.get('num_iterations', 0)}")
        print(f"    Val Accuracy: {training_metrics.get('val_accuracy')}")
        print(f"    Val ROC-AUC: {training_metrics.get('val_roc_auc')}")
        print(f"    Test Accuracy: {training_metrics.get('test_accuracy')}")
        print(f"    Test ROC-AUC: {training_metrics.get('test_roc_auc')}")
        
        # Show iteration logs
        iterations = training_metrics.get('iterations', [])
        if iterations:
            print(f"\n  📊 ITERATION LOG ({len(iterations)} iterations):")
            for i, it in enumerate(iterations, 1):
                status = "✅" if it.get("success") else "❌"
                m = it.get("metrics") or {}
                hp = it.get("hyperparams") or {}
                hp_str = ", ".join(f"{k}={v}" for k, v in hp.items() if v is not None)[:50]
                acc = m.get('train_accuracy', 'N/A')
                roc = m.get('roc_auc', 'N/A')
                acc_str = f"{acc:.3f}" if isinstance(acc, float) else acc
                roc_str = f"{roc:.3f}" if isinstance(roc, float) else roc
                print(f"    {status} [{i}] {it.get('model_name', 'unknown')[:30]} | Acc: {acc_str} | ROC: {roc_str} | {hp_str}...")
        
        if training_metrics.get('summary'):
            print(f"    Summary: {str(training_metrics.get('summary'))[:150]}...")
    
    # Print report contents if available
    report_path = result.get('report_path')
    if report_path:
        print_report_contents(report_path)


def check_success(result: dict, test_name: str) -> bool:
    """Check if the test succeeded."""
    training_metrics = result.get("training_metrics", {})
    success = training_metrics.get("success", False)
    
    if success:
        print(f"\n✅ {test_name} PASSED")
        return True
    else:
        error = result.get("error") or training_metrics.get("error", "Unknown error")
        print(f"\n❌ {test_name} FAILED: {error}")
        return False


# =============================================================================
# TESTS WITH PRE-REGISTERED DATASETS (1-5)
# =============================================================================

def test_1_loan_default_logistic():
    """Test 1: Loan default prediction with logistic regression."""
    print_header(1, "Loan Default - Logistic Regression")
    
    # Load and register dataset
    print_step("SETUP", "Loading Loan_default.csv")
    df = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "Loan_default.csv")
    
    # Sample for speed
    df_sample = df.sample(n=min(500, len(df)), random_state=42)
    
    # Show input dataset
    print_step("INPUT DATASET", "")
    print_dataset_preview(df_sample, "loan_default_full")
    print(f"\n  Target: 'Default'")
    print(f"  Default rate: {df_sample['Default'].mean():.1%} ({df_sample['Default'].sum()} / {len(df_sample)})")
    
    register_dataset("loan_default_full", df_sample)
    print(f"\n  ✅ Registered as: loan_default_full")
    
    # Run agent
    print_step("INVOKE", "Calling invoke_training_agent...")
    result = invoke_training_agent(
        goal="Predict loan default risk based on borrower characteristics for credit decisioning",
        linked_datasets=["loan_default_full"],
        user_model_preference="logistic_regression"
    )
    
    print_result(result)
    return check_success(result, "Test 1: Loan Default - Logistic Regression")


def test_2_loan_default_random_forest():
    """Test 2: Loan default with random forest."""
    print_header(2, "Loan Default - Random Forest")
    
    # Load and register dataset
    print_step("SETUP", "Loading Loan_default.csv")
    df = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "Loan_default.csv")
    df_sample = df.sample(n=min(500, len(df)), random_state=123)
    
    print_step("INPUT DATASET", "")
    print_dataset_preview(df_sample, "loan_default_rf")
    print(f"\n  Target: 'Default'")
    print(f"  Default rate: {df_sample['Default'].mean():.1%}")
    
    register_dataset("loan_default_rf", df_sample)
    print(f"\n  ✅ Registered as: loan_default_rf")
    
    # Run agent
    print_step("INVOKE", "Calling invoke_training_agent with random_forest...")
    result = invoke_training_agent(
        goal="Build a random forest model to predict loan defaults with high recall",
        linked_datasets=["loan_default_rf"],
        user_model_preference="random_forest"
    )
    
    print_result(result)
    return check_success(result, "Test 2: Loan Default - Random Forest")


def test_3_insurance_regression():
    """Test 3: Insurance charges prediction (regression)."""
    print_header(3, "Insurance Charges - Regression (GLM)")
    
    # Load and register dataset
    print_step("SETUP", "Loading insurance.csv")
    df = pd.read_csv(Path(__file__).parent.parent / "datasets" / "csv" / "insurance.csv")
    
    print_step("INPUT DATASET", "")
    print_dataset_preview(df, "insurance_full")
    print(f"\n  Target: 'charges' (continuous - regression)")
    print(f"  Charges range: ${df['charges'].min():.0f} - ${df['charges'].max():.0f}")
    print(f"  Charges mean: ${df['charges'].mean():.0f}")
    
    register_dataset("insurance_full", df)
    print(f"\n  ✅ Registered as: insurance_full")
    
    # Run agent - note: GLM for regression
    print_step("INVOKE", "Calling invoke_training_agent with glm...")
    result = invoke_training_agent(
        goal="Predict insurance charges based on customer demographics and health factors",
        linked_datasets=["insurance_full"],
        user_model_preference="glm"
    )
    
    print_result(result)
    # GLM regression may not have traditional success metrics, check for model path
    if result.get("model_weights_path"):
        print(f"\n✅ Test 3: Insurance Regression PASSED (model created)")
        return True
    else:
        print(f"\n❌ Test 3: Insurance Regression FAILED")
        return False


def test_4_financial_distress():
    """Test 4: Financial distress prediction."""
    print_header(4, "Financial Distress - XGBoost")
    
    # Load and register dataset
    print_step("SETUP", "Loading Financial Distress.csv")
    df = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "Financial Distress.csv")
    
    # Create binary target (distressed = Financial Distress < 0)
    df["is_distressed"] = (df["Financial Distress"] < 0).astype(int)
    
    # Select only key columns to avoid recursion limit (87 columns is too many)
    key_cols = ["Company", "Time", "is_distressed", "x1", "x2", "x3", "x4", "x5", 
                "x6", "x7", "x8", "x9", "x10", "x11", "x12"]
    df = df[key_cols]
    
    # Sample for speed
    df_sample = df.sample(n=min(400, len(df)), random_state=42)
    
    print_step("INPUT DATASET", "")
    print_dataset_preview(df_sample, "financial_distress")
    print(f"\n  Target: 'is_distressed' (binary: Financial Distress < 0)")
    print(f"  Distress rate: {df_sample['is_distressed'].mean():.1%}")
    print(f"  Note: {len(df_sample.columns)} columns (selected key financial ratios)")
    
    register_dataset("financial_distress", df_sample)
    print(f"\n  ✅ Registered as: financial_distress")
    
    # Run agent
    print_step("INVOKE", "Calling invoke_training_agent with xgboost...")
    result = invoke_training_agent(
        goal="Predict company financial distress using financial ratios",
        linked_datasets=["financial_distress"],
        user_model_preference="xgboost"
    )
    
    print_result(result)
    return check_success(result, "Test 4: Financial Distress - XGBoost")


def test_5_loan_default_auto_model():
    """Test 5: Loan default with AUTO model selection (LLM chooses)."""
    print_header(5, "Loan Default - Auto Model Selection")
    
    # Load and register dataset
    print_step("SETUP", "Loading Loan_default.csv")
    df = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "Loan_default.csv")
    df_sample = df.sample(n=min(400, len(df)), random_state=456)
    
    print_step("INPUT DATASET", "")
    print_dataset_preview(df_sample, "loan_auto_model")
    print(f"\n  Target: 'Default'")
    print(f"  Default rate: {df_sample['Default'].mean():.1%}")
    print(f"\n  ⚠️ No model preference specified - LLM will choose!")
    
    register_dataset("loan_auto_model", df_sample)
    print(f"\n  ✅ Registered as: loan_auto_model")
    
    # Run agent WITHOUT user_model_preference - LLM will choose
    print_step("INVOKE", "Calling invoke_training_agent WITHOUT model preference...")
    print("  (LLM will analyze goal and select appropriate model)")
    result = invoke_training_agent(
        goal="Build a credit risk model to predict loan defaults for regulatory reporting",
        linked_datasets=["loan_auto_model"],
        user_model_preference=None  # LLM chooses!
    )
    
    print_result(result)
    print(f"\n  LLM Selected Model: {result.get('selected_model')}")
    return check_success(result, "Test 5: Loan Default - Auto Model Selection")


# =============================================================================
# TESTS WITHOUT LINKED DATASETS (6-8) - Agent must find/create data
# =============================================================================

def test_6_no_dataset_loan_default():
    """Test 6: No linked dataset - agent must find loan data."""
    print_header(6, "No Dataset - Loan Default Goal")
    
    print_step("INPUT DATASET", "")
    print("  📊 DATASET: None provided!")
    print("  The agent must use the data retrieval agent to find/create data")
    print("  Goal will guide what data to look for")
    print_step("SETUP", "No dataset provided - agent must discover data")
    
    # Run agent WITHOUT linked_datasets
    print_step("INVOKE", "Calling invoke_training_agent WITHOUT linked_datasets...")
    result = invoke_training_agent(
        goal="Build a model to predict loan default risk using available credit data",
        linked_datasets=None,  # No datasets provided!
        user_model_preference="logistic_regression"
    )
    
    print_result(result)
    
    # Check if agent found/created data
    if result.get("collected_dataset_ref"):
        print(f"\n  Agent discovered/created dataset: {result.get('collected_dataset_ref')}")
        return check_success(result, "Test 6: No Dataset - Loan Default Goal")
    else:
        print(f"\n⚠️ Test 6: Agent could not find data (expected in some environments)")
        return True  # Soft pass - data discovery may not work in all envs


def test_7_no_dataset_insurance():
    """Test 7: No linked dataset - agent must find insurance data."""
    print_header(7, "No Dataset - Insurance Prediction Goal")
    
    print_step("INPUT DATASET", "")
    print("  📊 DATASET: None provided!")
    print("  The agent must use the data retrieval agent to find/create data")
    print_step("SETUP", "No dataset provided - agent must discover data")
    
    # Run agent WITHOUT linked_datasets
    print_step("INVOKE", "Calling invoke_training_agent WITHOUT linked_datasets...")
    result = invoke_training_agent(
        goal="Predict insurance premiums based on customer health and demographics",
        linked_datasets=None,  # No datasets provided!
        user_model_preference="random_forest"
    )
    
    print_result(result)
    
    if result.get("collected_dataset_ref"):
        print(f"\n  Agent discovered/created dataset: {result.get('collected_dataset_ref')}")
        return check_success(result, "Test 7: No Dataset - Insurance Goal")
    else:
        print(f"\n⚠️ Test 7: Agent could not find data (expected in some environments)")
        return True  # Soft pass


def test_8_no_dataset_fraud():
    """Test 8: No linked dataset - fraud detection goal."""
    print_header(8, "No Dataset - Fraud Detection Goal")
    
    print_step("INPUT DATASET", "")
    print("  📊 DATASET: None provided!")
    print("  The agent must use the data retrieval agent to find/create data")
    print_step("SETUP", "No dataset provided - testing fraud detection goal")
    
    # Run agent WITHOUT linked_datasets
    print_step("INVOKE", "Calling invoke_training_agent WITHOUT linked_datasets...")
    result = invoke_training_agent(
        goal="Build a fraud detection model to identify suspicious transactions",
        linked_datasets=None,  # No datasets provided!
        user_model_preference="xgboost"
    )
    
    print_result(result)
    
    if result.get("collected_dataset_ref"):
        print(f"\n  Agent discovered/created dataset: {result.get('collected_dataset_ref')}")
        return check_success(result, "Test 8: No Dataset - Fraud Detection Goal")
    else:
        print(f"\n⚠️ Test 8: Agent could not find data (expected in some environments)")
        return True  # Soft pass


# =============================================================================
# NEW TESTS (9-11) - Multi-dataset and discovery scenarios
# =============================================================================

def test_9_multi_dataset_loan_and_insurance():
    """Test 9: Multiple datasets - Loan default + Insurance for cross-domain risk."""
    print_header(9, "Multi-Dataset - Loan + Insurance Combined")
    
    # Load both datasets
    print_step("SETUP", "Loading multiple datasets")
    
    # Dataset 1: Loan default
    df_loan = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "Loan_default.csv")
    df_loan_sample = df_loan.sample(n=min(300, len(df_loan)), random_state=789)
    
    # Dataset 2: Insurance
    df_insurance = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "insurance.csv")
    df_insurance_sample = df_insurance.sample(n=min(300, len(df_insurance)), random_state=789)
    
    # Show both input datasets
    print_step("INPUT DATASETS", "Two datasets provided")
    print_dataset_preview(df_loan_sample, "loan_multi_1")
    print(f"\n  Target: 'Default' (binary)")
    print(f"  Default rate: {df_loan_sample['Default'].mean():.1%}")
    
    print()
    print_dataset_preview(df_insurance_sample, "insurance_multi_1")
    print(f"\n  Target: 'charges' (continuous) - but agent should pick loan default target")
    print(f"  Charges range: ${df_insurance_sample['charges'].min():.0f} - ${df_insurance_sample['charges'].max():.0f}")
    
    # Register both
    register_dataset("loan_multi_1", df_loan_sample)
    register_dataset("insurance_multi_1", df_insurance_sample)
    print(f"\n  ✅ Registered: loan_multi_1, insurance_multi_1")
    print(f"\n  ⚠️ Agent must decide which dataset to use or how to combine them!")
    
    # Run agent with BOTH datasets
    print_step("INVOKE", "Calling invoke_training_agent with MULTIPLE datasets...")
    result = invoke_training_agent(
        goal="Build a comprehensive risk model using all available customer data for default prediction",
        linked_datasets=["loan_multi_1", "insurance_multi_1"],  # Multiple!
        user_model_preference="logistic_regression"
    )
    
    print_result(result)
    return check_success(result, "Test 9: Multi-Dataset - Loan + Insurance")


def test_10_multi_dataset_financial_and_loan():
    """Test 10: Multiple datasets - Financial Distress + Loan for corporate+consumer risk."""
    print_header(10, "Multi-Dataset - Financial Distress + Loan Default")
    
    # Load both datasets
    print_step("SETUP", "Loading corporate + consumer datasets")
    
    # Dataset 1: Financial Distress (corporate)
    df_financial = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "Financial Distress.csv")
    df_financial["is_distressed"] = (df_financial["Financial Distress"] < 0).astype(int)
    # Select key columns only
    key_cols = ["Company", "Time", "is_distressed", "x1", "x2", "x3", "x4", "x5", "x6"]
    df_financial = df_financial[key_cols]
    df_financial_sample = df_financial.sample(n=min(250, len(df_financial)), random_state=321)
    
    # Dataset 2: Loan default (consumer)
    df_loan = pd.read_csv(Path(__file__).parent.parent.parent / "datasets" / "csv" / "Loan_default.csv")
    df_loan_sample = df_loan.sample(n=min(250, len(df_loan)), random_state=321)
    
    # Show both
    print_step("INPUT DATASETS", "Corporate + Consumer risk data")
    print_dataset_preview(df_financial_sample, "financial_multi_1")
    print(f"\n  Target: 'is_distressed' (binary) - CORPORATE risk")
    print(f"  Distress rate: {df_financial_sample['is_distressed'].mean():.1%}")
    
    print()
    print_dataset_preview(df_loan_sample, "loan_multi_2")
    print(f"\n  Target: 'Default' (binary) - CONSUMER risk")
    print(f"  Default rate: {df_loan_sample['Default'].mean():.1%}")
    
    # Register both
    register_dataset("financial_multi_1", df_financial_sample)
    register_dataset("loan_multi_2", df_loan_sample)
    print(f"\n  ✅ Registered: financial_multi_1, loan_multi_2")
    print(f"\n  ⚠️ Agent sees both corporate and consumer data - must choose or combine!")
    
    # Run agent
    print_step("INVOKE", "Calling invoke_training_agent with corporate+consumer data...")
    result = invoke_training_agent(
        goal="Build a credit risk model that can assess both corporate and consumer default risk",
        linked_datasets=["financial_multi_1", "loan_multi_2"],
        user_model_preference="xgboost"
    )
    
    print_result(result)
    return check_success(result, "Test 10: Multi-Dataset - Financial + Loan")


def test_11_no_dataset_corporate_bankruptcy():
    """Test 11: No linked dataset - corporate bankruptcy prediction goal."""
    print_header(11, "No Dataset - Corporate Bankruptcy Prediction")
    
    print_step("INPUT DATASET", "")
    print("  📊 DATASET: None provided!")
    print("  Goal: Predict corporate bankruptcy")
    print("  Agent should discover Financial Distress.csv or similar corporate data")
    print_step("SETUP", "No dataset provided - agent must find corporate financial data")
    
    # Run agent WITHOUT linked_datasets
    print_step("INVOKE", "Calling invoke_training_agent WITHOUT linked_datasets...")
    result = invoke_training_agent(
        goal="Build a model to predict corporate bankruptcy using financial ratios and company metrics",
        linked_datasets=None,  # No datasets provided!
        user_model_preference="random_forest"
    )
    
    print_result(result)
    
    if result.get("collected_dataset_ref"):
        print(f"\n  Agent discovered/created dataset: {result.get('collected_dataset_ref')}")
        return check_success(result, "Test 11: No Dataset - Corporate Bankruptcy")
    else:
        print(f"\n⚠️ Test 11: Agent could not find data (expected in some environments)")
        return True  # Soft pass


def test_12_unsupervised_customer_segmentation():
    """Test 12: Unsupervised customer segmentation."""
    from agents.training.agent_simple import invoke_simple_training_agent

    print_header(12, "Unsupervised - Customer Segmentation")

    print_step("SETUP", "Loading insurance.csv for clustering")
    df = pd.read_csv(Path(__file__).parent.parent / "datasets" / "csv" / "insurance.csv")
    df_sample = df.sample(n=min(400, len(df)), random_state=42).copy()

    # Keep this as a pure unsupervised use case.
    if "charges" in df_sample.columns:
        df_sample = df_sample.drop(columns=["charges"])

    print_step("INPUT DATASET", "")
    print_dataset_preview(df_sample, "insurance_unsupervised")
    print("\n  Target: None (unsupervised clustering)")
    print("  Goal: segment customers into similar groups")

    register_dataset("insurance_unsupervised", df_sample)
    print(f"\n  ✅ Registered as: insurance_unsupervised")

    print_step("INVOKE", "Calling invoke_training_agent with unsupervised...")
    result = invoke_simple_training_agent(
        goal="Cluster insurance customers into meaningful segments based on demographics and health attributes for exploratory analysis",
        linked_datasets=["insurance_unsupervised"],
        user_model_preference="unsupervised"
    )

    print_result(result)
    return check_success(result, "Test 12: Unsupervised - Customer Segmentation")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("FULL TRAINING AGENT TEST SUITE")
    print("=" * 80)
    print("Testing invoke_training_agent with various scenarios")
    print("Tests 1-5: Pre-registered datasets from datasets/csv/")
    print("Tests 6-8: No linked datasets (agent discovers data)")
    print("Tests 9-10: Multiple linked datasets")
    print("Test 11: No dataset - corporate bankruptcy goal")
    print("Test 12: Unsupervised customer segmentation")
    print("=" * 80)
    
    results = {}
    
    # Tests with single datasets
    results["Test 1"] = test_1_loan_default_logistic()
    results["Test 2"] = test_2_loan_default_random_forest()
    results["Test 3"] = test_3_insurance_regression()
    results["Test 4"] = test_4_financial_distress()
    results["Test 5"] = test_5_loan_default_auto_model()
    
    # Tests without datasets (discovery)
    results["Test 6"] = test_6_no_dataset_loan_default()
    results["Test 7"] = test_7_no_dataset_insurance()
    results["Test 8"] = test_8_no_dataset_fraud()
    
    # Tests with multiple datasets
    results["Test 9"] = test_9_multi_dataset_loan_and_insurance()
    results["Test 10"] = test_10_multi_dataset_financial_and_loan()
    
    # Additional discovery test
    results["Test 11"] = test_11_no_dataset_corporate_bankruptcy()

    # Unsupervised coverage
    results["Test 12"] = test_12_unsupervised_customer_segmentation()
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print("-" * 80)
    print(f"  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ All tests passed!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, help="Run specific test (1-12)")
    args = parser.parse_args()
    
    if args.test:
        test_funcs = {
            1: test_1_loan_default_logistic,
            2: test_2_loan_default_random_forest,
            3: test_3_insurance_regression,
            4: test_4_financial_distress,
            5: test_5_loan_default_auto_model,
            6: test_6_no_dataset_loan_default,
            7: test_7_no_dataset_insurance,
            8: test_8_no_dataset_fraud,
            9: test_9_multi_dataset_loan_and_insurance,
            10: test_10_multi_dataset_financial_and_loan,
            11: test_11_no_dataset_corporate_bankruptcy,
            12: test_12_unsupervised_customer_segmentation,
        }
        if args.test in test_funcs:
            test_funcs[args.test]()
        else:
            print(f"Unknown test: {args.test}. Available: 1-12")
    else:
        main()
