"""
Verbose Tests for Training Agent (Step 7)

Shows detailed inputs and outputs for each step.
Run with: python3 tests/test_training_agent_verbose.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "models-tools" / "training"))

import numpy as np
import pandas as pd
from utils import clear_registry, register_dataset, get_registered_dataset


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    """Print a subsection header."""
    print(f"\n[{title}]")
    print("-" * 50)


def print_dataframe_info(name, df):
    """Print detailed info about a DataFrame."""
    print(f"  {name}:")
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Dtypes: {dict(df.dtypes)}")
    if 'default' in df.columns:
        print(f"    Target distribution: {df['default'].value_counts().to_dict()}")
    print(f"    Sample rows:")
    for i, row in df.head(3).iterrows():
        print(f"      {i}: {dict(row)}")


# =============================================================================
# TEST 1: Basic Training Flow with Verbose Output
# =============================================================================

def test_basic_verbose():
    """Test basic training with detailed input/output logging."""
    print_section("TEST 1: Basic Training Agent - Verbose")
    
    clear_registry()
    
    # ==========================================================================
    # STEP 1: Create Data
    # ==========================================================================
    print_subsection("STEP 1: Creating Synthetic Data")
    
    np.random.seed(42)
    n = 200
    
    # Features with predictive signal
    age = np.random.randint(25, 65, n)
    income = np.random.normal(60000, 20000, n).clip(20000, 150000)
    credit_score = np.random.randint(500, 800, n)
    
    # Target: higher age/income/credit = lower default
    risk_score = -0.05 * age - 0.00003 * income - 0.008 * credit_score + 8
    default_prob = 1 / (1 + np.exp(-risk_score))
    default = (np.random.random(n) < default_prob).astype(int)
    
    # Split 70/15/15
    train_df = pd.DataFrame({
        "age": age[:140],
        "income": income[:140],
        "credit_score": credit_score[:140],
        "default": default[:140],
    })
    
    val_df = pd.DataFrame({
        "age": age[140:170],
        "income": income[140:170],
        "credit_score": credit_score[140:170],
        "default": default[140:170],
    })
    
    test_df = pd.DataFrame({
        "age": age[170:],
        "income": income[170:],
        "credit_score": credit_score[170:],
        "default": default[170:],
    })
    
    print_dataframe_info("train_df", train_df)
    print_dataframe_info("val_df", val_df)
    print_dataframe_info("test_df", test_df)
    
    # ==========================================================================
    # STEP 2: Register Datasets
    # ==========================================================================
    print_subsection("STEP 2: Registering Datasets")
    
    register_dataset("verbose_train", train_df, register_sql=False)
    register_dataset("verbose_val", val_df, register_sql=False)
    register_dataset("verbose_test", test_df, register_sql=False)
    
    print("  Registered datasets:")
    print("    - verbose_train")
    print("    - verbose_val")
    print("    - verbose_test")
    
    # ==========================================================================
    # STEP 3: Prepare Agent Inputs
    # ==========================================================================
    print_subsection("STEP 3: Agent Input Parameters")
    
    agent_inputs = {
        "train_ref": "verbose_train",
        "val_ref": "verbose_val",
        "test_ref": "verbose_test",
        "target_column": "default",
        "selected_model": "logistic_regression",
        "goal": "Predict loan default risk for credit decisioning",
        "model_name": "verbose_test_model",
        "max_iterations": 2,
        "llm_model": "openai:gpt-4o",
    }
    
    print("  Agent will be called with:")
    for key, value in agent_inputs.items():
        print(f"    {key}: {value}")
    
    # ==========================================================================
    # STEP 4: Run Training Agent
    # ==========================================================================
    print_subsection("STEP 4: Running Training Agent")
    print("  (This will make LLM calls and train models...)")
    
    from agents.training.training import run_training_agent
    
    result = run_training_agent(**agent_inputs)
    
    # ==========================================================================
    # STEP 5: Analyze Results
    # ==========================================================================
    print_subsection("STEP 5: Agent Output")
    
    print("  Result keys:", list(result.keys()))
    print(f"  success: {result.get('success')}")
    print(f"  model_name: {result.get('model_name')}")
    print(f"  model_type: {result.get('model_type')}")
    print(f"  task_type: {result.get('task_type')}")
    print(f"  target_column: {result.get('target_column')}")
    print(f"  train_size: {result.get('train_size')}")
    print(f"  val_size: {result.get('val_size')}")
    print(f"  test_size: {result.get('test_size')}")
    
    if result.get('error'):
        print(f"  ERROR: {result.get('error')}")
    
    print_subsection("STEP 6: Agent Response (LLM Output)")
    
    if result.get('agent_response'):
        response = result['agent_response']
        print(f"  Total length: {len(response)} characters")
        print("\n  Full response:")
        print("  " + "-" * 60)
        # Print full response with indentation
        for line in response.split('\n'):
            print(f"  {line}")
        print("  " + "-" * 60)
    
    # ==========================================================================
    # STEP 7: Verify Model was Saved
    # ==========================================================================
    print_subsection("STEP 7: Verifying Model in Registry")
    
    from model_storage import list_trained_models_tool, get_model_info_tool
    
    models = list_trained_models_tool.invoke({})
    print("  Available models:")
    print(f"  {models[:1000]}")
    
    # Check if our model versions exist
    for version in ["_v1", "_v2", "_v3"]:
        model_name = f"verbose_test_model{version}"
        try:
            info = get_model_info_tool.invoke({"model_name": model_name})
            print(f"\n  Model {model_name} found!")
            print(f"  {info[:500]}")
        except:
            pass
    
    # ==========================================================================
    # VALIDATION
    # ==========================================================================
    print_subsection("VALIDATION")
    
    if result.get('success'):
        print("  ✅ Training completed successfully!")
    else:
        print(f"  ❌ Training failed: {result.get('error')}")
    
    return result


# =============================================================================
# TEST 2: Verify Tool Calls with Dataset Refs
# =============================================================================

def test_tool_calls_verbose():
    """Test that tools are called with dataset_ref (not raw data)."""
    print_section("TEST 2: Tool Calls with dataset_ref")
    
    clear_registry()
    
    print_subsection("Creating Test Data")
    
    np.random.seed(123)
    train_df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    })
    
    test_df = pd.DataFrame({
        "feature1": np.random.randn(10),
        "feature2": np.random.randn(10),
        "target": np.random.choice([0, 1], 10),
    })
    
    print(f"  train_df: {train_df.shape}")
    print(f"  test_df: {test_df.shape}")
    
    register_dataset("tool_train", train_df, register_sql=False)
    register_dataset("tool_test", test_df, register_sql=False)
    
    # Test sklearn_logistic_regression
    print_subsection("Testing sklearn_logistic_regression")
    
    from logistic_regression import sklearn_logistic_regression_tool
    
    tool_input = {
        "model_name": "tool_test_lr",
        "train_dataset_ref": "tool_train",
        "target_column": "target",
        "C": 1.0,
        "class_weight": "balanced",
    }
    
    print("  INPUT:")
    for k, v in tool_input.items():
        print(f"    {k}: {v}")
    
    result = sklearn_logistic_regression_tool.invoke(tool_input)
    
    print("\n  OUTPUT (first 800 chars):")
    print("  " + "-" * 50)
    for line in result[:800].split('\n'):
        print(f"  {line}")
    print("  " + "-" * 50)
    
    # Test evaluate_model
    print_subsection("Testing evaluate_model")
    
    from model_storage import evaluate_model_tool
    
    tool_input = {
        "model_name": "tool_test_lr",
        "dataset_ref": "tool_test",
        "target_column": "target",
    }
    
    print("  INPUT:")
    for k, v in tool_input.items():
        print(f"    {k}: {v}")
    
    result = evaluate_model_tool.invoke(tool_input)
    
    print("\n  OUTPUT:")
    print("  " + "-" * 50)
    for line in result.split('\n'):
        print(f"  {line}")
    print("  " + "-" * 50)
    
    # Test predict_with_model
    print_subsection("Testing predict_with_model")
    
    from model_storage import predict_with_model_tool
    
    tool_input = {
        "model_name": "tool_test_lr",
        "dataset_ref": "tool_test",
    }
    
    print("  INPUT:")
    for k, v in tool_input.items():
        print(f"    {k}: {v}")
    
    result = predict_with_model_tool.invoke(tool_input)
    
    print("\n  OUTPUT (first 800 chars):")
    print("  " + "-" * 50)
    for line in result[:800].split('\n'):
        print(f"  {line}")
    print("  " + "-" * 50)
    
    print("\n  ✅ All tools working with dataset_ref!")
    
    return True


# =============================================================================
# TEST 3: Multiple Model Types
# =============================================================================

def test_multiple_models():
    """Test training with different model types."""
    print_section("TEST 3: Multiple Model Types")
    
    clear_registry()
    
    np.random.seed(456)
    n = 100
    
    train_df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n),
        "x3": np.random.randn(n),
        "y": np.random.choice([0, 1], n, p=[0.6, 0.4]),
    })
    
    register_dataset("multi_train", train_df, register_sql=False)
    
    print_dataframe_info("Training data", train_df)
    
    # Test Logistic Regression
    print_subsection("Logistic Regression")
    from logistic_regression import sklearn_logistic_regression_tool
    
    result = sklearn_logistic_regression_tool.invoke({
        "model_name": "multi_lr",
        "train_dataset_ref": "multi_train",
        "target_column": "y",
    })
    print(f"  Result: {result[:400]}...")
    
    # Test Random Forest
    print_subsection("Random Forest")
    from random_forest import sklearn_random_forest_tool
    
    result = sklearn_random_forest_tool.invoke({
        "model_name": "multi_rf",
        "train_dataset_ref": "multi_train",
        "target_column": "y",
        "task_type": "classification",
        "n_estimators": 50,
    })
    print(f"  Result: {result[:400]}...")
    
    # Test XGBoost
    print_subsection("XGBoost")
    from xgboost_model import xgboost_train_tool
    
    result = xgboost_train_tool.invoke({
        "model_name": "multi_xgb",
        "train_dataset_ref": "multi_train",
        "target_column": "y",
        "task_type": "classification",
        "n_estimators": 50,
    })
    print(f"  Result: {result[:400]}...")
    
    print("\n  ✅ All model types trained successfully!")
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def run_all():
    """Run all verbose tests."""
    print("\n" + "=" * 70)
    print("  TRAINING AGENT VERBOSE TEST SUITE")
    print("=" * 70)
    print("  Running comprehensive tests with detailed I/O logging...")
    print("  Check LangSmith for agent traces!")
    print("=" * 70)
    
    results = []
    
    # Test 1: Basic flow
    try:
        test_basic_verbose()
        results.append(("Basic Training Flow", "PASSED"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(("Basic Training Flow", f"FAILED: {e}"))
    
    # Test 2: Tool calls
    try:
        test_tool_calls_verbose()
        results.append(("Tool Calls with dataset_ref", "PASSED"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(("Tool Calls with dataset_ref", f"FAILED: {e}"))
    
    # Test 3: Multiple models
    try:
        test_multiple_models()
        results.append(("Multiple Model Types", "PASSED"))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(("Multiple Model Types", f"FAILED: {e}"))
    
    # Summary
    print_section("TEST SUMMARY")
    
    passed = sum(1 for _, status in results if status == "PASSED")
    
    for name, status in results:
        icon = "✅" if status == "PASSED" else "❌"
        print(f"  {icon} {name}: {status}")
    
    print(f"\n  Total: {passed}/{len(results)} passed")
    
    return passed == len(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, help="Run specific test (1-3)")
    args = parser.parse_args()
    
    if args.test == 1:
        test_basic_verbose()
    elif args.test == 2:
        test_tool_calls_verbose()
    elif args.test == 3:
        test_multiple_models()
    else:
        success = run_all()
        exit(0 if success else 1)
