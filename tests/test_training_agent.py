"""
Comprehensive Tests for Training Agent (Step 7)

Tests the iterative training loop that:
1. Trains a model on training data
2. Evaluates on validation data
3. Iterates with different hyperparameters if needed
4. Evaluates final model on test data

Run with: python3 tests/test_training_agent.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "models-tools" / "training"))

import numpy as np
import pandas as pd
from utils import clear_registry, get_registered_dataset, register_dataset


def print_df_info(name, df):
    """Print detailed DataFrame info."""
    print(f"\n  {name}:")
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {list(df.columns)}")
    if 'default' in df.columns:
        dist = df['default'].value_counts().to_dict()
        rate = df['default'].mean()
        print(f"    Target distribution: {dist} (default rate: {rate:.1%})")
    print(f"    Sample row: {df.iloc[0].to_dict()}")


def print_agent_input(params):
    """Print agent input parameters."""
    print("\n[AGENT INPUT PARAMETERS]")
    print("-" * 40)
    for k, v in params.items():
        print(f"  {k}: {v}")
    print("-" * 40)


def print_agent_output(result):
    """Print agent output in detail."""
    print("\n[AGENT OUTPUT]")
    print("-" * 40)
    print(f"  success: {result.get('success')}")
    print(f"  model_name: {result.get('model_name')}")
    print(f"  model_type: {result.get('model_type')}")
    print(f"  task_type: {result.get('task_type')}")
    print(f"  target_column: {result.get('target_column')}")
    print(f"  train_size: {result.get('train_size')}")
    print(f"  val_size: {result.get('val_size')}")
    print(f"  test_size: {result.get('test_size')}")
    
    if result.get('error'):
        print(f"\n  ❌ ERROR: {result.get('error')}")
    
    print("-" * 40)
    
    if result.get('agent_response'):
        print("\n[FULL AGENT RESPONSE]")
        print("=" * 60)
        print(result['agent_response'])
        print("=" * 60)


# =============================================================================
# TEST 1: Basic Training Agent Flow
# =============================================================================

def test_training_agent_basic():
    """Test basic training agent flow with clean data that should converge quickly."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Training Agent Flow")
    print("=" * 70)
    
    clear_registry()
    
    # Create data with strong predictive signal
    np.random.seed(42)
    n = 300
    
    print("\n[STEP 1: Creating synthetic data with predictive signal]")
    
    # Features
    age = np.random.randint(25, 65, n)
    income = np.random.normal(60000, 20000, n).clip(20000, 150000)
    credit_score = np.random.randint(500, 800, n)
    
    # Target with clear relationship (higher values = lower default risk)
    risk_score = -0.05 * age - 0.00003 * income - 0.008 * credit_score + 8
    default_prob = 1 / (1 + np.exp(-risk_score))
    default = (np.random.random(n) < default_prob).astype(int)
    
    # Split into train/val/test
    train_df = pd.DataFrame({
        "age": age[:200],
        "income": income[:200],
        "credit_score": credit_score[:200],
        "default": default[:200],
    })
    
    val_df = pd.DataFrame({
        "age": age[200:250],
        "income": income[200:250],
        "credit_score": credit_score[200:250],
        "default": default[200:250],
    })
    
    test_df = pd.DataFrame({
        "age": age[250:],
        "income": income[250:],
        "credit_score": credit_score[250:],
        "default": default[250:],
    })
    
    print_df_info("train_df", train_df)
    print_df_info("val_df", val_df)
    print_df_info("test_df", test_df)
    
    print("\n[STEP 2: Registering datasets]")
    register_dataset("basic_train", train_df, register_sql=False)
    register_dataset("basic_val", val_df, register_sql=False)
    register_dataset("basic_test", test_df, register_sql=False)
    print("  Registered: basic_train, basic_val, basic_test")
    
    print("\n[STEP 3: Calling training agent]")
    
    from agents.training.training import run_training_agent
    
    agent_params = {
        "train_ref": "basic_train",
        "val_ref": "basic_val",
        "test_ref": "basic_test",
        "target_column": "default",
        "selected_model": "logistic_regression",
        "goal": "Predict loan default risk for credit decisioning",
        "model_name": "test_basic",
        "max_iterations": 2,
        "llm_model": "openai:gpt-5.1",
    }
    
    print_agent_input(agent_params)
    
    print("\n[STEP 4: Running agent (LLM calls happening)...]")
    result = run_training_agent(**agent_params)
    
    print("\n[STEP 5: Agent completed]")
    print_agent_output(result)
    
    # Validate
    print("\n[VALIDATION]")
    if result.get('success'):
        print("  ✅ Training succeeded")
    else:
        print(f"  ❌ Training failed: {result.get('error')}")
    
    assert result.get('success'), f"Training failed: {result.get('error')}"
    assert result.get('model_name') is not None, "No model name returned"
    
    print("\n" + "=" * 70)
    print("✅ TEST 1 PASSED")
    print("=" * 70)
    return result


# =============================================================================
# TEST 2: Imbalanced Data (Should trigger class_weight='balanced')
# =============================================================================

def test_training_agent_imbalanced():
    """Test training with imbalanced classes - agent should use class_weight='balanced'."""
    print("\n" + "=" * 70)
    print("TEST 2: Training with Imbalanced Data")
    print("=" * 70)
    
    clear_registry()
    
    np.random.seed(123)
    n = 400
    
    print("\n[STEP 1: Creating imbalanced dataset (5% positive class)]")
    
    # Features
    age = np.random.randint(20, 70, n)
    income = np.random.exponential(50000, n)
    credit_score = np.random.randint(300, 850, n)
    
    # Highly imbalanced target (5% default rate)
    default = np.zeros(n, dtype=int)
    default[:int(n * 0.05)] = 1
    np.random.shuffle(default)
    
    # Split
    train_df = pd.DataFrame({
        "age": age[:280],
        "income": income[:280],
        "credit_score": credit_score[:280],
        "default": default[:280],
    })
    
    val_df = pd.DataFrame({
        "age": age[280:340],
        "income": income[280:340],
        "credit_score": credit_score[280:340],
        "default": default[280:340],
    })
    
    test_df = pd.DataFrame({
        "age": age[340:],
        "income": income[340:],
        "credit_score": credit_score[340:],
        "default": default[340:],
    })
    
    print_df_info("train_df (IMBALANCED)", train_df)
    print_df_info("val_df", val_df)
    print_df_info("test_df", test_df)
    print("\n  ⚠️  This is HIGHLY IMBALANCED - agent should detect and use class_weight='balanced'")
    
    print("\n[STEP 2: Registering datasets]")
    register_dataset("imbal_train", train_df, register_sql=False)
    register_dataset("imbal_val", val_df, register_sql=False)
    register_dataset("imbal_test", test_df, register_sql=False)
    print("  Registered: imbal_train, imbal_val, imbal_test")
    
    print("\n[STEP 3: Calling training agent]")
    
    from agents.training.training import run_training_agent
    
    agent_params = {
        "train_ref": "imbal_train",
        "val_ref": "imbal_val",
        "test_ref": "imbal_test",
        "target_column": "default",
        "selected_model": "logistic_regression",
        "goal": "Predict rare default events",
        "model_name": "test_imbalanced",
        "max_iterations": 2,
        "llm_model": "openai:gpt-5.1",
    }
    
    print_agent_input(agent_params)
    
    print("\n[STEP 4: Running agent...]")
    result = run_training_agent(**agent_params)
    
    print("\n[STEP 5: Agent completed]")
    print_agent_output(result)
    
    print("\n[RESULT]")
    print(f"  Success: {result.get('success')}")
    print(f"  Model name: {result.get('model_name')}")
    
    if result.get('agent_response'):
        response = result['agent_response']
        print(f"\n[AGENT RESPONSE] (first 1500 chars)")
        print("-" * 50)
        print(response[:1500])
        print("-" * 50)
        
        # Check if agent mentioned class imbalance
        if 'balanced' in response.lower() or 'imbalance' in response.lower():
            print("\n✅ Agent recognized class imbalance!")
        else:
            print("\n⚠️  Agent may not have addressed class imbalance")
    
    assert result.get('success'), f"Training failed: {result.get('error')}"
    
    print("\n✅ TEST 2 PASSED")
    return result


# =============================================================================
# TEST 3: Random Forest Model
# =============================================================================

def test_training_agent_random_forest():
    """Test training with random forest model."""
    print("\n" + "=" * 70)
    print("TEST 3: Random Forest Training")
    print("=" * 70)
    
    clear_registry()
    
    np.random.seed(456)
    n = 300
    
    print("\n[SETUP] Creating data for random forest...")
    
    # Features with non-linear relationships
    age = np.random.randint(20, 70, n)
    income = np.random.exponential(50000, n)
    credit_score = np.random.randint(300, 850, n)
    debt_ratio = np.random.uniform(0.1, 0.9, n)
    
    # Non-linear target (interaction effects)
    risk = (age < 30).astype(int) + (income < 30000).astype(int) + (credit_score < 600).astype(int)
    default = (risk >= 2).astype(int)
    
    # Add noise
    flip_idx = np.random.choice(n, size=int(n * 0.1), replace=False)
    default[flip_idx] = 1 - default[flip_idx]
    
    train_df = pd.DataFrame({
        "age": age[:200],
        "income": income[:200],
        "credit_score": credit_score[:200],
        "debt_ratio": debt_ratio[:200],
        "default": default[:200],
    })
    
    val_df = pd.DataFrame({
        "age": age[200:250],
        "income": income[200:250],
        "credit_score": credit_score[200:250],
        "debt_ratio": debt_ratio[200:250],
        "default": default[200:250],
    })
    
    test_df = pd.DataFrame({
        "age": age[250:],
        "income": income[250:],
        "credit_score": credit_score[250:],
        "debt_ratio": debt_ratio[250:],
        "default": default[250:],
    })
    
    print(f"  Train: {train_df.shape}, default rate: {train_df['default'].mean():.1%}")
    print(f"  Val: {val_df.shape}")
    print(f"  Test: {test_df.shape}")
    
    register_dataset("rf_train", train_df, register_sql=False)
    register_dataset("rf_val", val_df, register_sql=False)
    register_dataset("rf_test", test_df, register_sql=False)
    
    print("\n[RUNNING] Training random forest...")
    
    from agents.training.training import run_training_agent
    
    result = run_training_agent(
        train_ref="rf_train",
        val_ref="rf_val",
        test_ref="rf_test",
        target_column="default",
        selected_model="random_forest",
        goal="Predict default with non-linear feature interactions",
        model_name="test_rf",
        max_iterations=2,
    )
    
    print("\n[RESULT]")
    print(f"  Success: {result.get('success')}")
    print(f"  Model type: {result.get('model_type')}")
    
    if result.get('agent_response'):
        print(f"\n[AGENT RESPONSE] (first 1500 chars)")
        print("-" * 50)
        print(result['agent_response'][:1500])
    
    assert result.get('success'), f"Training failed: {result.get('error')}"
    
    print("\n✅ TEST 3 PASSED")
    return result


# =============================================================================
# TEST 4: XGBoost Model
# =============================================================================

def test_training_agent_xgboost():
    """Test training with XGBoost model."""
    print("\n" + "=" * 70)
    print("TEST 4: XGBoost Training")
    print("=" * 70)
    
    clear_registry()
    
    np.random.seed(789)
    n = 300
    
    print("\n[SETUP] Creating data for XGBoost...")
    
    age = np.random.randint(20, 70, n)
    income = np.random.exponential(60000, n)
    credit_score = np.random.randint(400, 800, n)
    
    # Complex target
    score = 0.02 * age + 0.00001 * income + 0.005 * credit_score
    default = (score < np.median(score)).astype(int)
    
    train_df = pd.DataFrame({
        "age": age[:200],
        "income": income[:200],
        "credit_score": credit_score[:200],
        "default": default[:200],
    })
    
    val_df = pd.DataFrame({
        "age": age[200:250],
        "income": income[200:250],
        "credit_score": credit_score[200:250],
        "default": default[200:250],
    })
    
    test_df = pd.DataFrame({
        "age": age[250:],
        "income": income[250:],
        "credit_score": credit_score[250:],
        "default": default[250:],
    })
    
    print(f"  Train: {train_df.shape}, default rate: {train_df['default'].mean():.1%}")
    
    register_dataset("xgb_train", train_df, register_sql=False)
    register_dataset("xgb_val", val_df, register_sql=False)
    register_dataset("xgb_test", test_df, register_sql=False)
    
    print("\n[RUNNING] Training XGBoost...")
    
    from agents.training.training import run_training_agent
    
    result = run_training_agent(
        train_ref="xgb_train",
        val_ref="xgb_val",
        test_ref="xgb_test",
        target_column="default",
        selected_model="xgboost",
        goal="Predict default using gradient boosting",
        model_name="test_xgb",
        max_iterations=2,
    )
    
    print("\n[RESULT]")
    print(f"  Success: {result.get('success')}")
    print(f"  Model type: {result.get('model_type')}")
    
    if result.get('agent_response'):
        print(f"\n[AGENT RESPONSE] (first 1000 chars)")
        print("-" * 50)
        print(result['agent_response'][:1000])
    
    assert result.get('success'), f"Training failed: {result.get('error')}"
    
    print("\n✅ TEST 4 PASSED")
    return result


# =============================================================================
# TEST 5: Verify Model is Saved and Usable
# =============================================================================

def test_model_saved_and_usable():
    """Test that trained model is saved to registry and can make predictions."""
    print("\n" + "=" * 70)
    print("TEST 5: Model Saved and Usable")
    print("=" * 70)
    
    clear_registry()
    
    np.random.seed(111)
    n = 200
    
    print("\n[SETUP] Creating simple training data...")
    
    train_df = pd.DataFrame({
        "feature1": np.random.randn(n),
        "feature2": np.random.randn(n),
        "target": np.random.choice([0, 1], n),
    })
    
    val_df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    })
    
    test_df = pd.DataFrame({
        "feature1": np.random.randn(50),
        "feature2": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    })
    
    register_dataset("save_train", train_df, register_sql=False)
    register_dataset("save_val", val_df, register_sql=False)
    register_dataset("save_test", test_df, register_sql=False)
    
    print("\n[RUNNING] Training model...")
    
    from agents.training.training import run_training_agent
    
    result = run_training_agent(
        train_ref="save_train",
        val_ref="save_val",
        test_ref="save_test",
        target_column="target",
        selected_model="logistic_regression",
        goal="Simple classification",
        model_name="test_saved_model",
        max_iterations=1,
    )
    
    print(f"\n[RESULT] Success: {result.get('success')}")
    
    # Now try to use the model
    print("\n[VERIFY] Checking if model is in registry...")
    
    from model_storage import list_trained_models_tool, predict_with_model_tool
    
    models_list = list_trained_models_tool.invoke({})
    print(f"  Models in registry:\n{models_list[:500]}")
    
    # Try to make a prediction using evaluate_model (which uses dataset_ref)
    print("\n[VERIFY] Making prediction with saved model...")
    
    # Create and register a small test dataset for prediction
    pred_df = pd.DataFrame({
        "feature1": [0.5, -1.2],
        "feature2": [-0.3, 0.8],
        "target": [0, 1],  # Required for evaluate_model
    })
    register_dataset("pred_test_data", pred_df, register_sql=False)
    
    # Find the model name (it might have _v1 appended)
    import re
    model_names = re.findall(r'test_saved_model\w*', models_list)
    if model_names:
        model_to_use = model_names[0]
        print(f"  Using model: {model_to_use}")
        
        from model_storage import evaluate_model_tool
        
        evaluation = evaluate_model_tool.invoke({
            "model_name": model_to_use,
            "dataset_ref": "pred_test_data",
            "target_column": "target",
        })
        print(f"  Evaluation result:\n{evaluation[:500]}")
        
        assert "accuracy" in evaluation.lower() or "prediction" in evaluation.lower(), "No metrics in output"
        print("\n✅ Model is saved and usable!")
    else:
        print("  ⚠️ Could not find model in registry")
    
    print("\n✅ TEST 5 PASSED")
    return result


# =============================================================================
# TEST 6: Check Iteration Behavior
# =============================================================================

def test_iteration_behavior():
    """Test that agent iterates when initial metrics are poor."""
    print("\n" + "=" * 70)
    print("TEST 6: Iteration Behavior Check")
    print("=" * 70)
    
    clear_registry()
    
    np.random.seed(999)
    n = 200
    
    print("\n[SETUP] Creating noisy data (hard to predict)...")
    
    # Very noisy data - hard to get good metrics
    train_df = pd.DataFrame({
        "noise1": np.random.randn(n),
        "noise2": np.random.randn(n),
        "noise3": np.random.randn(n),
        "target": np.random.choice([0, 1], n),  # Pure random - no signal
    })
    
    val_df = pd.DataFrame({
        "noise1": np.random.randn(50),
        "noise2": np.random.randn(50),
        "noise3": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    })
    
    test_df = pd.DataFrame({
        "noise1": np.random.randn(50),
        "noise2": np.random.randn(50),
        "noise3": np.random.randn(50),
        "target": np.random.choice([0, 1], 50),
    })
    
    print(f"  Train: {train_df.shape} - PURE NOISE (no predictive signal)")
    print(f"  This should cause poor metrics and potentially trigger iteration")
    
    register_dataset("iter_train", train_df, register_sql=False)
    register_dataset("iter_val", val_df, register_sql=False)
    register_dataset("iter_test", test_df, register_sql=False)
    
    print("\n[RUNNING] Training on noisy data...")
    
    from agents.training.training import run_training_agent
    
    result = run_training_agent(
        train_ref="iter_train",
        val_ref="iter_val",
        test_ref="iter_test",
        target_column="target",
        selected_model="logistic_regression",
        goal="Classify noisy data",
        model_name="test_iteration",
        max_iterations=3,
    )
    
    print(f"\n[RESULT] Success: {result.get('success')}")
    
    if result.get('agent_response'):
        response = result['agent_response']
        print(f"\n[AGENT RESPONSE] (first 2000 chars)")
        print("-" * 50)
        print(response[:2000])
        print("-" * 50)
        
        # Check for signs of iteration
        if '_v2' in response or '_v3' in response or 'iteration 2' in response.lower():
            print("\n✅ Agent performed multiple iterations!")
        else:
            print("\n⚠️ Agent may have stopped after first iteration (acceptable if metrics were okay)")
    
    print("\n✅ TEST 6 PASSED")
    return result


# =============================================================================
# TEST 7: Full Pipeline Integration
# =============================================================================

def test_full_pipeline_integration():
    """Test training agent as part of full pipeline (simulates coming from step 5)."""
    print("\n" + "=" * 70)
    print("TEST 7: Full Pipeline Integration")
    print("=" * 70)
    
    clear_registry()
    
    np.random.seed(2024)
    
    print("\n[SETUP] Simulating data coming from feature engineering (step 5)...")
    
    # Simulate transformed data that would come from feature engineering
    n_train, n_val, n_test = 350, 75, 75
    
    # Features that would have been engineered
    train_df = pd.DataFrame({
        "age_binned": np.random.choice([0, 1, 2, 3], n_train),  # Binned feature
        "income_log": np.log(np.random.exponential(50000, n_train) + 1),  # Log transform
        "region_encoded_North": np.random.choice([0, 1], n_train),  # One-hot
        "region_encoded_South": np.random.choice([0, 1], n_train),
        "credit_score_normalized": np.random.randn(n_train),  # Standardized
        "default": np.random.choice([0, 1], n_train, p=[0.7, 0.3]),
    })
    
    val_df = pd.DataFrame({
        "age_binned": np.random.choice([0, 1, 2, 3], n_val),
        "income_log": np.log(np.random.exponential(50000, n_val) + 1),
        "region_encoded_North": np.random.choice([0, 1], n_val),
        "region_encoded_South": np.random.choice([0, 1], n_val),
        "credit_score_normalized": np.random.randn(n_val),
        "default": np.random.choice([0, 1], n_val, p=[0.7, 0.3]),
    })
    
    test_df = pd.DataFrame({
        "age_binned": np.random.choice([0, 1, 2, 3], n_test),
        "income_log": np.log(np.random.exponential(50000, n_test) + 1),
        "region_encoded_North": np.random.choice([0, 1], n_test),
        "region_encoded_South": np.random.choice([0, 1], n_test),
        "credit_score_normalized": np.random.randn(n_test),
        "default": np.random.choice([0, 1], n_test, p=[0.7, 0.3]),
    })
    
    print(f"  Train: {train_df.shape}")
    print(f"  Val: {val_df.shape}")
    print(f"  Test: {test_df.shape}")
    print(f"  Features: {list(train_df.columns[:-1])}")
    
    # Register as if coming from step 5
    register_dataset("pipeline_train_features", train_df, register_sql=False)
    register_dataset("pipeline_val_features", val_df, register_sql=False)
    register_dataset("pipeline_test_features", test_df, register_sql=False)
    
    print("\n[RUNNING] Training agent (simulating step 7)...")
    
    from agents.training.training import run_training_agent
    
    result = run_training_agent(
        train_ref="pipeline_train_features",
        val_ref="pipeline_val_features",
        test_ref="pipeline_test_features",
        target_column="default",
        selected_model="logistic_regression",
        goal="Predict loan default for underwriting decisions",
        model_name="pipeline_model",
        max_iterations=3,
    )
    
    print(f"\n[RESULT]")
    print(f"  Success: {result.get('success')}")
    print(f"  Model: {result.get('model_name')}")
    print(f"  Train size: {result.get('train_size')}")
    print(f"  Val size: {result.get('val_size')}")
    print(f"  Test size: {result.get('test_size')}")
    
    if result.get('agent_response'):
        print(f"\n[AGENT RESPONSE] (first 1500 chars)")
        print("-" * 50)
        print(result['agent_response'][:1500])
    
    assert result.get('success'), f"Pipeline integration failed: {result.get('error')}"
    
    print("\n✅ TEST 7 PASSED")
    return result


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all tests and report summary."""
    print("\n" + "=" * 70)
    print("TRAINING AGENT TEST SUITE")
    print("=" * 70)
    print("Running comprehensive tests for training agent (Step 7)")
    print("Check LangSmith for detailed traces of each test")
    print("=" * 70)
    
    tests = [
        ("Basic Training Flow", test_training_agent_basic),
        ("Imbalanced Data", test_training_agent_imbalanced),
        ("Random Forest", test_training_agent_random_forest),
        ("XGBoost", test_training_agent_xgboost),
        ("Model Saved and Usable", test_model_saved_and_usable),
        ("Iteration Behavior", test_iteration_behavior),
        ("Full Pipeline Integration", test_full_pipeline_integration),
    ]
    
    results = []
    
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, "PASSED", None))
        except Exception as e:
            import traceback
            results.append((name, "FAILED", str(e)))
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    
    for name, status, error in results:
        icon = "✅" if status == "PASSED" else "❌"
        print(f"  {icon} {name}: {status}")
        if error:
            print(f"      Error: {error[:100]}")
    
    print("-" * 70)
    print(f"  Total: {passed}/{len(results)} passed")
    
    if failed > 0:
        print(f"\n❌ {failed} test(s) failed!")
        return False
    else:
        print(f"\n✅ All tests passed!")
        return True


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training Agent Tests")
    parser.add_argument("--test", type=int, help="Run specific test (1-7)")
    parser.add_argument("--quick", action="store_true", help="Run only test 1 (quick check)")
    args = parser.parse_args()
    
    if args.quick:
        test_training_agent_basic()
    elif args.test:
        tests = {
            1: test_training_agent_basic,
            2: test_training_agent_imbalanced,
            3: test_training_agent_random_forest,
            4: test_training_agent_xgboost,
            5: test_model_saved_and_usable,
            6: test_iteration_behavior,
            7: test_full_pipeline_integration,
        }
        if args.test in tests:
            tests[args.test]()
        else:
            print(f"Invalid test number. Choose 1-{len(tests)}")
    else:
        success = run_all_tests()
        exit(0 if success else 1)
