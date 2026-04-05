"""
Test for Feature Engineering Redo functionality.

This test creates a scenario with poor features that should trigger
the training agent to request a feature engineering redo.

Run: python3 tests/test_feature_redo.py
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


def print_section(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subsection(title):
    print(f"\n[{title}]")
    print("-" * 50)


# =============================================================================
# TEST: Feature Engineering Redo Trigger
# =============================================================================

def test_feature_redo_trigger():
    """
    Create a dataset with:
    - Target that is PURELY RANDOM with respect to the features
    - No model can do better than random chance
    
    This should cause all models to fail and trigger the LLM to request
    a feature engineering redo.
    """
    print_section("TEST: Feature Engineering Redo Trigger")
    
    clear_registry()
    
    # ==========================================================================
    # STEP 1: Create Data with NO Signal (Random Target)
    # ==========================================================================
    print_subsection("STEP 1: Creating Dataset with NO Signal")
    
    np.random.seed(42)
    n = 300
    
    # Create features that look plausible but have NO relationship to target
    feature_cols = {
        "credit_score": np.random.randint(500, 800, n),
        "income": np.random.randint(30000, 150000, n),
        "age": np.random.randint(22, 65, n),
        "loan_amount": np.random.randint(5000, 100000, n),
        "months_employed": np.random.randint(0, 240, n),
        "num_credit_lines": np.random.randint(1, 15, n),
        "debt_ratio": np.random.uniform(0, 0.8, n),
    }
    
    # Target is COMPLETELY RANDOM - no model can predict this!
    target = np.random.choice([0, 1], n, p=[0.7, 0.3])
    
    df = pd.DataFrame({
        **feature_cols,
        "default": target,
    })
    
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Target distribution: {df['default'].value_counts().to_dict()}")
    print(f"  Target rate: {df['default'].mean():.1%}")
    print()
    print("  ⚠️ SECRET: The target is COMPLETELY RANDOM!")
    print("    No model can do better than ~0.5 ROC-AUC (random chance)")
    print("    This should trigger the LLM to request feature redo")
    print()
    print("  The features look realistic (credit_score, income, etc.)")
    print("  but they have ZERO predictive power for this target.")
    
    # ==========================================================================
    # STEP 2: Split and Register
    # ==========================================================================
    print_subsection("STEP 2: Splitting and Registering")
    
    # 70/15/15 split
    train_df = df.iloc[:210].copy()
    val_df = df.iloc[210:255].copy()
    test_df = df.iloc[255:].copy()
    
    print(f"  Train: {train_df.shape}, target rate: {train_df['default'].mean():.1%}")
    print(f"  Val: {val_df.shape}, target rate: {val_df['default'].mean():.1%}")
    print(f"  Test: {test_df.shape}, target rate: {test_df['default'].mean():.1%}")
    
    register_dataset("redo_train", train_df, register_sql=False)
    register_dataset("redo_val", val_df, register_sql=False)
    register_dataset("redo_test", test_df, register_sql=False)
    
    # ==========================================================================
    # STEP 3: Run Training Agent
    # ==========================================================================
    print_subsection("STEP 3: Running Training Agent")
    print("  The agent will struggle because:")
    print("    - The target is completely random")
    print("    - No model can achieve ROC-AUC > 0.5 (random chance)")
    print("    - All features have ZERO predictive power")
    print()
    print("  Expected behavior:")
    print("    - All models will show poor performance (ROC-AUC ~0.5)")
    print("    - LLM should try multiple models/hyperparameters")
    print("    - Eventually, LLM should call request_feature_engineering_redo")
    print("      recognizing that the features themselves are the problem")
    print()
    
    from agents.training.training import run_training_agent
    
    result = run_training_agent(
        train_ref="redo_train",
        val_ref="redo_val",
        test_ref="redo_test",
        target_column="default",
        selected_model="logistic_regression",
        goal="Predict loan defaults based on borrower characteristics.",
        model_name="redo_test_model",
        max_iterations=6,  # Give it more chances to try different things
        llm_model="openai:gpt-5.1",  # Use gpt-5-mini for higher rate limits
    )
    
    # ==========================================================================
    # STEP 4: Analyze Results
    # ==========================================================================
    print_subsection("STEP 4: Training Results")
    
    print(f"  Success: {result.get('success')}")
    print(f"  Model: {result.get('model_name')}")
    print(f"  Val Accuracy: {result.get('val_accuracy')}")
    print(f"  Val ROC-AUC: {result.get('val_roc_auc')}")
    print(f"  Test Accuracy: {result.get('test_accuracy')}")
    print(f"  Test ROC-AUC: {result.get('test_roc_auc')}")
    print(f"  Num Iterations: {result.get('num_iterations')}")
    
    # Check if feature redo was requested
    print_subsection("STEP 5: Feature Redo Check")
    
    feature_redo_requested = result.get("feature_redo_requested", False)
    feature_redo_recommendation = result.get("feature_redo_recommendation")
    feature_redo_reason = result.get("feature_redo_reason")
    
    print(f"  Feature Redo Requested: {feature_redo_requested}")
    
    if feature_redo_requested:
        print(f"\n  ✅ FEATURE REDO WAS TRIGGERED!")
        print(f"\n  Recommendation from LLM:")
        print(f"    {feature_redo_recommendation}")
        print(f"\n  Reason:")
        print(f"    {feature_redo_reason}")
    else:
        print(f"\n  ⚠️ Feature redo was NOT triggered")
        # Check the summary for mentions of features
        summary = result.get('summary', '')
        if 'feature' in summary.lower():
            print(f"\n  Summary mentions features:")
            print(f"    {summary}")
    
    # Show iteration log
    print_subsection("STEP 6: Iteration Log")
    
    iterations = result.get('iterations', [])
    for i, it in enumerate(iterations, 1):
        status = "✅" if it.get("success") else "❌"
        m = it.get("metrics") or {}
        print(f"  {status} [{i}] {it.get('model_name', 'unknown')[:30]}")
        print(f"       Tool: {it.get('tool')}")
        print(f"       Val Acc: {m.get('val_accuracy')}, ROC: {m.get('roc_auc')}")
        if it.get('error'):
            print(f"       Error: {it.get('error')}")
    
    return result


def test_feature_redo_full_pipeline():
    """
    Test the full pipeline including the feature engineering redo loop.
    
    Uses invoke_training_agent which runs the full graph including
    the conditional edge back to feature engineering.
    """
    print_section("TEST: Full Pipeline with Feature Redo Loop")
    
    clear_registry()
    
    # ==========================================================================
    # Create a dataset that will likely trigger redo
    # ==========================================================================
    print_subsection("Creating Dataset")
    
    np.random.seed(123)
    n = 300
    
    # Simple nonlinear relationship that linear models will struggle with
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    
    # Target is XOR-like: positive in 2 quadrants, negative in other 2
    target = ((x1 > 0) ^ (x2 > 0)).astype(int)
    
    # Add noise
    noise = np.random.randn(n) * 0.5
    x1_noisy = x1 + noise
    
    # Add random noise columns
    df = pd.DataFrame({
        "feature_x": x1_noisy,
        "feature_y": x2,
        "random_1": np.random.randn(n),
        "random_2": np.random.randn(n),
        "random_3": np.random.randn(n),
        "target": target,
    })
    
    print(f"  Dataset: {df.shape}")
    print(f"  Target rate: {df['target'].mean():.1%}")
    print()
    print("  This is an XOR problem - linear models will fail!")
    print("  The LLM should realize it needs interaction features.")
    
    # Register for full pipeline
    register_dataset("xor_dataset", df, register_sql=False)
    
    # ==========================================================================
    # Run Full Pipeline
    # ==========================================================================
    print_subsection("Running Full Training Pipeline")
    
    from agents.training.agent import invoke_training_agent
    
    result = invoke_training_agent(
        goal="Predict target. This may require feature interactions.",
        linked_datasets=["xor_dataset"],
        user_model_preference="logistic_regression",
    )
    
    # ==========================================================================
    # Check Results
    # ==========================================================================
    print_subsection("Pipeline Results")
    
    print(f"  Error: {result.get('error')}")
    print(f"  Selected Model: {result.get('selected_model')}")
    print(f"  Report Path: {result.get('report_path')}")
    
    training_metrics = result.get("training_metrics", {})
    print(f"\n  Training Metrics:")
    print(f"    Success: {training_metrics.get('success')}")
    print(f"    Val ROC-AUC: {training_metrics.get('val_roc_auc')}")
    print(f"    Test ROC-AUC: {training_metrics.get('test_roc_auc')}")
    
    # Check feature redo iteration
    print(f"\n  Feature Redo Status:")
    print(f"    Redo Requested: {result.get('feature_redo_requested')}")
    print(f"    Redo Iteration: {result.get('feature_redo_iteration')}")
    print(f"    Redo Recommendation: {result.get('feature_redo_recommendation')}")
    
    # Check audit trace for redo
    audit_trace = result.get("audit_trace", [])
    redo_count = sum(1 for item in audit_trace 
                     if item.get("step") == "feature_selection_specification" 
                     and item.get("is_redo"))
    
    print(f"\n  Audit Trace:")
    print(f"    Total steps: {len(audit_trace)}")
    print(f"    Feature redo iterations: {redo_count}")
    
    for item in audit_trace:
        step = item.get("step", "unknown")
        if item.get("is_redo"):
            print(f"    ⟳ [{step}] REDO - Recommendation: {item.get('redo_recommendation', '')[:50]}...")
        elif step == "training":
            print(f"    - [{step}] Model: {item.get('model_name')}, Redo requested: {item.get('feature_redo_requested')}")
        else:
            print(f"    - [{step}]")
    
    if redo_count > 0:
        print(f"\n  ✅ Feature engineering redo was triggered {redo_count} time(s)!")
    else:
        print(f"\n  ⚠️ Feature engineering redo was not triggered")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, choices=[1, 2], 
                       help="Run specific test (1=training agent only, 2=full pipeline)")
    args = parser.parse_args()
    
    if args.test == 1:
        test_feature_redo_trigger()
    elif args.test == 2:
        test_feature_redo_full_pipeline()
    else:
        # Run both tests
        print("\n" + "=" * 70)
        print("  FEATURE ENGINEERING REDO TEST SUITE")
        print("=" * 70)
        
        print("\n  Test 1: Training Agent Only (direct call)")
        result1 = test_feature_redo_trigger()
        
        print("\n\n" + "=" * 70)
        print("  Test 2: Full Pipeline with Graph Routing")
        print("=" * 70)
        
        result2 = test_feature_redo_full_pipeline()
        
        # Summary
        print_section("TEST SUMMARY")
        
        redo1 = result1.get("feature_redo_requested", False)
        redo2 = result2.get("feature_redo_iteration", 0) > 0
        
        print(f"  Test 1 (Training Agent): Feature redo {'TRIGGERED' if redo1 else 'not triggered'}")
        print(f"  Test 2 (Full Pipeline): Feature redo {'TRIGGERED' if redo2 else 'not triggered'}")
