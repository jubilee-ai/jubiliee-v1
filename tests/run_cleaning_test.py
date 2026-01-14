"""
Full integration test for cleaning_standardization.
Runs the complete flow with real LLM and prints all inputs/outputs.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agents" / "training"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from utils import clear_dataset_registry, register_dataset, get_registered_dataset
from cleaning_standardization import cleaning_standardization, should_continue_cleaning

# =============================================================================
# HELPERS
# =============================================================================

def print_section(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_dataframe(df: pd.DataFrame, name: str, max_rows: int = 10):
    print(f"\n📊 {name}")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n   Sample data (first {min(max_rows, len(df))} rows):")
    print(df.head(max_rows).to_string(index=True))
    print(f"\n   Null counts:")
    print(df.isnull().sum().to_string())


def print_state(state: dict, title: str = "State"):
    print(f"\n📋 {title}")
    for key, value in state.items():
        if key in ["explanations", "cleaning_transformations", "audit_trace"]:
            print(f"   {key}: {json.dumps(value, indent=6, default=str)[:500]}")
        elif isinstance(value, dict):
            print(f"   {key}: {json.dumps(value, indent=6, default=str)[:300]}")
        else:
            print(f"   {key}: {value}")


# =============================================================================
# CREATE TEST DATASET
# =============================================================================

def create_dirty_dataset() -> pd.DataFrame:
    """Create a dataset with various data quality issues."""
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        # High nulls (40%) - should trigger impute/drop recommendation
        "income": np.where(np.random.random(n) > 0.4, 
                          np.random.normal(50000, 15000, n), np.nan),
        
        # High skew (lognormal) - should trigger bin/log recommendation
        "claim_amount": np.random.lognormal(8, 2, n),
        
        # Some nulls (15%)
        "age": np.where(np.random.random(n) > 0.15, 
                       np.random.randint(18, 80, n).astype(float), np.nan),
        
        # Clean numeric
        "credit_score": np.random.randint(300, 850, n),
        
        # Clean categorical
        "region": np.random.choice(["North", "South", "East", "West"], n),
        
        # ID-like column - should be flagged
        "customer_id": [f"CUST-{i:06d}" for i in range(n)],
        
        # Target variable (binary classification)
        "churned": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    
    return df


def create_clean_dataset() -> pd.DataFrame:
    """Create a clean dataset with minimal issues."""
    np.random.seed(42)
    n = 500
    
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(50000, 15000, n),
        "credit_score": np.random.randint(300, 850, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "churned": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    
    return df


# =============================================================================
# RUN TEST
# =============================================================================

def run_cleaning_test(dataset_type: str = "dirty", max_iterations: int = 5):
    """Run the cleaning flow and print all details."""
    
    print_section("CLEANING STANDARDIZATION INTEGRATION TEST")
    print(f"Dataset type: {dataset_type}")
    print(f"Max test iterations: {max_iterations}")
    
    # Clear registry
    clear_dataset_registry()
    
    # Create dataset
    print_section("1. INPUT DATASET")
    if dataset_type == "dirty":
        df = create_dirty_dataset()
    else:
        df = create_clean_dataset()
    
    dataset_ref = f"test_{dataset_type}_data"
    register_dataset(dataset_ref, df)
    print_dataframe(df, f"Dataset: {dataset_ref}")
    
    # Create initial state
    print_section("2. INITIAL STATE")
    state = {
        "goal": "Build a model to predict customer churn based on demographics and financial data",
        "collected_dataset_ref": dataset_ref,
        "cleaned_dataset_ref": None,
        "cleaning_transformations": [],
        "cleaning_iteration": 0,
        "cleaning_status": None,
        "explanations": [],
        "audit_trace": [],
        "error": None,
    }
    print_state(state, "Initial State")
    
    # Run iterations
    iteration = 0
    while iteration < max_iterations:
        print_section(f"3.{iteration + 1} ITERATION {iteration + 1}")
        
        print("\n🔄 Calling cleaning_standardization()...")
        result = cleaning_standardization(state)
        
        print_state(result, f"Result State (iteration {iteration + 1})")
        
        # Show the current dataset
        current_ref = result.get("cleaned_dataset_ref") or result.get("collected_dataset_ref")
        if current_ref:
            current_df = get_registered_dataset(current_ref)
            if current_df is not None:
                print_dataframe(current_df, f"Current Dataset: {current_ref}")
        
        # Check if we should continue
        decision = should_continue_cleaning(result)
        print(f"\n🔀 should_continue_cleaning() returned: '{decision}'")
        
        if decision == "done":
            print("\n✅ Cleaning complete!")
            break
        
        # Update state for next iteration
        state = result
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"\n⚠️ Reached max test iterations ({max_iterations})")
    
    # Final summary
    print_section("4. FINAL SUMMARY")
    print(f"Total iterations: {iteration + 1}")
    print(f"Final status: {result.get('cleaning_status')}")
    print(f"Final dataset_ref: {result.get('cleaned_dataset_ref')}")
    print(f"Error: {result.get('error')}")
    
    print("\n📝 All explanations:")
    for i, exp in enumerate(result.get("explanations", [])):
        print(f"   {i + 1}. {exp}")
    
    print("\n🔧 All transformations applied:")
    for i, t in enumerate(result.get("cleaning_transformations", [])):
        print(f"   {i + 1}. {t['tool']}")
        print(f"      Args: {t.get('args', {})}")
        print(f"      Result: {t.get('result', t.get('error', 'n/a'))[:150]}")
    
    # Show final dataset comparison
    if result.get("cleaned_dataset_ref"):
        final_df = get_registered_dataset(result["cleaned_dataset_ref"])
        if final_df is not None:
            print_section("5. FINAL DATASET COMPARISON")
            print(f"\n📊 Original: {df.shape[0]} rows × {df.shape[1]} columns")
            print(f"📊 Final:    {final_df.shape[0]} rows × {final_df.shape[1]} columns")
            print(f"\n   Original null counts:\n{df.isnull().sum().to_string()}")
            print(f"\n   Final null counts:\n{final_df.isnull().sum().to_string()}")
    
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run cleaning standardization test")
    parser.add_argument("--dataset", choices=["dirty", "clean"], default="dirty",
                       help="Type of test dataset to use")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Maximum iterations to run")
    
    args = parser.parse_args()
    
    run_cleaning_test(dataset_type=args.dataset, max_iterations=args.max_iterations)
