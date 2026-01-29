"""
Test for the simple cleaning agent.
Run with: python -m pytest tests/test_cleaning_simple.py -v -s
Or directly: python tests/test_cleaning_simple.py
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))

import pandas as pd
import numpy as np
from utils import register_dataset, get_registered_dataset, clear_registry

def test_cleaning_simple_full_flow():
    """Test the full cleaning agent flow with a messy dataset."""
    
    print("\n" + "="*60)
    print("TEST: Simple Cleaning Agent Full Flow")
    print("="*60)
    
    # Clear any existing datasets
    clear_registry()
    
    # Create a messy test dataset
    print("\n[1] Creating test dataset with issues...")
    np.random.seed(42)
    n_rows = 100
    
    df = pd.DataFrame({
        "customer_id": range(1, n_rows + 1),  # ID-like column
        "age": np.random.randint(18, 80, n_rows).astype(float),
        "income": np.random.normal(50000, 20000, n_rows),
        "credit_score": np.random.randint(300, 850, n_rows),
        "loan_amount": np.random.uniform(1000, 50000, n_rows),
        "default": np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    })
    
    # Add some nulls
    df.loc[0:9, "age"] = np.nan  # 10% nulls
    df.loc[10:14, "income"] = np.nan  # 5% nulls
    
    # Add some outliers
    df.loc[95, "income"] = 500000  # Extreme outlier
    df.loc[96, "income"] = -10000  # Negative income (invalid)
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Nulls: age={df['age'].isna().sum()}, income={df['income'].isna().sum()}")
    print(f"   Income range: [{df['income'].min():.0f}, {df['income'].max():.0f}]")
    
    # Register the dataset
    test_ref = "test_messy_data"
    register_dataset(test_ref, df)
    print(f"\n[2] Registered dataset as: `{test_ref}`")
    
    # Import and run the agent
    print("\n[3] Running cleaning agent...")
    print("-"*60)
    
    from agents.training.steps.cleaning_simple import run_cleaning_simple
    
    result = run_cleaning_simple(
        dataset_ref=test_ref,
        goal="Predict loan defaults",
        max_iterations=25,
    )
    
    # Print all messages and track tool usage
    print("\n[4] Agent Messages:")
    print("-"*60)
    
    messages = result.get("messages", [])
    tool_call_counts = {}
    batched_transforms = []
    
    for i, msg in enumerate(messages):
        role = getattr(msg, "type", "unknown")
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        
        print(f"\n--- Message {i+1} ({role}) ---")
        
        if content:
            # Truncate long content
            if len(content) > 500:
                print(content[:500] + "...[truncated]")
            else:
                print(content)
        
        if tool_calls:
            for tc in tool_calls:
                tool_name = tc.get('name', str(tc))
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                
                # Check for batched transformations
                if tool_name == "apply_transformations_tool":
                    args = tc.get('args', {})
                    transforms = args.get('transformations', [])
                    print(f"\n  🔧 BATCHED TRANSFORMATIONS ({len(transforms)} operations):")
                    for t in transforms:
                        t_name = t.get('tool_name', 'unknown')
                        t_params = {k: v for k, v in t.items() if k != 'tool_name'}
                        print(f"     - {t_name}: {t_params}")
                        batched_transforms.append(t_name)
                else:
                    print(f"  Tool call: {tool_name}")
    
    # Summary of tool usage
    print("\n" + "="*60)
    print("[4.5] Tool Usage Summary")
    print("="*60)
    print(f"Tool call counts: {tool_call_counts}")
    print(f"Total batched transformations: {len(batched_transforms)}")
    if batched_transforms:
        print(f"Batched ops: {batched_transforms}")
    
    # Check final result
    print("\n" + "="*60)
    print("[5] Final Result")
    print("="*60)
    
    # Look for the cleaned dataset ref in the last tool message
    cleaned_ref = None
    for msg in reversed(messages):
        content = getattr(msg, "content", "") or ""
        if "CLEANING COMPLETE" in content:
            print(f"\n✅ Cleaning completed!")
            print(content)
            # Extract ref
            import re
            match = re.search(r"Cleaned dataset: `([^`]+)`", content)
            if match:
                cleaned_ref = match.group(1)
            break
    
    if cleaned_ref:
        print(f"\n[6] Cleaned dataset: `{cleaned_ref}`")
        cleaned_df = get_registered_dataset(cleaned_ref)
        if cleaned_df is not None:
            print(f"   Shape: {cleaned_df.shape}")
            print(f"   Nulls remaining: {cleaned_df.isna().sum().sum()}")
            print(f"   Columns: {list(cleaned_df.columns)}")
        else:
            print("   ⚠️ Could not retrieve cleaned dataset")
    else:
        print("\n⚠️ Cleaning did not complete - check messages above")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60 + "\n")
    
    return result


if __name__ == "__main__":
    test_cleaning_simple_full_flow()
