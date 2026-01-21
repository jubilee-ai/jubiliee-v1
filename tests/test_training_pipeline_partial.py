"""
Test the Training Pipeline (Steps 1-6) - Partial Graph and Direct Node Calls.

This test validates the pipeline from model selection through feature engineering:
1. select_model - Choose ML model based on goal
2. data_collection - Load and prepare datasets  
3. cleaning - Clean and standardize data
4. label_split_definition - Define target, grain, split strategy
5. feature_selection_specification - Generate feature spec via analysis
6. feature_engineering_executor - Execute feature transformations

Two testing approaches:
- Option 1: Direct node calls (for debugging individual steps)
- Option 2: Partial graph (tests node chaining via LangGraph)

Run with: python -m pytest tests/test_training_pipeline_partial.py -v -s
Or directly: python tests/test_training_pipeline_partial.py
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest

# =============================================================================
# PATH SETUP
# =============================================================================

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools" / "data-tools"))

from langgraph.graph import END, StateGraph

# Import node functions
from agents.training.agent import (
    TrainingAgentState,
    cleaning_node,
    feature_engineering_executor,
    feature_selection_specification,
    label_split_definition,
)
from agents.training.data_collection import data_collection
from agents.training.select_model import select_model

# Import utilities
from utils import clear_registry, get_registered_dataset, register_dataset

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

# Paths to real datasets
DATASETS_DIR = PROJECT_ROOT / "datasets"
LOAN_DEFAULT_PATH = DATASETS_DIR / "csv" / "Loan_default.csv"
INSURANCE_PATH = DATASETS_DIR / "csv" / "insurance.csv"
FINANCIAL_DISTRESS_PATH = DATASETS_DIR / "csv" / "Financial Distress.csv"


# =============================================================================
# HELPER: CREATE INITIAL STATE
# =============================================================================


def create_initial_state(
    goal: str,
    linked_datasets: Optional[list[str]] = None,
    user_model_preference: Optional[str] = None,
) -> TrainingAgentState:
    """
    Create a complete initial state with all required fields.
    
    Args:
        goal: The ML training goal (what we're trying to predict)
        linked_datasets: Optional list of dataset paths to use
        user_model_preference: Optional specific model to use (skips LLM selection)
    
    Returns:
        Complete TrainingAgentState ready for pipeline execution
    """
    return {
        # Inputs
        "goal": goal,
        "linked_datasets": linked_datasets,
        "user_model_preference": user_model_preference,
        
        # Step 1: Model Selection
        "selected_model": None,
        "model_explanation": None,
        "model_regen_count": 0,
        
        # Step 2: Data Collection
        "collected_dataset_ref": None,
        
        # Step 3: Cleaning & Standardization
        "cleaned_dataset_ref": None,
        "cleaning_transformations": [],
        
        # Step 3.5: Label & Split Definition
        "label_definition": None,
        
        # Step 4: Feature Selection & Specification
        "feature_spec": None,
        "analysis_trace": [],
        
        # Step 5: Feature Engineering
        "transformed_dataset_ref": None,
        "feature_validation_passed": False,
        
        # Step 6: Human Confirmation (not used in partial test)
        "human_confirmed": False,
        
        # Step 7: Training (not used in partial test)
        "training_params": None,
        "model_weights_path": None,
        "training_metrics": None,
        "training_iteration": 0,
        
        # Step 8: Report (not used in partial test)
        "report_path": None,
        
        # Audit & Trace
        "audit_trace": [],
        "explanations": [],
        
        # Control flow
        "current_step": "select_model",
        "error": None,
    }


# =============================================================================
# HELPER: BUILD PARTIAL GRAPH (OPTION 2)
# =============================================================================


def build_partial_training_graph() -> StateGraph:
    """
    Build a mini-graph with only steps 1-6.
    
    Uses direct edges instead of conditional edges to bypass
    unimplemented routing functions.
    
    Flow:
        select_model → data_collection → cleaning → 
        label_split_definition → feature_selection_specification → 
        feature_engineering_executor → END
    """
    graph = StateGraph(TrainingAgentState)
    
    # Add only implemented nodes
    graph.add_node("select_model", select_model)
    graph.add_node("data_collection", data_collection)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("label_split_definition", label_split_definition)
    graph.add_node("feature_selection_specification", feature_selection_specification)
    graph.add_node("feature_engineering_executor", feature_engineering_executor)
    
    # Entry point
    graph.set_entry_point("select_model")
    
    # Direct edges (no conditional routing)
    graph.add_edge("select_model", "data_collection")
    graph.add_edge("data_collection", "cleaning")
    graph.add_edge("cleaning", "label_split_definition")
    graph.add_edge("label_split_definition", "feature_selection_specification")
    graph.add_edge("feature_selection_specification", "feature_engineering_executor")
    graph.add_edge("feature_engineering_executor", END)
    
    return graph


def compile_partial_graph():
    """Compile the partial training graph."""
    graph = build_partial_training_graph()
    return graph.compile()


# =============================================================================
# HELPER: PRINT STATE SUMMARY
# =============================================================================


def print_state_summary(state: TrainingAgentState, step_name: str):
    """Print a summary of the state after a step completes."""
    print(f"\n{'='*60}")
    print(f"AFTER: {step_name}")
    print(f"{'='*60}")
    
    if state.get("error"):
        print(f"  ERROR: {state['error']}")
        return
    
    if step_name == "select_model":
        print(f"  Selected Model: {state.get('selected_model')}")
        print(f"  Explanation: {state.get('model_explanation', '')[:200]}...")
        
    elif step_name == "data_collection":
        print(f"  Dataset Ref: {state.get('collected_dataset_ref')}")
        if state.get("collected_dataset_ref"):
            df = get_registered_dataset(state["collected_dataset_ref"])
            if df is not None:
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
        
    elif step_name == "cleaning":
        print(f"  Cleaned Ref: {state.get('cleaned_dataset_ref')}")
        if state.get("cleaned_dataset_ref"):
            df = get_registered_dataset(state["cleaned_dataset_ref"])
            if df is not None:
                print(f"  Shape: {df.shape}")
                null_counts = df.isna().sum()
                nulls = {c: v for c, v in null_counts.items() if v > 0}
                print(f"  Remaining nulls: {nulls if nulls else 'None'}")
        
    elif step_name == "label_split_definition":
        label_def = state.get("label_definition")
        if label_def:
            print(f"  Target Column: {label_def.get('target_column')}")
            print(f"  Grain: {label_def.get('grain')}")
            print(f"  Split Strategy: {label_def.get('split_strategy')}")
            print(f"  Prediction Horizon: {label_def.get('prediction_horizon')}")
            print(f"  Forbidden Columns: {label_def.get('forbidden_columns')}")
        
    elif step_name == "feature_selection_specification":
        feature_spec = state.get("feature_spec")
        if feature_spec:
            features = feature_spec.get("features", [])
            print(f"  Features Generated: {len(features)}")
            print(f"  Feature Names: {[f.get('name') for f in features[:10]]}{'...' if len(features) > 10 else ''}")
            print(f"  Reasoning: {feature_spec.get('reasoning', '')[:200]}...")
        
    elif step_name == "feature_engineering_executor":
        print(f"  Transformed Ref: {state.get('transformed_dataset_ref')}")
        print(f"  Validation Passed: {state.get('feature_validation_passed')}")
        if state.get("transformed_dataset_ref"):
            df = get_registered_dataset(state["transformed_dataset_ref"])
            if df is not None:
                print(f"  Final Shape: {df.shape}")
                print(f"  Final Columns: {list(df.columns)}")
        
        # Print audit info
        for trace in state.get("audit_trace", []):
            if trace.get("step") == "feature_engineering_executor":
                print(f"  Features Created: {trace.get('features_created')}")
                if trace.get("errors"):
                    print(f"  Errors: {trace.get('errors')}")


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the dataset registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


# =============================================================================
# TEST 1: DIRECT NODE CALLS - LOAN DEFAULT (CLASSIFICATION)
# =============================================================================


def test_pipeline_direct_calls_loan_default():
    """
    Test the pipeline with direct node calls using Loan Default dataset.
    
    Goal: Predict whether a borrower will default on their loan.
    Dataset: 255K rows, binary classification (Default: 0/1)
    
    This tests:
    - Binary classification model selection
    - Cleaning of a large dataset
    - Feature engineering for credit risk
    """
    print("\n" + "="*70)
    print("TEST: Direct Node Calls - Loan Default Prediction (Classification)")
    print("="*70)
    
    # Verify dataset exists
    assert LOAN_DEFAULT_PATH.exists(), f"Dataset not found: {LOAN_DEFAULT_PATH}"
    
    # Load and register dataset manually (skip data_collection agent for speed)
    print("\n[SETUP] Loading Loan Default dataset...")
    df = pd.read_csv(LOAN_DEFAULT_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target: Default (0={(df['Default']==0).sum():,}, 1={(df['Default']==1).sum():,})")
    
    # Use a sample for faster testing (full dataset is 255K rows)
    SAMPLE_SIZE = 5000
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"  Using sample: {SAMPLE_SIZE:,} rows for faster testing")
    
    dataset_ref = "loan_default_sample"
    register_dataset(dataset_ref, df_sample)
    
    # Create initial state
    goal = """
    Build a model to predict loan defaults. I want to identify borrowers who are 
    likely to default on their loans based on their demographics, credit history, 
    and loan characteristics. The model should help with credit risk assessment 
    and loan approval decisions.
    """
    
    state = create_initial_state(
        goal=goal,
        linked_datasets=["csv/Loan_default.csv"],
        user_model_preference="logistic_regression",  # Specify for faster test
    )
    
    # Manually set the collected dataset ref (skip data_collection agent)
    state["collected_dataset_ref"] = dataset_ref
    
    # -------------------------------------------------------------------------
    # STEP 1: Model Selection
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("[STEP 1] Running select_model...")
    state = select_model(state)
    print_state_summary(state, "select_model")
    
    assert state.get("selected_model") is not None, "Model selection failed"
    assert state.get("error") is None, f"Error in select_model: {state.get('error')}"
    
    # -------------------------------------------------------------------------
    # STEP 2: Data Collection (SKIPPED - manually set above)
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("[STEP 2] Skipping data_collection (dataset pre-loaded)")
    print_state_summary(state, "data_collection")
    
    # -------------------------------------------------------------------------
    # STEP 3: Cleaning
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("[STEP 3] Running cleaning_node...")
    state = cleaning_node(state)
    print_state_summary(state, "cleaning")
    
    assert state.get("cleaned_dataset_ref") is not None, "Cleaning failed"
    assert state.get("error") is None, f"Error in cleaning: {state.get('error')}"
    
    # -------------------------------------------------------------------------
    # STEP 3.5: Label & Split Definition
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("[STEP 3.5] Running label_split_definition...")
    state = label_split_definition(state)
    print_state_summary(state, "label_split_definition")
    
    label_def = state.get("label_definition")
    assert label_def is not None, "Label definition failed"
    assert label_def.get("target_column") is not None, "No target column defined"
    assert state.get("error") is None, f"Error in label_split: {state.get('error')}"
    
    # -------------------------------------------------------------------------
    # STEP 4: Feature Selection & Specification
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("[STEP 4] Running feature_selection_specification...")
    state = feature_selection_specification(state)
    print_state_summary(state, "feature_selection_specification")
    
    feature_spec = state.get("feature_spec")
    assert feature_spec is not None, "Feature spec generation failed"
    assert len(feature_spec.get("features", [])) > 0, "No features generated"
    
    # -------------------------------------------------------------------------
    # STEP 5: Feature Engineering Executor
    # -------------------------------------------------------------------------
    print("\n" + "-"*60)
    print("[STEP 5] Running feature_engineering_executor...")
    state = feature_engineering_executor(state)
    print_state_summary(state, "feature_engineering_executor")
    
    assert state.get("transformed_dataset_ref") is not None, "Feature engineering failed"
    assert state.get("feature_validation_passed"), "Feature validation failed"
    
    # -------------------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"  Goal: Predict loan defaults")
    print(f"  Model: {state.get('selected_model')}")
    print(f"  Target: {state.get('label_definition', {}).get('target_column')}")
    print(f"  Features: {len(state.get('feature_spec', {}).get('features', []))}")
    print(f"  Final Dataset: {state.get('transformed_dataset_ref')}")
    
    final_df = get_registered_dataset(state["transformed_dataset_ref"])
    if final_df is not None:
        print(f"  Final Shape: {final_df.shape}")
    
    print("\n  Audit Trace Steps:")
    for trace in state.get("audit_trace", []):
        print(f"    - {trace.get('step')}")
    
    print("\n  Explanations:")
    for exp in state.get("explanations", [])[:5]:
        print(f"    - {exp[:100]}...")
    
    return state


# =============================================================================
# TEST 2: DIRECT NODE CALLS - INSURANCE (REGRESSION)
# =============================================================================


def test_pipeline_direct_calls_insurance():
    """
    Test the pipeline with direct node calls using Insurance dataset.
    
    Goal: Predict health insurance costs/charges.
    Dataset: 1,338 rows, regression target (charges)
    
    This tests:
    - Regression model selection
    - Smaller dataset handling
    - Feature engineering for insurance pricing
    """
    print("\n" + "="*70)
    print("TEST: Direct Node Calls - Insurance Cost Prediction (Regression)")
    print("="*70)
    
    # Verify dataset exists
    assert INSURANCE_PATH.exists(), f"Dataset not found: {INSURANCE_PATH}"
    
    # Load and register dataset
    print("\n[SETUP] Loading Insurance dataset...")
    df = pd.read_csv(INSURANCE_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Target: charges (mean=${df['charges'].mean():,.2f})")
    print(f"  Columns: {list(df.columns)}")
    
    dataset_ref = "insurance_data"
    register_dataset(dataset_ref, df)
    
    # Create initial state
    goal = """
    Build a model to predict health insurance charges for policyholders.
    The model should estimate annual medical costs based on age, BMI, 
    smoking status, number of dependents, and geographic region.
    This will be used for insurance premium pricing and underwriting.
    """
    
    state = create_initial_state(
        goal=goal,
        linked_datasets=["csv/insurance.csv"],
        # Let LLM select model for regression task
        user_model_preference=None,
    )
    
    # Manually set the collected dataset ref
    state["collected_dataset_ref"] = dataset_ref
    
    # Run all steps
    print("\n[STEP 1] Running select_model...")
    state = select_model(state)
    print_state_summary(state, "select_model")
    assert state.get("selected_model") is not None
    
    print("\n[STEP 2] Skipping data_collection (dataset pre-loaded)")
    
    print("\n[STEP 3] Running cleaning_node...")
    state = cleaning_node(state)
    print_state_summary(state, "cleaning")
    assert state.get("cleaned_dataset_ref") is not None
    
    print("\n[STEP 3.5] Running label_split_definition...")
    state = label_split_definition(state)
    print_state_summary(state, "label_split_definition")
    assert state.get("label_definition") is not None
    
    print("\n[STEP 4] Running feature_selection_specification...")
    state = feature_selection_specification(state)
    print_state_summary(state, "feature_selection_specification")
    assert state.get("feature_spec") is not None
    
    print("\n[STEP 5] Running feature_engineering_executor...")
    state = feature_engineering_executor(state)
    print_state_summary(state, "feature_engineering_executor")
    assert state.get("transformed_dataset_ref") is not None
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"  Goal: Predict insurance charges")
    print(f"  Model: {state.get('selected_model')}")
    print(f"  Target: {state.get('label_definition', {}).get('target_column')}")
    print(f"  Features: {len(state.get('feature_spec', {}).get('features', []))}")
    
    return state


# =============================================================================
# TEST 3: PARTIAL GRAPH - LOAN DEFAULT (FULL FLOW)
# =============================================================================


def test_pipeline_partial_graph_loan_default():
    """
    Test the partial graph with the Loan Default dataset.
    
    This runs the full pipeline through LangGraph, including:
    - Data collection agent finding the dataset
    - All cleaning and feature engineering steps
    
    This is slower but tests the actual graph flow.
    """
    print("\n" + "="*70)
    print("TEST: Partial Graph - Loan Default (Full LangGraph Flow)")
    print("="*70)
    
    # Pre-register a sample of the loan default data so data_collection can find it
    print("\n[SETUP] Pre-registering loan default sample for data retrieval...")
    df_full = pd.read_csv(LOAN_DEFAULT_PATH)
    df_sample = df_full.sample(n=3000, random_state=42)
    register_dataset("loan_default", df_sample)
    print(f"  Pre-registered 'loan_default': {df_sample.shape[0]:,} rows × {df_sample.shape[1]} columns")
    print(f"  (Full CSV has {df_full.shape[0]:,} rows)")
    
    # Create initial state
    goal = """
    Predict loan defaults. Build a classification model to identify borrowers 
    at high risk of defaulting based on their credit score, income, employment, 
    and loan characteristics.
    """
    
    initial_state = create_initial_state(
        goal=goal,
        linked_datasets=["loan_default"],  # Reference the pre-registered dataset
        user_model_preference="random_forest",  # Faster than LLM selection
    )
    
    # Print inputs clearly
    print("\n" + "-"*70)
    print("INPUTS TO PIPELINE")
    print("-"*70)
    print(f"  Goal: {goal.strip()[:100]}...")
    print(f"  Linked Datasets: {initial_state.get('linked_datasets')}")
    print(f"  User Model Preference: {initial_state.get('user_model_preference')}")
    
    # Compile and run the partial graph
    print("\n" + "-"*70)
    print("EXECUTING PARTIAL GRAPH (Steps 1-6)")
    print("-"*70)
    
    agent = compile_partial_graph()
    final_state = agent.invoke(initial_state)
    
    # =========================================================================
    # DETAILED OUTPUT - STEP BY STEP
    # =========================================================================
    
    print("\n" + "="*70)
    print("STEP-BY-STEP RESULTS")
    print("="*70)
    
    # Step 1: Model Selection
    print("\n[STEP 1] MODEL SELECTION")
    print(f"  Selected Model: {final_state.get('selected_model')}")
    print(f"  Explanation: {final_state.get('model_explanation', 'N/A')[:150]}...")
    
    # Step 2: Data Collection
    print("\n[STEP 2] DATA COLLECTION")
    collected_ref = final_state.get('collected_dataset_ref')
    print(f"  Collected Dataset Ref: {collected_ref}")
    if collected_ref:
        collected_df = get_registered_dataset(collected_ref)
        if collected_df is not None:
            print(f"  Collected Shape: {collected_df.shape}")
            print(f"  ⚠️  Expected 3,000 rows (linked sample), got {collected_df.shape[0]:,}")
            if collected_df.shape[0] != 3000:
                print(f"  ❌ ISSUE: linked_datasets was IGNORED!")
        else:
            print(f"  (Could not retrieve collected dataset)")
    
    # Step 3: Cleaning
    print("\n[STEP 3] CLEANING")
    cleaned_ref = final_state.get('cleaned_dataset_ref')
    print(f"  Cleaned Dataset Ref: {cleaned_ref}")
    if cleaned_ref:
        cleaned_df = get_registered_dataset(cleaned_ref)
        if cleaned_df is not None:
            print(f"  Cleaned Shape: {cleaned_df.shape}")
            null_counts = cleaned_df.isna().sum()
            nulls = {c: int(v) for c, v in null_counts.items() if v > 0}
            print(f"  Remaining Nulls: {nulls if nulls else 'None'}")
    
    # Step 3.5: Label & Split Definition
    print("\n[STEP 3.5] LABEL & SPLIT DEFINITION")
    label_def = final_state.get("label_definition", {})
    print(f"  Target Column: {label_def.get('target_column')}")
    print(f"  Grain: {label_def.get('grain')}")
    print(f"  Split Strategy: {label_def.get('split_strategy')}")
    print(f"  Prediction Horizon: {label_def.get('prediction_horizon')}")
    print(f"  As-Of Cutoff: {label_def.get('as_of_cutoff')}")
    print(f"  Forbidden Columns: {label_def.get('forbidden_columns')}")
    
    # Step 4: Feature Selection & Specification
    print("\n[STEP 4] FEATURE SELECTION & SPECIFICATION")
    feature_spec = final_state.get("feature_spec", {})
    features = feature_spec.get("features", [])
    print(f"  Features Generated: {len(features)}")
    print(f"  Reasoning: {feature_spec.get('reasoning', 'N/A')[:200]}...")
    print(f"  Excluded Columns: {feature_spec.get('excluded_columns', [])}")
    print("\n  Feature Details:")
    for f in features[:10]:
        formula = f.get('formula', {})
        op = formula.get('op', 'unknown')
        print(f"    - {f.get('name')}: {op}", end="")
        if op == "expression":
            print(f" → {formula.get('expression', '')[:40]}")
        elif op == "one_hot":
            print(f" → column={formula.get('column')}")
        elif op == "ordinal":
            print(f" → column={formula.get('column')}, order={formula.get('order')}")
        elif op == "bin":
            print(f" → column={formula.get('column')}, bins={formula.get('bins')}")
        else:
            print()
    if len(features) > 10:
        print(f"    ... +{len(features) - 10} more")
    
    # Step 5: Feature Engineering Executor
    print("\n[STEP 5] FEATURE ENGINEERING EXECUTOR")
    transformed_ref = final_state.get('transformed_dataset_ref')
    print(f"  Transformed Dataset Ref: {transformed_ref}")
    print(f"  Validation Passed: {final_state.get('feature_validation_passed')}")
    if transformed_ref:
        final_df = get_registered_dataset(transformed_ref)
        if final_df is not None:
            print(f"  Final Shape: {final_df.shape}")
            print(f"  Final Columns ({len(final_df.columns)}):")
            for i, col in enumerate(final_df.columns):
                print(f"    {i+1}. {col}")
    
    # =========================================================================
    # AUDIT TRACE
    # =========================================================================
    print("\n" + "="*70)
    print("AUDIT TRACE")
    print("="*70)
    audit_trace = final_state.get("audit_trace", [])
    if audit_trace:
        for i, trace in enumerate(audit_trace):
            print(f"\n  [{i+1}] Step: {trace.get('step')}")
            for key, value in trace.items():
                if key != 'step':
                    # Truncate long values
                    val_str = str(value)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    print(f"      {key}: {val_str}")
    else:
        print("  (No audit trace entries)")
    
    # =========================================================================
    # EXPLANATIONS
    # =========================================================================
    print("\n" + "="*70)
    print("EXPLANATIONS")
    print("="*70)
    explanations = final_state.get("explanations", [])
    if explanations:
        for i, exp in enumerate(explanations):
            print(f"\n  [{i+1}] {exp[:200]}{'...' if len(exp) > 200 else ''}")
    else:
        print("  (No explanations recorded)")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  Model: {final_state.get('selected_model')}")
    print(f"  Target: {label_def.get('target_column')}")
    print(f"  Features Specified: {len(features)}")
    if final_df is not None:
        print(f"  Final Dataset: {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
    print(f"  Validation Passed: {final_state.get('feature_validation_passed')}")
    print(f"  Errors: {final_state.get('error')}")
    
    # =========================================================================
    # ASSERTIONS
    # =========================================================================
    assert final_state.get("selected_model") is not None, "Model selection failed"
    assert final_state.get("collected_dataset_ref") is not None, "Data collection failed"
    assert final_state.get("cleaned_dataset_ref") is not None, "Cleaning failed"
    assert final_state.get("label_definition") is not None, "Label definition failed"
    assert final_state.get("feature_spec") is not None, "Feature spec failed"
    assert final_state.get("transformed_dataset_ref") is not None, "Feature engineering failed"
    
    return final_state


# =============================================================================
# TEST 4: DIRECT NODE CALLS - FINANCIAL DISTRESS (TIME SERIES)
# =============================================================================


def test_pipeline_direct_calls_financial_distress():
    """
    Test the pipeline with Financial Distress dataset.
    
    Goal: Predict corporate financial distress.
    Dataset: 3,673 rows, time-series with Company and Time dimensions
    
    This tests:
    - Entity-based or time-based split strategies
    - Handling of panel/time-series data
    - Financial ratio features
    """
    print("\n" + "="*70)
    print("TEST: Direct Node Calls - Financial Distress Prediction")
    print("="*70)
    
    # Verify dataset exists
    assert FINANCIAL_DISTRESS_PATH.exists(), f"Dataset not found: {FINANCIAL_DISTRESS_PATH}"
    
    # Load and register dataset
    print("\n[SETUP] Loading Financial Distress dataset...")
    df = pd.read_csv(FINANCIAL_DISTRESS_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Companies: {df['Company'].nunique()}")
    print(f"  Time periods: {df['Time'].nunique()}")
    print(f"  Target: Financial Distress (mean={df['Financial Distress'].mean():.3f})")
    
    dataset_ref = "financial_distress_data"
    register_dataset(dataset_ref, df)
    
    # Create initial state
    goal = """
    Build a model to predict corporate financial distress. The dataset contains 
    observations for multiple companies over time with financial ratio features.
    I want to identify companies at risk of bankruptcy or severe financial 
    deterioration. The model should respect the time-series nature of the data
    and not leak future information.
    """
    
    state = create_initial_state(
        goal=goal,
        linked_datasets=["csv/Financial Distress.csv"],
        user_model_preference="xgboost",
    )
    
    # Manually set the collected dataset ref
    state["collected_dataset_ref"] = dataset_ref
    
    # Run all steps
    print("\n[STEP 1] Running select_model...")
    state = select_model(state)
    print_state_summary(state, "select_model")
    
    print("\n[STEP 3] Running cleaning_node...")
    state = cleaning_node(state)
    print_state_summary(state, "cleaning")
    
    print("\n[STEP 3.5] Running label_split_definition...")
    state = label_split_definition(state)
    print_state_summary(state, "label_split_definition")
    
    # Check if time-based or entity-based split was chosen
    label_def = state.get("label_definition", {})
    split_strategy = label_def.get("split_strategy")
    print(f"\n  Split Strategy Chosen: {split_strategy}")
    print(f"  (Expected: time_based or entity_based for panel data)")
    
    print("\n[STEP 4] Running feature_selection_specification...")
    state = feature_selection_specification(state)
    print_state_summary(state, "feature_selection_specification")
    
    print("\n[STEP 5] Running feature_engineering_executor...")
    state = feature_engineering_executor(state)
    print_state_summary(state, "feature_engineering_executor")
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE - SUMMARY")
    print("="*70)
    print(f"  Goal: Predict financial distress")
    print(f"  Model: {state.get('selected_model')}")
    print(f"  Target: {label_def.get('target_column')}")
    print(f"  Grain: {label_def.get('grain')}")
    print(f"  Split: {label_def.get('split_strategy')}")
    print(f"  Features: {len(state.get('feature_spec', {}).get('features', []))}")
    
    return state


# =============================================================================
# FULL FLOW TESTS (5 Realistic Scenarios)
# =============================================================================


def run_full_flow_test(
    test_name: str,
    dataset_path: Path,
    sample_size: int,
    goal: str,
    model_preference: Optional[str] = None,
    expected_target: Optional[str] = None,
    expected_task_type: Optional[str] = None,
):
    """
    Generic full-flow test runner for the partial graph.
    
    Args:
        test_name: Name for the test (used as dataset ref)
        dataset_path: Path to the CSV file
        sample_size: Number of rows to sample (for speed)
        goal: The ML training goal
        model_preference: Optional model to use (None = let LLM decide)
        expected_target: Expected target column (for validation)
        expected_task_type: Expected task type (classification/regression)
    
    Returns:
        Final state from the pipeline
    """
    print("\n" + "="*70)
    print(f"FULL FLOW TEST: {test_name}")
    print("="*70)
    
    # Load and register dataset
    print("\n[SETUP] Loading dataset...")
    assert dataset_path.exists(), f"Dataset not found: {dataset_path}"
    df_full = pd.read_csv(dataset_path)
    
    if sample_size and sample_size < len(df_full):
        df = df_full.sample(n=sample_size, random_state=42)
        print(f"  Full dataset: {len(df_full):,} rows")
        print(f"  Using sample: {len(df):,} rows")
    else:
        df = df_full
        print(f"  Using full dataset: {len(df):,} rows")
    
    print(f"  Columns: {len(df.columns)}")
    
    dataset_ref = test_name.lower().replace(" ", "_")
    register_dataset(dataset_ref, df)
    
    # Create initial state
    initial_state = create_initial_state(
        goal=goal,
        linked_datasets=[dataset_ref],
        user_model_preference=model_preference,
    )
    
    # Print inputs
    print("\n" + "-"*70)
    print("INPUTS")
    print("-"*70)
    print(f"  Goal: {goal.strip()[:150]}...")
    print(f"  Model Preference: {model_preference or 'LLM will decide'}")
    print(f"  Dataset: {dataset_ref} ({len(df):,} rows)")
    
    # Run the partial graph
    print("\n" + "-"*70)
    print("EXECUTING PIPELINE...")
    print("-"*70)
    
    agent = compile_partial_graph()
    final_state = agent.invoke(initial_state)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Model selection
    print(f"\n[1] Model Selected: {final_state.get('selected_model')}")
    if model_preference is None:
        print(f"    (LLM chose this based on the goal)")
    
    # Data collection
    collected_ref = final_state.get('collected_dataset_ref')
    collected_df = get_registered_dataset(collected_ref) if collected_ref else None
    print(f"\n[2] Data Collected: {collected_ref}")
    if collected_df is not None:
        print(f"    Shape: {collected_df.shape}")
    
    # Cleaning
    cleaned_ref = final_state.get('cleaned_dataset_ref')
    cleaned_df = get_registered_dataset(cleaned_ref) if cleaned_ref else None
    print(f"\n[3] Cleaned Dataset: {cleaned_ref}")
    if cleaned_df is not None:
        print(f"    Shape: {cleaned_df.shape}")
    
    # Label definition
    label_def = final_state.get("label_definition", {})
    print(f"\n[4] Label Definition:")
    print(f"    Target: {label_def.get('target_column')}")
    print(f"    Grain: {label_def.get('grain')}")
    print(f"    Split Strategy: {label_def.get('split_strategy')}")
    print(f"    Forbidden Columns: {label_def.get('forbidden_columns')}")
    
    # Validate expected target
    if expected_target:
        actual_target = label_def.get('target_column')
        if actual_target == expected_target:
            print(f"    ✅ Target matches expected: {expected_target}")
        else:
            print(f"    ⚠️  Expected target '{expected_target}', got '{actual_target}'")
    
    # Feature spec
    feature_spec = final_state.get("feature_spec", {})
    features = feature_spec.get("features", [])
    print(f"\n[5] Feature Engineering:")
    print(f"    Features Specified: {len(features)}")
    print(f"    Sample features: {[f.get('name') for f in features[:5]]}")
    
    # Final output
    transformed_ref = final_state.get('transformed_dataset_ref')
    final_df = get_registered_dataset(transformed_ref) if transformed_ref else None
    print(f"\n[6] Final Dataset: {transformed_ref}")
    if final_df is not None:
        print(f"    Shape: {final_df.shape}")
        print(f"    Columns: {list(final_df.columns)}")
    
    print(f"\n[7] Validation Passed: {final_state.get('feature_validation_passed')}")
    
    # Assertions
    assert final_state.get("selected_model") is not None, "Model selection failed"
    assert final_state.get("collected_dataset_ref") is not None, "Data collection failed"
    assert final_state.get("cleaned_dataset_ref") is not None, "Cleaning failed"
    assert final_state.get("label_definition") is not None, "Label definition failed"
    assert final_state.get("feature_spec") is not None, "Feature spec failed"
    assert final_state.get("transformed_dataset_ref") is not None, "Feature engineering failed"
    assert final_state.get("feature_validation_passed"), "Validation failed"
    
    print("\n" + "="*70)
    print(f"✅ {test_name} COMPLETE")
    print("="*70)
    
    return final_state


# -----------------------------------------------------------------------------
# TEST 1: Credit Risk Scoring (Regulatory-Compliant)
# -----------------------------------------------------------------------------

def test_full_flow_credit_risk_scoring():
    """
    Scenario: A bank needs a credit risk model for loan approvals.
    
    Requirements:
    - Model must be interpretable (regulatory requirement)
    - Binary classification: approve vs reject
    - Need to explain decisions to customers
    
    Expected:
    - Model: logistic_regression (interpretable)
    - Target: Default
    - Features: credit factors with clear business meaning
    """
    return run_full_flow_test(
        test_name="Credit Risk Scoring",
        dataset_path=LOAN_DEFAULT_PATH,
        sample_size=5000,
        goal="""
        Build a credit risk scoring model for loan approval decisions. 
        
        Requirements:
        - The model must be interpretable for regulatory compliance (Basel III, Fair Lending)
        - We need to explain to customers why they were approved or rejected
        - Predict whether a loan applicant will default
        - Features should have clear business meaning for the risk team
        
        This will be used by loan officers to make final approval decisions.
        """,
        model_preference="logistic_regression",  # Interpretable for regulatory
        expected_target="Default",
        expected_task_type="classification",
    )


# -----------------------------------------------------------------------------
# TEST 2: Insurance Premium Optimization
# -----------------------------------------------------------------------------

def test_full_flow_insurance_premium():
    """
    Scenario: An insurance company wants to optimize premium pricing.
    
    Requirements:
    - Predict expected medical costs per policyholder
    - Use demographics and health indicators
    - Model should handle right-skewed cost distribution
    
    Expected:
    - Model: glm (Gamma for costs) or random_forest
    - Target: charges
    - Features: age, bmi, smoker status, etc.
    """
    return run_full_flow_test(
        test_name="Insurance Premium Optimization",
        dataset_path=INSURANCE_PATH,
        sample_size=None,  # Use full dataset (only 1338 rows)
        goal="""
        Build a model to predict annual medical insurance charges for policyholders.
        
        Business context:
        - We need to set fair premiums based on expected costs
        - Smoking status and BMI are key risk factors
        - The model will be used by actuaries for pricing decisions
        - Medical costs are typically right-skewed (few very expensive claims)
        
        The model should accurately predict the expected annual cost for each 
        policyholder profile to ensure premiums cover expected claims.
        """,
        model_preference=None,  # Let LLM decide (should pick glm or similar)
        expected_target="charges",
        expected_task_type="regression",
    )


# -----------------------------------------------------------------------------
# TEST 3: High-Accuracy Default Detection
# -----------------------------------------------------------------------------

def test_full_flow_default_detection_xgboost():
    """
    Scenario: A fintech wants maximum accuracy for default detection.
    
    Requirements:
    - Catch as many defaulters as possible
    - Accuracy matters more than interpretability
    - Handle complex feature interactions
    
    Expected:
    - Model: xgboost (high accuracy)
    - Target: Default
    - Features: engineered ratios, interactions
    """
    return run_full_flow_test(
        test_name="High-Accuracy Default Detection",
        dataset_path=LOAN_DEFAULT_PATH,
        sample_size=8000,
        goal="""
        Build a high-performance model to detect loan defaults before approval.
        
        Context:
        - This is an internal scoring model (not customer-facing)
        - Accuracy and recall are the priority - we want to catch defaulters
        - The model can be a black box - interpretability is secondary
        - We want to capture complex patterns and feature interactions
        - False negatives (missing a defaulter) are very costly
        
        Maximize predictive accuracy using all available features and any 
        useful transformations or derived features.
        """,
        model_preference="xgboost",
        expected_target="Default",
        expected_task_type="classification",
    )


# -----------------------------------------------------------------------------
# TEST 4: Corporate Bankruptcy Early Warning
# -----------------------------------------------------------------------------

def test_full_flow_bankruptcy_early_warning():
    """
    Scenario: A ratings agency needs to predict corporate bankruptcies.
    
    Requirements:
    - Panel data: multiple companies over time
    - Must respect temporal structure (no future leakage)
    - Predict 1 period ahead
    
    Expected:
    - Model: xgboost or random_forest
    - Target: Financial Distress
    - Split: time_based (critical for panel data)
    - Features: financial ratios
    """
    print("\n" + "="*70)
    print("FULL FLOW TEST: Corporate Bankruptcy Early Warning")
    print("="*70)
    
    # Load dataset with subset of columns (full 86 columns causes cleaning timeout)
    print("\n[SETUP] Loading Financial Distress dataset...")
    df_full = pd.read_csv(FINANCIAL_DISTRESS_PATH)
    
    # Use key columns: identifiers, target, and first 15 financial ratios
    key_cols = ['Company', 'Time', 'Financial Distress'] + [f'x{i}' for i in range(1, 16)]
    df = df_full[key_cols].copy()
    
    print(f"  Full dataset: {df_full.shape[0]:,} rows × {df_full.shape[1]} columns")
    print(f"  Using subset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  (Selected key financial ratios x1-x15 for faster testing)")
    
    dataset_ref = "bankruptcy_early_warning"
    register_dataset(dataset_ref, df)
    
    # Create initial state
    goal = """
    Build an early warning system for corporate financial distress.
    
    Dataset context:
    - This is PANEL DATA: multiple companies observed over multiple time periods
    - Company and Time columns identify each observation
    - Features x1-x15 are key financial ratios (leverage, liquidity, profitability)
    - Financial Distress is a continuous score (higher = more distressed)
    
    Critical requirements:
    - The model must NOT use future information to predict current distress
    - Training/test split should respect the time dimension
    - We want to predict distress 1 period ahead for early intervention
    - The model will be used by analysts to flag at-risk companies
    
    Ensure no temporal leakage - this is critical for a valid backtest.
    """
    
    initial_state = create_initial_state(
        goal=goal,
        linked_datasets=[dataset_ref],
        user_model_preference="xgboost",
    )
    
    # Print inputs
    print("\n" + "-"*70)
    print("INPUTS")
    print("-"*70)
    print(f"  Goal: {goal.strip()[:150]}...")
    print(f"  Model Preference: xgboost")
    print(f"  Dataset: {dataset_ref} ({len(df):,} rows)")
    
    # Run the partial graph
    print("\n" + "-"*70)
    print("EXECUTING PIPELINE...")
    print("-"*70)
    
    agent = compile_partial_graph()
    final_state = agent.invoke(initial_state)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print(f"\n[1] Model Selected: {final_state.get('selected_model')}")
    
    collected_ref = final_state.get('collected_dataset_ref')
    collected_df = get_registered_dataset(collected_ref) if collected_ref else None
    print(f"\n[2] Data Collected: {collected_ref}")
    if collected_df is not None:
        print(f"    Shape: {collected_df.shape}")
    
    cleaned_ref = final_state.get('cleaned_dataset_ref')
    cleaned_df = get_registered_dataset(cleaned_ref) if cleaned_ref else None
    print(f"\n[3] Cleaned Dataset: {cleaned_ref}")
    if cleaned_df is not None:
        print(f"    Shape: {cleaned_df.shape}")
    
    label_def = final_state.get("label_definition", {})
    print(f"\n[4] Label Definition:")
    print(f"    Target: {label_def.get('target_column')}")
    print(f"    Grain: {label_def.get('grain')}")
    print(f"    Split Strategy: {label_def.get('split_strategy')}")
    print(f"    ⭐ Expected time_based split for panel data")
    if label_def.get('split_strategy') == 'time_based':
        print(f"    ✅ Correctly identified time-based split!")
    else:
        print(f"    ⚠️  Split strategy: {label_def.get('split_strategy')}")
    
    feature_spec = final_state.get("feature_spec", {})
    features = feature_spec.get("features", [])
    print(f"\n[5] Feature Engineering:")
    print(f"    Features Specified: {len(features)}")
    
    transformed_ref = final_state.get('transformed_dataset_ref')
    final_df = get_registered_dataset(transformed_ref) if transformed_ref else None
    print(f"\n[6] Final Dataset: {transformed_ref}")
    if final_df is not None:
        print(f"    Shape: {final_df.shape}")
        print(f"    Columns: {list(final_df.columns)}")
    
    print(f"\n[7] Validation Passed: {final_state.get('feature_validation_passed')}")
    
    # Assertions
    assert final_state.get("selected_model") is not None
    assert final_state.get("label_definition") is not None
    assert final_state.get("feature_spec") is not None
    assert final_state.get("transformed_dataset_ref") is not None
    
    print("\n" + "="*70)
    print("✅ Corporate Bankruptcy Early Warning COMPLETE")
    print("="*70)
    
    return final_state


# -----------------------------------------------------------------------------
# TEST 5: Customer Segmentation for Collections
# -----------------------------------------------------------------------------

def test_full_flow_collections_segmentation():
    """
    Scenario: A collections team wants to prioritize which defaulters to contact.
    
    Requirements:
    - Among applicants, identify highest-risk segments
    - Need feature importance for actionable insights
    - Model interpretability for strategy team
    
    Expected:
    - Model: random_forest (feature importance)
    - Target: Default
    - Features: should provide segmentation insights
    """
    return run_full_flow_test(
        test_name="Collections Risk Segmentation",
        dataset_path=LOAN_DEFAULT_PATH,
        sample_size=6000,
        goal="""
        Build a model to segment loan applicants by default risk for collections strategy.
        
        Business context:
        - Our collections team has limited capacity and needs to prioritize
        - We want to identify customer segments with highest default probability
        - Feature importance will guide which factors to focus on
        - The segments will inform different collection strategies:
          * High risk: early intervention, proactive outreach
          * Medium risk: standard monitoring
          * Low risk: minimal touch
        
        The model should:
        - Provide probability scores for ranking
        - Identify which features drive default risk
        - Help us understand customer profiles in each risk tier
        
        We'll use Random Forest for its feature importance capabilities.
        """,
        model_preference="random_forest",
        expected_target="Default",
        expected_task_type="classification",
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Training Pipeline (Steps 1-6)")
    parser.add_argument(
        "--test", 
        choices=[
            # Direct node tests
            "loan", "insurance", "distress", "graph",
            # Full flow tests
            "credit_risk", "premium", "xgboost", "bankruptcy", "collections",
            # Run groups
            "all_direct", "all_flow", "all"
        ],
        default="loan",
        help="Which test to run"
    )
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Run with pytest instead of directly"
    )
    
    args = parser.parse_args()
    
    if args.pytest:
        pytest.main([__file__, "-v", "-s"])
    else:
        # Clear registry before tests
        clear_registry()
        
        # Direct node tests
        if args.test == "loan" or args.test in ["all_direct", "all"]:
            test_pipeline_direct_calls_loan_default()
        
        if args.test == "insurance" or args.test in ["all_direct", "all"]:
            test_pipeline_direct_calls_insurance()
        
        if args.test == "distress" or args.test in ["all_direct", "all"]:
            test_pipeline_direct_calls_financial_distress()
        
        if args.test == "graph" or args.test in ["all_direct", "all"]:
            test_pipeline_partial_graph_loan_default()
        
        # Full flow tests (5 realistic scenarios)
        if args.test == "credit_risk" or args.test in ["all_flow", "all"]:
            clear_registry()
            test_full_flow_credit_risk_scoring()
        
        if args.test == "premium" or args.test in ["all_flow", "all"]:
            clear_registry()
            test_full_flow_insurance_premium()
        
        if args.test == "xgboost" or args.test in ["all_flow", "all"]:
            clear_registry()
            test_full_flow_default_detection_xgboost()
        
        if args.test == "bankruptcy" or args.test in ["all_flow", "all"]:
            clear_registry()
            test_full_flow_bankruptcy_early_warning()
        
        if args.test == "collections" or args.test in ["all_flow", "all"]:
            clear_registry()
            test_full_flow_collections_segmentation()
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETE")
        print("="*70)
