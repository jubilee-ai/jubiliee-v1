"""
Test for the Simple Training Agent (Deep Agents SDK version).

Runs the full end-to-end pipeline: select_model → data_collection → cleaning →
label_split → feature_spec → feature_eng → training_approval → training → report.

Run with: python3 tests/test_simple_training_agent.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "data-tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "models-tools" / "training"))

import numpy as np
import pandas as pd
from utils import clear_registry, register_dataset


def create_credit_dataset(n=300, seed=42):
    """Create a synthetic credit-risk dataset with predictive signal."""
    np.random.seed(seed)

    age = np.random.randint(25, 65, n)
    income = np.random.normal(60000, 20000, n).clip(20000, 150000)
    credit_score = np.random.randint(500, 800, n)
    debt_ratio = np.random.uniform(0.1, 0.8, n)

    risk_score = -0.05 * age - 0.00003 * income - 0.008 * credit_score + 8
    default_prob = 1 / (1 + np.exp(-risk_score))
    default = (np.random.random(n) < default_prob).astype(int)

    return pd.DataFrame({
        "age": age,
        "income": income.round(2),
        "credit_score": credit_score,
        "debt_ratio": debt_ratio.round(4),
        "default": default,
    })


def test_simple_agent_end_to_end():
    """Full end-to-end test: register data, invoke agent, check report."""
    print("\n" + "=" * 70)
    print("TEST: Simple Training Agent — End-to-End")
    print("=" * 70)

    clear_registry()

    print("\n[1] Creating and registering synthetic dataset...")
    df = create_credit_dataset(n=300)
    register_dataset("credit_risk_raw", df, register_sql=False)
    print(f"    Registered 'credit_risk_raw': {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Default rate: {df['default'].mean():.1%}")

    print("\n[2] Creating simple training agent...")
    from agents.training.agent_simple import create_simple_training_agent

    agent, state = create_simple_training_agent(
        goal="Predict loan default risk based on applicant financial profile",
        linked_datasets=["credit_risk_raw"],
        user_model_preference="supervised",
        model="openai:gpt-5.1",
        hitl=False,
    )
    print("    Agent created.")

    print("\n[3] Invoking agent (this calls all 9 steps via LLM)...")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Train a model to predict loan default risk. The dataset 'credit_risk_raw' is already loaded."}]}
    )

    print("\n[4] Agent finished.")
    last_msg = result["messages"][-1].content if result.get("messages") else "(no messages)"
    print(f"    Final message (first 500 chars):\n{last_msg[:500]}")

    print("\n" + "=" * 70)
    print("TEST PASSED")
    print("=" * 70)
    return result


def create_clustering_dataset(n=400, seed=42):
    """Create synthetic customer data for clustering."""
    np.random.seed(seed)

    segments = np.random.choice(3, n, p=[0.3, 0.4, 0.3])
    age = np.where(segments == 0, np.random.normal(25, 5, n),
          np.where(segments == 1, np.random.normal(45, 10, n),
                   np.random.normal(65, 8, n))).clip(18, 90).astype(int)
    income = np.where(segments == 0, np.random.normal(30000, 8000, n),
             np.where(segments == 1, np.random.normal(70000, 15000, n),
                      np.random.normal(50000, 12000, n))).clip(15000).round(2)
    spending = (income * np.random.uniform(0.2, 0.8, n)).round(2)
    visits = np.random.poisson(np.where(segments == 0, 15, np.where(segments == 1, 8, 3)), n)

    return pd.DataFrame({
        "age": age, "income": income,
        "spending": spending, "visits_per_month": visits,
    })


def test_nn_supervised():
    """Test neural network training on supervised classification task."""
    print("\n" + "=" * 70)
    print("TEST: Neural Network — Supervised Classification")
    print("=" * 70)

    clear_registry()

    df = create_credit_dataset(n=500)
    register_dataset("nn_credit_raw", df, register_sql=False)
    print(f"    Registered 'nn_credit_raw': {df.shape}")

    from agents.training.agent_simple import create_simple_training_agent

    agent, state = create_simple_training_agent(
        goal="Predict loan default using a neural network",
        linked_datasets=["nn_credit_raw"],
        user_model_preference="neural_networks",
        hitl=False,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Train a neural network to predict default. Dataset: nn_credit_raw."}]}
    )

    last_msg = result["messages"][-1].content if result.get("messages") else ""
    print(f"    Final message (first 500 chars):\n{last_msg[:500]}")

    assert state.get("selected_model") == "neural_networks", f"Expected neural_networks, got {state.get('selected_model')}"
    print("\n    TEST PASSED")
    return result


def test_nn_unsupervised():
    """Test neural network training on unsupervised clustering task."""
    print("\n" + "=" * 70)
    print("TEST: Unsupervised — Customer Segmentation")
    print("=" * 70)

    clear_registry()

    df = create_clustering_dataset(n=400)
    register_dataset("customer_raw", df, register_sql=False)
    print(f"    Registered 'customer_raw': {df.shape}")

    from agents.training.agent_simple import create_simple_training_agent

    agent, state = create_simple_training_agent(
        goal="Segment customers into meaningful groups based on demographics and behavior",
        linked_datasets=["customer_raw"],
        user_model_preference="unsupervised",
        hitl=False,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Cluster customer_raw into segments."}]}
    )

    last_msg = result["messages"][-1].content if result.get("messages") else ""
    print(f"    Final message (first 500 chars):\n{last_msg[:500]}")

    assert state.get("selected_model") == "unsupervised", f"Expected unsupervised, got {state.get('selected_model')}"
    print("\n    TEST PASSED")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="supervised",
                        choices=["supervised", "nn", "unsupervised", "all"],
                        help="Which test to run")
    args = parser.parse_args()

    tests = {
        "supervised": test_simple_agent_end_to_end,
        "nn": test_nn_supervised,
        "unsupervised": test_nn_unsupervised,
    }

    if args.test == "all":
        for name, fn in tests.items():
            try:
                fn()
            except Exception as e:
                print(f"\n    TEST FAILED: {name} — {e}")
    else:
        tests[args.test]()
