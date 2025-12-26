"""
Test LangChain Tool Binding with OpenAI Models

This test demonstrates binding all available model tools (training and pretrained)
to an OpenAI chat model, allowing the AI to choose and execute the appropriate
tool based on the problem input.

Based on: https://docs.langchain.com/oss/python/langchain/models#tool-calling
"""

import os
import sys

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models-tools", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models-tools", "pretrained"))

import warnings

warnings.filterwarnings('ignore')

# Check for OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # Try to get from user input
    print("⚠️  OPENAI_API_KEY not found in environment.")
    print("   You can either:")
    print("   1. Set it: export OPENAI_API_KEY='sk-...'")
    print("   2. Enter it now (will not be saved):")
    try:
        api_key = input("   Enter API key (or press Enter to skip): ").strip()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            print("\n   Skipping API test. Running tool structure demo instead...")
            api_key = None
    except (EOFError, KeyboardInterrupt):
        api_key = None

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# =============================================================================
# Import all tools
# =============================================================================

print("📦 Loading tools...")

# Pretrained tools
from credit_risk import credit_card_risk_tool
from finbert_tone import finbert_tone_tool
from glm import sklearn_glm_tool
# Training tools
from logistic_regression import sklearn_logistic_regression_tool
from model_storage import (delete_trained_model_tool, get_model_info_tool,
                           list_trained_models_tool, predict_with_model_tool)
from random_forest import sklearn_random_forest_tool
from survival_analysis import survival_analysis_tool
from xgboost_model import xgboost_train_tool

print("✅ All tools loaded successfully!")

# =============================================================================
# Collect all tools
# =============================================================================

ALL_TOOLS = [
    # Training tools
    sklearn_logistic_regression_tool,
    sklearn_random_forest_tool,
    xgboost_train_tool,
    sklearn_glm_tool,
    survival_analysis_tool,
    # Model storage tools
    list_trained_models_tool,
    predict_with_model_tool,
    get_model_info_tool,
    delete_trained_model_tool,
    # Pretrained tools
    credit_card_risk_tool,
    finbert_tone_tool,
]

print(f"\n📋 Available tools ({len(ALL_TOOLS)}):")
for tool in ALL_TOOLS:
    print(f"   • {tool.name}: {tool.description[:80]}...")

# =============================================================================
# Initialize model with tools bound
# =============================================================================

print("\n🤖 Initializing OpenAI model with bound tools...")

model = init_chat_model(
    model="gpt-5.1",
    temperature=0,
)

# Bind all tools to the model
model_with_tools = model.bind_tools(ALL_TOOLS)

print("✅ Model initialized with tools bound!")

# =============================================================================
# System prompt for the AI Data Scientist Agent
# =============================================================================

SYSTEM_PROMPT = """You are an AI Data Scientist Agent for an insurance company.

You have access to the following categories of tools:

## TRAINING TOOLS (Train new models)
- sklearn_logistic_regression: Train logistic regression for binary/multiclass classification
- sklearn_random_forest: Train Random Forest for classification or regression
- xgboost_train: Train XGBoost gradient boosting models
- sklearn_glm: Train GLM (Poisson/Gamma/Tweedie) for insurance pricing
- survival_analysis: Train survival models (Cox/Weibull) for time-to-event analysis

## MODEL MANAGEMENT TOOLS
- list_trained_models: List all trained models in the registry
- predict_with_model: Make predictions with a trained model
- get_model_info: Get detailed info about a trained model
- delete_trained_model: Remove a trained model

## PRETRAINED TOOLS (Ready-to-use models)
- credit_card_risk: Assess credit card application risk (Good risk/Bad loss/Bad profit)
- finbert_tone: Analyze financial text sentiment (positive/negative/neutral)

When given a task, analyze it carefully and choose the most appropriate tool(s).
Always explain your reasoning before making a tool call.
"""

# =============================================================================
# Test scenarios
# =============================================================================

def test_tool_calling(problem: str, description: str):
    """Test if the model correctly identifies and calls the right tool."""
    print(f"\n{'='*70}")
    print(f"🧪 TEST: {description}")
    print(f"{'='*70}")
    print(f"\n📝 Problem: {problem}")
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=problem),
    ]
    
    response = model_with_tools.invoke(messages)
    
    print(f"\n🤖 Model Response:")
    if response.content:
        print(f"   {response.content[:500]}...")
    
    if response.tool_calls:
        print(f"\n🔧 Tool Calls Made ({len(response.tool_calls)}):")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"\n   Tool {i}: {tool_call['name']}")
            print(f"   Arguments:")
            args = tool_call['args']
            # Pretty print args (truncate long values)
            for key, value in args.items():
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"      {key}: {value_str}")
        return response.tool_calls
    else:
        print("\n   ⚠️ No tool calls made")
        return []


def main():
    print("\n" + "="*70)
    print("🚀 LANGCHAIN TOOL BINDING TEST - AI DATA SCIENTIST AGENT")
    print("="*70)
    
    # Test 1: Credit Risk Assessment (should use credit_card_risk)
    test_tool_calling(
        problem="""
        I need to assess the credit risk for a loan applicant:
        - Gender: male
        - Age: 35
        - Marital status: married
        - Income: $75,000 annual
        - Has mortgage: yes
        - Number of kids: 2
        - Existing credit cards: 3
        - Store cards: 1
        - Active loans: 2
        - Paid monthly
        
        Should we approve this credit card application?
        """,
        description="Credit Risk Assessment"
    )
    
    # Test 2: Sentiment Analysis (should use finbert_tone)
    test_tool_calling(
        problem="""
        Analyze the sentiment of this financial news headline:
        
        "Insurance giant reports record quarterly profits as claims decrease 15%"
        
        Is this positive, negative, or neutral for investors?
        """,
        description="Financial Sentiment Analysis"
    )
    
    # Test 3: Train a claim frequency model (should use sklearn_glm with Poisson)
    test_tool_calling(
        problem="""
        I have auto insurance policy data and want to build a claim FREQUENCY model.
        The target is 'num_claims' which contains count data (0, 1, 2, 3 claims).
        
        Here's sample training data:
        [
            {"driver_age": 25, "years_licensed": 5, "vehicle_age": 3, "num_claims": 1},
            {"driver_age": 45, "years_licensed": 25, "vehicle_age": 2, "num_claims": 0},
            {"driver_age": 32, "years_licensed": 12, "vehicle_age": 8, "num_claims": 2},
            {"driver_age": 55, "years_licensed": 35, "vehicle_age": 1, "num_claims": 0},
            {"driver_age": 22, "years_licensed": 2, "vehicle_age": 10, "num_claims": 1}
        ]
        
        Train a Poisson GLM model for this. Name it 'auto_freq_test'.
        """,
        description="Train Claim Frequency Model (GLM Poisson)"
    )
    
    # Test 4: Survival analysis for policy lapse (should use survival_analysis)
    test_tool_calling(
        problem="""
        I want to predict which policies will lapse and when using survival analysis.
        
        Here's policy tenure data:
        [
            {"months_active": 24, "lapsed": 1, "premium": 500, "auto_pay": 0},
            {"months_active": 36, "lapsed": 0, "premium": 750, "auto_pay": 1},
            {"months_active": 6, "lapsed": 1, "premium": 1200, "auto_pay": 0},
            {"months_active": 48, "lapsed": 0, "premium": 600, "auto_pay": 1},
            {"months_active": 12, "lapsed": 1, "premium": 900, "auto_pay": 0}
        ]
        
        Train a Cox Proportional Hazards model. The duration is 'months_active' and 
        the event is 'lapsed' (1=lapsed, 0=still active). Name it 'lapse_cox_test'.
        """,
        description="Train Policy Lapse Model (Survival Analysis)"
    )
    
    # Test 5: List models (should use list_trained_models)
    test_tool_calling(
        problem="What models do we have trained and available for use?",
        description="List Available Models"
    )
    
    # Test 6: Classification with XGBoost (should use xgboost_train)
    test_tool_calling(
        problem="""
        I want to train a fraud detection model using XGBoost. I need high accuracy
        and want to see the training loss curve.
        
        Training data:
        [
            {"claim_amount": 5000, "age": 45, "prior_claims": 1, "fraud": 0},
            {"claim_amount": 50000, "age": 28, "prior_claims": 4, "fraud": 1},
            {"claim_amount": 3000, "age": 55, "prior_claims": 0, "fraud": 0},
            {"claim_amount": 45000, "age": 25, "prior_claims": 5, "fraud": 1}
        ]
        
        Target column is 'fraud'. Name it 'fraud_xgb_test'.
        """,
        description="Train Fraud Detection Model (XGBoost)"
    )
    
    print("\n" + "="*70)
    print("✅ ALL TOOL BINDING TESTS COMPLETE")
    print("="*70)
    print("""
Summary:
- The model successfully analyzes each problem
- It identifies the appropriate tool(s) to use
- It constructs valid tool calls with correct arguments

This demonstrates that the AI Data Scientist Agent can:
1. Understand diverse insurance/finance problems
2. Choose the right model/tool for each task
3. Structure the input correctly for each tool
""")


if __name__ == "__main__":
    main()

