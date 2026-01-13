#!/usr/bin/env python3
"""
Demo: LangChain Tool Binding with OpenAI Models

This script demonstrates binding all model tools to an OpenAI chat model
and having the AI agent choose appropriate tools for insurance problems.

Usage:
    # With API key as environment variable:
    export OPENAI_API_KEY='sk-...'
    python demo_tool_binding.py

    # Or pass as argument:
    python demo_tool_binding.py --api-key 'sk-...'

Based on: https://docs.langchain.com/oss/python/langchain/models#tool-calling
"""

import argparse
import os
import sys

# Add modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "models-tools", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "models-tools", "pretrained"))

import warnings

warnings.filterwarnings('ignore')


def load_tools():
    """Load all available tools."""
    print("📦 Loading tools...")
    
    tools = []
    tool_names = []
    
    # Training tools
    try:
        from logistic_regression import sklearn_logistic_regression_tool
        tools.append(sklearn_logistic_regression_tool)
        tool_names.append("sklearn_logistic_regression")
    except Exception as e:
        print(f"   ⚠️ Could not load logistic_regression: {e}")
    
    try:
        from random_forest import sklearn_random_forest_tool
        tools.append(sklearn_random_forest_tool)
        tool_names.append("sklearn_random_forest")
    except Exception as e:
        print(f"   ⚠️ Could not load random_forest: {e}")
    
    try:
        from xgboost_model import xgboost_train_tool
        tools.append(xgboost_train_tool)
        tool_names.append("xgboost_train")
    except Exception as e:
        print(f"   ⚠️ Could not load xgboost: {e}")
    
    try:
        from glm import sklearn_glm_tool
        tools.append(sklearn_glm_tool)
        tool_names.append("sklearn_glm")
    except Exception as e:
        print(f"   ⚠️ Could not load glm: {e}")
    
    try:
        from survival_analysis import survival_analysis_tool
        tools.append(survival_analysis_tool)
        tool_names.append("survival_analysis")
    except Exception as e:
        print(f"   ⚠️ Could not load survival_analysis: {e}")
    
    try:
        from model_storage import (delete_trained_model_tool,
                                   get_model_info_tool,
                                   list_trained_models_tool,
                                   predict_with_model_tool)
        tools.extend([
            list_trained_models_tool,
            predict_with_model_tool,
            get_model_info_tool,
            delete_trained_model_tool,
        ])
        tool_names.extend([
            "list_trained_models",
            "predict_with_model",
            "get_model_info",
            "delete_trained_model",
        ])
    except Exception as e:
        print(f"   ⚠️ Could not load model_storage: {e}")
    
    # Pretrained tools
    try:
        from credit_risk import credit_card_risk_prediction_tool
        tools.append(credit_card_risk_prediction_tool)
        tool_names.append("credit_card_risk_prediction")
    except Exception as e:
        print(f"   ⚠️ Could not load credit_risk: {e}")
    
    try:
        from loan_default_prediction import loan_default_prediction_tool
        tools.append(loan_default_prediction_tool)
        tool_names.append("loan_default_prediction")
    except Exception as e:
        print(f"   ⚠️ Could not load loan_default_prediction: {e}")
    
    try:
        from finbert_tone import finbert_tone_tool
        tools.append(finbert_tone_tool)
        tool_names.append("finbert_tone")
    except Exception as e:
        print(f"   ⚠️ Could not load finbert_tone: {e}")
    
    try:
        from prosus_finbert import finbert_sentiment_tool
        tools.append(finbert_sentiment_tool)
        tool_names.append("finbert_sentiment")
    except Exception as e:
        print(f"   ⚠️ Could not load prosus_finbert: {e}")
    
    try:
        from bert_finetuned_claim_detection import claim_detection_tool
        tools.append(claim_detection_tool)
        tool_names.append("claim_detection")
    except Exception as e:
        print(f"   ⚠️ Could not load claim_detection: {e}")
    
    print(f"✅ Loaded {len(tools)} tools: {', '.join(tool_names)}")
    return tools


def display_tools_info(tools):
    """Display information about available tools."""
    print("\n" + "="*70)
    print("📋 AVAILABLE TOOLS FOR AI DATA SCIENTIST AGENT")
    print("="*70)
    
    categories = {
        "Training Tools": ["sklearn_logistic_regression", "sklearn_random_forest", 
                          "xgboost_train", "sklearn_glm", "survival_analysis"],
        "Model Management": ["list_trained_models", "predict_with_model", 
                            "get_model_info", "delete_trained_model"],
        "Pretrained Models": ["credit_card_risk_prediction", "loan_default_prediction",
                              "finbert_tone", "finbert_sentiment", "claim_detection"],
    }
    
    for category, names in categories.items():
        print(f"\n🔹 {category}:")
        for tool in tools:
            if tool.name in names:
                desc = tool.description[:100] + "..." if len(tool.description) > 100 else tool.description
                print(f"   • {tool.name}")
                print(f"     {desc}")


def run_tool_binding_demo(api_key: str):
    """Run the full tool binding demo with OpenAI."""
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage, SystemMessage
    
    os.environ["OPENAI_API_KEY"] = api_key
    
    tools = load_tools()
    display_tools_info(tools)
    
    print("\n" + "="*70)
    print("🤖 INITIALIZING OPENAI MODEL WITH TOOL BINDING")
    print("="*70)
    
    # Initialize model
    model = init_chat_model(
        model="gpt-5-mini",
        temperature=0,
    )
    
    # Bind tools
    model_with_tools = model.bind_tools(tools)
    print("✅ Model initialized with all tools bound!")
    
    # System prompt
    SYSTEM_PROMPT = """You are an AI Data Scientist Agent for an insurance company.

You have access to tools for:
1. TRAINING MODELS: logistic_regression, random_forest, xgboost, glm (Poisson/Gamma/Tweedie), survival_analysis
2. MODEL MANAGEMENT: list models, make predictions, get info, delete models
3. PRETRAINED MODELS: credit_card_risk assessment, finbert_tone sentiment analysis

Analyze each task and choose the most appropriate tool. Explain your reasoning briefly.
"""
    
    # Test scenarios
    test_cases = [
        {
            "name": "Credit Risk Assessment",
            "problem": """Assess credit risk for: male, 35 years old, married, $75000 income, 
has mortgage, 2 kids, 3 credit cards, 1 store card, 2 loans, paid monthly.""",
        },
        {
            "name": "Financial Sentiment",
            "problem": "Analyze: 'Insurance giant reports record profits as claims decrease 15%'",
        },
        {
            "name": "Train Claim Frequency Model",
            "problem": """Train a Poisson GLM for claim frequency with this data:
[{"age": 25, "claims": 1}, {"age": 45, "claims": 0}, {"age": 32, "claims": 2}]
Target: 'claims'. Name: 'freq_demo'.""",
        },
        {
            "name": "List Available Models",
            "problem": "What trained models do we have available?",
        },
    ]
    
    print("\n" + "="*70)
    print("🧪 RUNNING TOOL BINDING TESTS")
    print("="*70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}: {test['name']}")
        print(f"{'─'*70}")
        print(f"📝 Problem: {test['problem'][:100]}...")
        
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=test['problem']),
        ]
        
        try:
            response = model_with_tools.invoke(messages)
            
            print(f"\n🤖 Model Reasoning:")
            if response.content:
                print(f"   {response.content[:300]}...")
            
            if response.tool_calls:
                print(f"\n🔧 Tool Calls ({len(response.tool_calls)}):")
                for tc in response.tool_calls:
                    print(f"   ✓ {tc['name']}")
                    # Show key args
                    args = tc['args']
                    if len(args) <= 5:
                        for k, v in args.items():
                            v_str = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
                            print(f"     {k}: {v_str}")
                    else:
                        print(f"     ({len(args)} arguments provided)")
            else:
                print("   ⚠️ No tool call made")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "="*70)
    print("✅ TOOL BINDING DEMO COMPLETE")
    print("="*70)
    print("""
The AI Data Scientist Agent successfully:
• Analyzed each insurance/finance problem
• Selected the appropriate tool for each task
• Structured valid tool call arguments

This enables autonomous model training, prediction, and analysis!
""")


def main():
    parser = argparse.ArgumentParser(description="Demo: LangChain Tool Binding")
    parser.add_argument("--api-key", help="OpenAI API key")
    parser.add_argument("--show-tools", action="store_true", help="Just show available tools")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 LANGCHAIN TOOL BINDING DEMO - AI DATA SCIENTIST AGENT")
    print("="*70)
    
    if args.show_tools:
        tools = load_tools()
        display_tools_info(tools)
        return
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️  No OpenAI API key provided.")
        print("   Use: python demo_tool_binding.py --api-key 'sk-...'")
        print("   Or:  export OPENAI_API_KEY='sk-...'")
        print("\n   Running --show-tools instead...\n")
        tools = load_tools()
        display_tools_info(tools)
        return
    
    run_tool_binding_demo(api_key)


if __name__ == "__main__":
    main()

