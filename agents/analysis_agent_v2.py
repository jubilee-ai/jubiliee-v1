"""
Analysis Agent v2 - Data analyst agent with access to statistical analysis, 
data retrieval, and pretrained model execution.

This agent orchestrates three capabilities:
1. Statistical analysis agent - for data profiling, validation, and feature analysis
2. Data retrieval agent - for discovering, loading, and preparing datasets
3. Model execution tool - for selecting and running appropriate pretrained models
"""

import json
import sys
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Load environment variables
_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")

# Add paths for imports
_PRETRAINED_DIR = _ROOT / "tools" / "models-tools" / "pretrained"
_DATA_RETRIEVAL_DIR = Path(__file__).parent / "data-retrieval"

if str(_PRETRAINED_DIR) not in sys.path:
    sys.path.insert(0, str(_PRETRAINED_DIR))
if str(_DATA_RETRIEVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_RETRIEVAL_DIR))

# Import data retrieval (for get_dataset and retrieve_data)
from agent import (DataRetrievalResult, get_dataset,  # type: ignore
                   retrieve_data)
from bert_finetuned_claim_detection import claim_detection_tool  # type: ignore
from chronos_2 import chronos2_forecast_tool  # type: ignore
# Import pretrained model tools
from credit_risk import credit_card_risk_prediction_tool  # type: ignore
from finbert_tone import finbert_tone_tool  # type: ignore
from google_timesfm import timesfm_forecast_tool  # type: ignore
from loan_default_prediction import \
    loan_default_prediction_tool  # type: ignore
from prosus_finbert import finbert_sentiment_tool  # type: ignore

from agents.model_index import (format_models_for_prompt, get_model_by_name,
                                get_model_index)
from agents.prompts import ANALYSIS_AGENT_SYSTEM_PROMPT, MODEL_SELECTION_PROMPT
# Import from agents package
from agents.statistical_analysis_agent import statistical_analysis_tool

# =============================================================================
# LLM Setup
# =============================================================================

llm = ChatOpenAI(model="gpt-5.1", temperature=0)

# =============================================================================
# Model Registry - Maps model names to their tool functions
# =============================================================================

MODEL_TOOLS = {
    "credit_card_risk_prediction": credit_card_risk_prediction_tool,
    "loan_default_prediction": loan_default_prediction_tool,
    "finbert_tone": finbert_tone_tool,
    "finbert_sentiment": finbert_sentiment_tool,
    "claim_detection": claim_detection_tool,
    "chronos2_forecast": chronos2_forecast_tool,
    "timesfm_forecast": timesfm_forecast_tool,
}

# =============================================================================
# Model Selection & Execution Tool
# =============================================================================


class ModelSelectionResponse(BaseModel):
    """Structured response for model selection."""
    decision: Literal["select", "skip"] = Field(
        description="Whether to select a model or skip"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Name of the selected model (if decision is 'select')"
    )
    parameters: Optional[str] = Field(
        default=None,
        description="JSON string of parameters to pass to the model, mapped from user data"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason for skipping (if decision is 'skip')"
    )


class ModelExecutionInput(BaseModel):
    """Input schema for model execution tool."""
    goal: str = Field(
        description="The goal or question to answer using a pretrained model. "
        "For text analysis, include the text in the goal. "
        "Examples: 'Analyze sentiment of: [text here]', 'Predict credit risk for this applicant'"
    )
    data: Optional[dict] = Field(
        default=None,
        description="Optional structured input data for the model. Required for tabular models like credit risk. "
        "Examples: {'age': 35, 'income': 50000, 'gender': 'm', ...} for credit risk. "
        "For text analysis, you can leave this empty and include text in the goal."
    )


@tool(args_schema=ModelExecutionInput)
def model_execution_tool(goal: str, data: Optional[dict] = None) -> str:
    """Select and execute the most appropriate pretrained model for a given goal and data.
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Run predictions or classifications on specific input data
    - Score or assess risk for individual records
    - Analyze text sentiment or detect claims
    - Generate forecasts from time series data
    
    DO NOT USE THIS TOOL IF:
    - You need to analyze a dataset
    - You need to retrieve/load data
    - The data doesn't match any model's expected input format
    
    The tool automatically selects the right model based on your goal and maps your data
    to the model's expected schema. For text analysis, include the text in the goal.
    """
    # Get all models and format for prompt
    all_models = get_model_index()
    models_text = format_models_for_prompt(all_models)
    
    # Format data for display (handle None case)
    if data is None:
        data = {}
    data_str = json.dumps(data, indent=2) if data else "No structured data provided - extract from goal if needed"
    
    # Ask LLM to select model using structured output
    prompt = MODEL_SELECTION_PROMPT.format(
        models=models_text,
        goal=goal,
        data=data_str
    )
    
    structured_llm = llm.with_structured_output(ModelSelectionResponse, method="function_calling")
    decision = structured_llm.invoke(prompt)
    
    if decision.decision == "skip":
        return f"No suitable model found. Reason: {decision.reason or 'Unknown'}"
    
    model_name = decision.model_name
    # Parse parameters from JSON string
    try:
        parameters = json.loads(decision.parameters) if decision.parameters else {}
    except json.JSONDecodeError:
        parameters = {}
    
    # Validate model exists
    model_info = get_model_by_name(model_name)
    if not model_info:
        return f"Error: Model '{model_name}' not found in registry"
    
    # Get the tool function
    tool_func = MODEL_TOOLS.get(model_name)
    if not tool_func:
        return f"Error: Model '{model_name}' has no registered tool function"
    
    # Execute the model
    try:
        result = tool_func.invoke(parameters)
        return f"Model: {model_name}\n\nResult:\n{result}"
    except Exception as e:
        return f"Error executing model '{model_name}': {type(e).__name__}: {e}"

# =============================================================================
# Data Retrieval Tool - For data lookup, display, and model input preparation
# =============================================================================

class DataLookupInput(BaseModel):
    """Input schema for data lookup tool."""
    request: str = Field(
        description="What data to find or retrieve. Be specific about what you need."
    )


@tool(args_schema=DataLookupInput)
def data_lookup_tool(request: str) -> str:
    """Find and retrieve data from the catalog.
    
    USE THIS TOOL ONLY FOR:
    - Showing the user what datasets are available
    - Displaying sample data or data structure to the user
    - Finding specific records to use as input for model_execution_tool
    - Looking up data the user explicitly asked to see
    
    DO NOT USE THIS TOOL FOR:
    - Statistical analysis (use statistical_analysis_tool instead)
    - Understanding patterns or trends (use statistical_analysis_tool instead)
    - Any analysis beyond simple data retrieval
    
    This is a lightweight lookup tool, not an analysis tool.
    """
    try:
        result = retrieve_data(request)
        
        if isinstance(result, DataRetrievalResult):
            return (
                f"Data Found:\n"
                f"  Dataset: {result.dataset_ref}\n"
                f"  Description: {result.description}\n"
                f"  Rows: {result.rows}\n"
                f"  Columns: {', '.join(result.columns)}\n"
                f"  Source: {result.source}"
            )
        elif isinstance(result, dict):
            if result.get("success") is False:
                return f"Data lookup failed: {result.get('error', 'Unknown error')}"
            return f"Data lookup result: {json.dumps(result, indent=2)}"
        else:
            return str(result)
    except Exception as e:
        return f"Data lookup failed: {type(e).__name__}: {e}"
# =============================================================================
# Agent Tools
# =============================================================================

ANALYSIS_TOOLS = [
    statistical_analysis_tool,  # Find data + run statistical analysis
    model_execution_tool,       # Run pretrained models on specific inputs
    data_lookup_tool,           # Simple data lookup, display, and model input prep
]


# =============================================================================
# Agent Construction
# =============================================================================

agent = create_agent(
        model=llm,
        tools=ANALYSIS_TOOLS,
        system_prompt=ANALYSIS_AGENT_SYSTEM_PROMPT,
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Agent
    "agent",
    # Tools
    "model_execution_tool",
    "data_lookup_tool",
    "ANALYSIS_TOOLS",
    # Model registry
    "MODEL_TOOLS",
]
