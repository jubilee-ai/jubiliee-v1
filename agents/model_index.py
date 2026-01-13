"""
Model Index - Metadata for all pretrained models available for the analysis agent.

Each model entry contains:
- name: Tool name for invocation
- description: What the model does
- when_to_use: Scenarios where this model is appropriate
- when_not_to_use: Scenarios to avoid
- schema: Required input fields with types and descriptions
- output: What the model returns
"""

MODEL_INDEX = [
    {
        "name": "credit_card_risk_prediction",
        "description": "Classify credit card applicant into risk category using logistic regression.",
        "when_to_use": [
            "Credit card application screening",
            "Risk stratification for applicants",
            "Determining approve/reject recommendation",
        ],
        "when_not_to_use": [
            "Non-credit-card lending decisions",
            "When applicant demographics differ significantly from training data",
        ],
        "schema": {
            "gender": {"type": "Literal['m', 'f']", "description": "Biological sex: 'm' or 'f'"},
            "marital": {"type": "Literal['married', 'single', 'divsepwid']", "description": "Marital status"},
            "howpaid": {"type": "Literal['monthly', 'weekly']", "description": "Salary payment frequency"},
            "mortgage": {"type": "Literal['y', 'n']", "description": "Has active mortgage"},
            "age": {"type": "int", "description": "Age in years (18+)"},
            "income": {"type": "float", "description": "Annual income (>0)"},
            "numkids": {"type": "int", "description": "Number of dependent children (>=0)"},
            "numcards": {"type": "int", "description": "Existing credit cards held (>=0)"},
            "storecar": {"type": "int", "description": "Retail/store cards held (>=0)"},
            "loans": {"type": "int", "description": "Active loans count (>=0)"},
        },
        "output": "Risk classification: 'Good risk', 'Bad loss', or 'Bad profit' with probabilities",
    },
    {
        "name": "loan_default_prediction",
        "description": "Predict loan default probability using Random Forest trained on Indian loan data.",
        "when_to_use": [
            "Loan application screening",
            "Credit risk triage",
            "Default probability scoring from demographics",
        ],
        "when_not_to_use": [
            "Non-Indian applicant data (model trained on Indian geography)",
            "When network access is unavailable (requires external API)",
        ],
        "schema": {
            "income": {"type": "int", "description": "Annual income in local currency"},
            "age": {"type": "int", "description": "Applicant age in years"},
            "experience": {"type": "int", "description": "Total professional experience in years"},
            "married": {"type": "Literal['single', 'married']", "description": "Marital status"},
            "house_ownership": {"type": "Literal['rented', 'owned', 'norent_noown']", "description": "Housing status"},
            "car_ownership": {"type": "Literal['yes', 'no']", "description": "Owns a car"},
            "profession": {"type": "str", "description": "Job title"},
            "city": {"type": "str", "description": "City of residence"},
            "state": {"type": "str", "description": "State of residence"},
            "current_job_yrs": {"type": "int", "description": "Years at current employer"},
            "current_house_yrs": {"type": "int", "description": "Years at current address"},
        },
        "output": "Prediction: 'Default' or 'No Default' with probability and risk level",
    },
    {
        "name": "finbert_tone",
        "description": "Analyze tone of financial text using FinBERT fine-tuned on analyst reports.",
        "when_to_use": [
            "Analyzing tone of analyst reports",
            "Evaluating sentiment in earnings calls",
            "Assessing tone in SEC filings (10-K, 10-Q)",
            "Financial news and press release analysis",
        ],
        "when_not_to_use": [
            "Non-financial text",
            "Numerical analysis or calculations",
        ],
        "schema": {
            "text": {"type": "str", "description": "Financial text to analyze (1-3 sentences, max 512 tokens)"},
        },
        "output": "Sentiment: 'positive', 'negative', or 'neutral' with confidence scores",
    },
    {
        "name": "finbert_sentiment",
        "description": "Analyze sentiment of financial text using ProsusAI FinBERT.",
        "when_to_use": [
            "Market sentiment from news headlines",
            "Tone of earnings calls and press releases",
            "Investor sentiment from analyst reports",
            "Determining if commentary is bullish/bearish/neutral",
        ],
        "when_not_to_use": [
            "Non-financial text",
            "Numerical analysis",
        ],
        "schema": {
            "text": {"type": "str", "description": "Financial text to analyze (1-3 sentences, max 512 tokens)"},
        },
        "output": "Sentiment: 'positive', 'negative', or 'neutral' with confidence scores",
    },
    {
        "name": "claim_detection",
        "description": "Identify check-worthy statements in insurance claims for screening and fraud detection.",
        "when_to_use": [
            "Insurance claim screening and triage",
            "Fraud detection in claim descriptions",
            "Flagging statements that require verification",
        ],
        "when_not_to_use": [
            "Determining if a claim is TRUE or FALSE (only detects check-worthiness)",
            "Sentiment analysis",
            "Non-English text",
        ],
        "schema": {
            "text": {"type": "str", "description": "Insurance claim text to analyze (1-3 sentences, max 512 tokens)"},
        },
        "output": "Classification: 'check-worthy' or 'not-check-worthy' with confidence",
    },
    {
        "name": "chronos2_forecast",
        "description": "Forecast future values from historical time series using Amazon Chronos-2.",
        "when_to_use": [
            "Predicting future stock prices, sales, demand",
            "Forecasting trends from historical data",
            "Time series with regular frequency (daily, weekly, monthly)",
        ],
        "when_not_to_use": [
            "Text/sentiment analysis",
            "Data without temporal ordering",
            "Fewer than 10 historical values",
        ],
        "schema": {
            "values": {"type": "list[float]", "description": "Historical values in chronological order (oldest first)"},
            "prediction_length": {"type": "int", "description": "Number of future steps to forecast (default 12)"},
        },
        "output": "Forecasts at 10th, 50th (median), and 90th percentiles for each future step",
    },
    {
        "name": "timesfm_forecast",
        "description": "Forecast future values using Google TimesFM foundation model.",
        "when_to_use": [
            "Forecasting stock prices, revenue, sales, demand",
            "Predicting trends from sequential data",
            "Generating probabilistic forecasts with uncertainty",
        ],
        "when_not_to_use": [
            "Text/sentiment analysis",
            "Classification tasks",
            "Fewer than 10 historical values",
        ],
        "schema": {
            "values": {"type": "list[float]", "description": "Historical values in chronological order (oldest first)"},
            "horizon": {"type": "int", "description": "Number of future steps to forecast (default 12, max 256)"},
        },
        "output": "Point forecast and quantile forecasts (10th-90th percentiles)",
    },
]


def get_model_index() -> list[dict]:
    """Return the full model index."""
    return MODEL_INDEX


def get_model_by_name(name: str) -> dict | None:
    """Get a specific model by name."""
    for model in MODEL_INDEX:
        if model["name"] == name:
            return model
    return None


def format_model_for_prompt(model: dict) -> str:
    """Format a model entry for inclusion in an LLM prompt."""
    schema_lines = []
    for field, info in model["schema"].items():
        schema_lines.append(f"    - {field} ({info['type']}): {info['description']}")
    
    return f"""**{model['name']}**
Description: {model['description']}
When to use: {', '.join(model['when_to_use'])}
When NOT to use: {', '.join(model['when_not_to_use'])}
Required inputs:
{chr(10).join(schema_lines)}
Output: {model['output']}"""


def format_models_for_prompt(models: list[dict]) -> str:
    """Format multiple models for an LLM prompt."""
    return "\n\n---\n\n".join(format_model_for_prompt(m) for m in models)

