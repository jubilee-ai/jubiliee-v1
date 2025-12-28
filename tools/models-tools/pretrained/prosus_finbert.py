import torch
from langchain.tools import tool
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinBERTInput(BaseModel):
    text: str


class FinBERTOutput(BaseModel):
    sentiment: str  # "positive", "negative", or "neutral"
    confidence: float
    scores: dict[str, float]  # all class scores


# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# FinBERT label mapping
LABELS = ["positive", "negative", "neutral"]


def analyze_sentiment(input_data: FinBERTInput) -> FinBERTOutput:
    """Analyze financial sentiment of text using FinBERT."""
    inputs = tokenizer(
        input_data.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

    scores = {label: prob.item() for label, prob in zip(LABELS, probabilities)}
    predicted_idx = probabilities.argmax().item()

    return FinBERTOutput(
        sentiment=LABELS[predicted_idx],
        confidence=probabilities[predicted_idx].item(),
        scores=scores
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class FinBERTToolInput(BaseModel):
    """Input schema for the FinBERT sentiment analysis tool."""

    text: str = Field(
        description=(
            "The financial text to analyze. Works best with 1-3 sentences. "
            "Examples: news headlines, earnings summaries, analyst quotes, SEC filing excerpts."
        )
    )


@tool("finbert_sentiment", args_schema=FinBERTToolInput)
def finbert_sentiment_tool(text: str) -> str:
    """Analyze the sentiment of financial text using FinBERT, a model trained on financial communications.

    WHEN TO USE:
    - Determining market sentiment from news headlines or articles
    - Analyzing tone of earnings calls, press releases, or SEC filings
    - Gauging investor sentiment from analyst reports or social media posts
    - Evaluating whether financial commentary is bullish, bearish, or neutral

    WHEN NOT TO USE:
    - Non-financial text (use a general sentiment model instead)
    - Numerical analysis or calculations
    - Factual lookups about companies or stocks

    OUTPUT:
    - sentiment: "positive" (bullish), "negative" (bearish), or "neutral"
    - confidence: How certain the model is (0-100%)
    - scores: Probability distribution across all three sentiment classes

    EXAMPLES OF GOOD INPUT:
    - "Apple beats Q3 earnings expectations, stock surges 5%"
    - "Fed signals potential rate cuts amid cooling inflation"
    - "Company announces 10,000 layoffs following disappointing guidance"

    Args:
        text: Financial text to analyze (1-3 sentences works best, max 512 tokens)
    """
    result = analyze_sentiment(FinBERTInput(text=text))

    return (
        f"Sentiment: {result.sentiment}\n"
        f"Confidence: {result.confidence:.2%}\n"
        f"Scores: positive={result.scores['positive']:.2%}, "
        f"negative={result.scores['negative']:.2%}, "
        f"neutral={result.scores['neutral']:.2%}"
    )