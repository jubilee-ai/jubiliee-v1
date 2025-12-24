import torch
from langchain.tools import tool
from pydantic import BaseModel, Field
from transformers import BertForSequenceClassification, BertTokenizer


class FinBERTToneInput(BaseModel):
    text: str


class FinBERTToneOutput(BaseModel):
    sentiment: str  # "positive", "negative", or "neutral"
    confidence: float
    scores: dict[str, float]  # all class scores


# Load model and tokenizer once
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone", num_labels=3)

# FinBERT-tone label mapping: LABEL_0=neutral, LABEL_1=positive, LABEL_2=negative
LABELS = ["neutral", "positive", "negative"]


def analyze_tone(input_data: FinBERTToneInput) -> FinBERTToneOutput:
    """Analyze financial tone of text using FinBERT-tone."""
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

    return FinBERTToneOutput(
        sentiment=LABELS[predicted_idx],
        confidence=probabilities[predicted_idx].item(),
        scores=scores
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class FinBERTToneToolInput(BaseModel):
    """Input schema for the FinBERT-tone sentiment analysis tool."""

    text: str = Field(
        description=(
            "The financial text to analyze for tone. Works best with 1-3 sentences. "
            "Examples: analyst reports, earnings call excerpts, SEC filings."
        )
    )


@tool("finbert_tone", args_schema=FinBERTToneToolInput)
def finbert_tone_tool(text: str) -> str:
    """Analyze the tone of financial text using FinBERT-tone, fine-tuned on analyst reports.

    WHEN TO USE:
    - Analyzing tone of analyst reports and recommendations
    - Evaluating sentiment in earnings calls and transcripts
    - Assessing tone in SEC filings (10-K, 10-Q)
    - Financial news and press release analysis

    WHEN NOT TO USE:
    - Non-financial text (use a general sentiment model instead)
    - Numerical analysis or calculations
    - Factual lookups about companies or stocks

    OUTPUT:
    - sentiment: "positive", "negative", or "neutral"
    - confidence: How certain the model is (0-100%)
    - scores: Probability distribution across all three sentiment classes

    EXAMPLES OF GOOD INPUT:
    - "there is a shortage of capital, and we need extra financing"
    - "growth is strong and we have plenty of liquidity"
    - "profits are flat"

    Args:
        text: Financial text to analyze (1-3 sentences works best, max 512 tokens)
    """
    result = analyze_tone(FinBERTToneInput(text=text))

    return (
        f"Sentiment: {result.sentiment}\n"
        f"Confidence: {result.confidence:.2%}\n"
        f"Scores: positive={result.scores['positive']:.2%}, "
        f"negative={result.scores['negative']:.2%}, "
        f"neutral={result.scores['neutral']:.2%}"
    )

