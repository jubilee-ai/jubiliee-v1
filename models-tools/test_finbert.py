import pytest
from finbert import (FinBERTInput, FinBERTOutput, analyze_sentiment,
                     finbert_sentiment_tool)


def test_positive_sentiment():
    """Test that bullish financial text is classified as positive."""
    result = analyze_sentiment(FinBERTInput(text="Company reports record profits and raises dividend by 20%"))
    print(result)
    assert isinstance(result, FinBERTOutput)
    assert result.sentiment == "positive"
    assert result.confidence > 0.5
    assert abs(sum(result.scores.values()) - 1.0) < 0.01  # scores sum to 1


def test_negative_sentiment():
    """Test that bearish financial text is classified as negative."""
    result = analyze_sentiment(FinBERTInput(text="Stock crashes after fraud allegations surface"))
    print(result)
    assert isinstance(result, FinBERTOutput)
    assert result.sentiment == "negative"
    assert result.confidence > 0.5
    assert "positive" in result.scores and "negative" in result.scores and "neutral" in result.scores


def test_langchain_tool_returns_formatted_string():
    """Test that the LangChain tool returns properly formatted output."""
    result = finbert_sentiment_tool.invoke({"text": "Fed holds interest rates steady as expected"})
    print(result)
    assert isinstance(result, str)
    assert "Sentiment:" in result
    assert "Confidence:" in result
    assert "Scores:" in result

