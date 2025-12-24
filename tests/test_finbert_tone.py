import pytest
from finbert_tone import (FinBERTToneInput, FinBERTToneOutput, analyze_tone,
                          finbert_tone_tool)


def test_positive_tone():
    """Test that positive financial text is classified correctly."""
    result = analyze_tone(FinBERTToneInput(text="growth is strong and we have plenty of liquidity"))
    print(f"Input: 'growth is strong and we have plenty of liquidity'")
    print(f"Output: {result}")
    assert isinstance(result, FinBERTToneOutput)
    assert result.sentiment == "positive"
    assert result.confidence > 0.5
    assert abs(sum(result.scores.values()) - 1.0) < 0.01  # scores sum to 1


def test_negative_tone():
    """Test that negative financial text is classified correctly."""
    result = analyze_tone(FinBERTToneInput(text="there is a shortage of capital, and we need extra financing"))
    print(f"Input: 'there is a shortage of capital, and we need extra financing'")
    print(f"Output: {result}")
    assert isinstance(result, FinBERTToneOutput)
    assert result.sentiment == "negative"
    assert result.confidence > 0.5
    assert "positive" in result.scores and "negative" in result.scores and "neutral" in result.scores


def test_langchain_tool_returns_formatted_string():
    """Test that the LangChain tool returns properly formatted output."""
    result = finbert_tone_tool.invoke({"text": "profits are flat"})
    print(f"Input: 'profits are flat'")
    print(f"Output:\n{result}")
    assert isinstance(result, str)
    assert "Sentiment:" in result
    assert "Confidence:" in result
    assert "Scores:" in result

