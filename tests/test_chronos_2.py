import pytest
from chronos_2 import (Chronos2Input, Chronos2Output, forecast,
                       chronos2_forecast_tool)


def test_stock_price_forecast():
    """Test forecasting stock prices with upward trend."""
    # Simulated quarterly stock prices (upward trend)
    stock_prices = [145.0, 148.5, 152.3, 149.8, 155.2, 158.7, 162.1, 165.4, 168.9, 172.5, 175.8, 180.2]
    
    result = forecast(Chronos2Input(
        context_data=[{"item_id": "AAPL", "timestamp": i, "target": v} for i, v in enumerate(stock_prices)],
        prediction_length=4,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    ))
    
    print(f"Input: Stock prices {stock_prices}")
    print(f"Output: {result.predictions}")
    
    assert isinstance(result, Chronos2Output)
    assert len(result.predictions) == 4
    assert all(isinstance(p, dict) for p in result.predictions)


def test_insurance_claims_forecast():
    """Test forecasting monthly insurance claim volumes."""
    # Simulated monthly insurance claims with seasonal pattern
    monthly_claims = [1200, 1150, 1300, 1450, 1380, 1420, 1550, 1480, 1350, 1280, 1320, 1400,
                      1250, 1180, 1340, 1500, 1420, 1460]
    
    result = forecast(Chronos2Input(
        context_data=[{"item_id": "claims", "timestamp": i, "target": v} for i, v in enumerate(monthly_claims)],
        prediction_length=6,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    ))
    
    print(f"Input: Monthly claims {monthly_claims}")
    print(f"Output: {result.predictions}")
    
    assert isinstance(result, Chronos2Output)
    assert len(result.predictions) == 6
    # Verify predictions have quantile keys
    for pred in result.predictions:
        assert len(pred) > 0


def test_langchain_tool_returns_formatted_string():
    """Test that the LangChain tool returns properly formatted forecast output."""
    # Simulated daily premium revenue
    premium_revenue = [52000, 53500, 51800, 54200, 55100, 53900, 56000, 57200, 55800, 58000]
    
    result = chronos2_forecast_tool.invoke({
        "values": premium_revenue,
        "prediction_length": 3
    })
    
    print(f"Input: Premium revenue {premium_revenue}")
    print(f"Output:\n{result}")
    
    assert isinstance(result, str)
    assert "Forecast for next 3 steps" in result
    assert "Low (10%)" in result
    assert "Median (50%)" in result
    assert "High (90%)" in result

