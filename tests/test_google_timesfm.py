import pytest
from google_timesfm import (TimesFMInput, TimesFMOutput, forecast,
                            timesfm_forecast_tool)


def test_insurance_premium_forecast():
    """Test forecasting quarterly insurance premium revenue."""
    # Quarterly premium revenue over 3 years (in thousands)
    premium_revenue = [
        2400, 2520, 2610, 2580,  # Year 1
        2650, 2780, 2850, 2820,  # Year 2
        2900, 3050, 3120, 3080,  # Year 3
    ]

    result = forecast(TimesFMInput(
        inputs=[premium_revenue],
        horizon=4,
    ))

    print(f"Input: Quarterly premiums {premium_revenue}")
    print(f"Point forecast: {result.point_forecast}")
    print(f"Quantile forecast shape: {len(result.quantile_forecast[0])} steps x {len(result.quantile_forecast[0][0])} quantiles")

    assert isinstance(result, TimesFMOutput)
    assert len(result.point_forecast) == 1
    assert len(result.point_forecast[0]) == 4
    assert len(result.quantile_forecast[0]) == 4


def test_claims_loss_ratio_forecast():
    """Test forecasting monthly claims loss ratio for underwriting."""
    # Monthly loss ratios (claims paid / premiums earned) as percentages
    loss_ratios = [
        62.5, 65.2, 61.8, 68.4, 64.1, 63.7,
        66.9, 70.2, 65.5, 62.8, 64.3, 67.1,
        63.2, 66.8, 64.5, 69.1, 65.8, 64.2,
    ]

    result = forecast(TimesFMInput(
        inputs=[loss_ratios],
        horizon=6,
    ))

    print(f"Input: Monthly loss ratios {loss_ratios}")
    print(f"Point forecast: {result.point_forecast[0]}")

    assert isinstance(result, TimesFMOutput)
    assert len(result.point_forecast[0]) == 6
    # Verify quantiles have expected structure (10 quantiles per step)
    assert len(result.quantile_forecast[0][0]) == 10


def test_langchain_tool_stock_portfolio():
    """Test LangChain tool with stock portfolio valuation forecast."""
    # Weekly portfolio values (in thousands)
    portfolio_values = [
        520.5, 525.2, 518.7, 530.4, 542.1,
        538.6, 545.3, 552.8, 548.2, 560.5,
        555.1, 568.4, 572.9, 565.2, 580.1,
    ]

    result = timesfm_forecast_tool.invoke({
        "values": portfolio_values,
        "horizon": 4,
    })

    print(f"Input: Weekly portfolio values {portfolio_values}")
    print(f"Output:\n{result}")

    assert isinstance(result, str)
    assert "TimesFM Forecast (4 steps ahead)" in result
    assert "Point" in result
    assert "Low(10%)" in result
    assert "High(90%)" in result
    assert "Interpretation" in result

