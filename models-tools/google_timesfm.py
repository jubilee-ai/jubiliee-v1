import numpy as np
import timesfm
from langchain.tools import tool
from pydantic import BaseModel, Field


class TimesFMInput(BaseModel):
    inputs: list[list[float]] = Field(
        description="List of time series, each a list of floats in chronological order."
    )
    horizon: int = Field(default=12, description="Number of future steps to forecast.")
    max_context: int = Field(default=1024, description="Maximum context length.")
    normalize_inputs: bool = Field(default=True, description="Whether to normalize inputs.")


class TimesFMOutput(BaseModel):
    point_forecast: list[list[float]] = Field(
        description="Point forecasts, shape (num_series, horizon)."
    )
    quantile_forecast: list[list[list[float]]] = Field(
        description="Quantile forecasts, shape (num_series, horizon, 10)."
    )


_model = None


def _get_model():
    global _model
    if _model is None:
        _model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch", torch_compile=True
        )
    return _model


def forecast(input_data: TimesFMInput) -> TimesFMOutput:
    """Generate time series forecasts using Google TimesFM."""
    model = _get_model()

    model.compile(
        timesfm.ForecastConfig(
            max_context=input_data.max_context,
            max_horizon=input_data.horizon,
            normalize_inputs=input_data.normalize_inputs,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    inputs = [np.array(series) for series in input_data.inputs]

    point_forecast, quantile_forecast = model.forecast(
        horizon=input_data.horizon,
        inputs=inputs,
    )

    return TimesFMOutput(
        point_forecast=point_forecast.tolist(),
        quantile_forecast=quantile_forecast.tolist(),
    )


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class TimesFMToolInput(BaseModel):
    """Input schema for the TimesFM forecasting tool."""

    values: list[float] = Field(
        description="Historical time series values in chronological order (oldest first). Minimum 10 values recommended."
    )
    horizon: int = Field(
        default=12,
        description="Number of future time steps to forecast (1-256)."
    )


@tool("timesfm_forecast", args_schema=TimesFMToolInput)
def timesfm_forecast_tool(
    values: list[float],
    horizon: int = 12,
) -> str:
    """Forecast future values from historical time series using Google TimesFM foundation model.

    WHEN TO USE:
    - Forecasting stock prices, revenue, sales, demand, or any numeric time series
    - Predicting trends from historical sequential data
    - Generating probabilistic forecasts with uncertainty quantiles
    - Any time series with regular frequency (daily, weekly, monthly, etc.)

    WHEN NOT TO USE:
    - Text/sentiment analysis (use FinBERT tools instead)
    - Classification tasks (this is regression only)
    - Data without temporal ordering
    - Fewer than 10 historical values (need sufficient context)

    INPUT FORMAT:
    - values: List of floats in chronological order, e.g., [100.5, 102.3, 98.7, 105.2]
    - horizon: Integer for forecast steps (default 12, max 256)

    OUTPUT FORMAT:
    - Point forecast (median prediction) for each future step
    - Quantile forecasts: 10th-90th percentiles for uncertainty bounds
    - Use p50 as point estimate, p10/p90 as 80% prediction interval

    EXAMPLE:
    Input: values=[120, 125, 130, 128, 135, 140], horizon=3
    Output: 3-step forecast with confidence intervals

    Args:
        values: Historical time series values (oldest to newest)
        horizon: Number of future steps to predict
    """
    result = forecast(TimesFMInput(
        inputs=[values],
        horizon=horizon,
    ))

    point = result.point_forecast[0]
    quantiles = result.quantile_forecast[0]

    lines = [f"TimesFM Forecast ({horizon} steps ahead):"]
    lines.append("")
    lines.append("Step |   Point   |  Low(10%)  |  High(90%)")
    lines.append("-" * 48)

    for i in range(horizon):
        pt = point[i]
        q = quantiles[i]
        low = q[1] if len(q) > 1 else q[0]   # 10th percentile
        high = q[-2] if len(q) > 2 else q[-1]  # 90th percentile
        lines.append(f"{i+1:4} | {pt:9.2f} | {low:10.2f} | {high:10.2f}")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- Point: Best estimate (median forecast)")
    lines.append("- Low/High: 80% prediction interval bounds")

    return "\n".join(lines)

