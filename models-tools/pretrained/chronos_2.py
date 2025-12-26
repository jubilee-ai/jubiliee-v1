import pandas as pd
from chronos import BaseChronosPipeline
from langchain.tools import tool
from pydantic import BaseModel, Field


class Chronos2Input(BaseModel):
    context_data: list[dict]
    prediction_length: int = 12
    quantile_levels: list[float] = [0.1, 0.5, 0.9]
    id_column: str = "item_id"
    timestamp_column: str = "timestamp"
    target: str = "target"
    future_data: list[dict] | None = None


class Chronos2Output(BaseModel):
    predictions: list[dict]


# Lazy load pipeline
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2", device_map="cuda"
            )
        except Exception:
            _pipeline = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2", device_map="cpu"
            )
    return _pipeline


def forecast(input_data: Chronos2Input) -> Chronos2Output:
    """Generate time series forecasts using Chronos-2."""
    context_df = pd.DataFrame(input_data.context_data)

    future_df = None
    if input_data.future_data:
        future_df = pd.DataFrame(input_data.future_data)

    pred_df = _get_pipeline().predict_df(
        context_df,
        future_df=future_df,
        prediction_length=input_data.prediction_length,
        quantile_levels=input_data.quantile_levels,
        id_column=input_data.id_column,
        timestamp_column=input_data.timestamp_column,
        target=input_data.target,
    )

    return Chronos2Output(predictions=pred_df.to_dict(orient="records"))


# =============================================================================
# LangChain Tool Implementation
# =============================================================================


class Chronos2ToolInput(BaseModel):
    """Input schema for the Chronos-2 forecasting tool."""

    values: list[float] = Field(
        description="Historical time series values in chronological order (oldest first)."
    )
    prediction_length: int = Field(
        default=12,
        description="Number of future steps to forecast."
    )


@tool("chronos2_forecast", args_schema=Chronos2ToolInput)
def chronos2_forecast_tool(
    values: list[float],
    prediction_length: int = 12,
) -> str:
    """Forecast future values from historical time series data using Chronos-2.

    WHEN TO USE:
    - Predicting future stock prices, sales, demand, or any numeric sequence
    - Forecasting trends from historical data
    - Time series with daily, weekly, monthly, or any regular frequency

    WHEN NOT TO USE:
    - Text analysis or sentiment (use FinBERT tools)
    - Data without temporal ordering
    - Single data points (need at least 10+ historical values)

    INPUT FORMAT:
    - values: List of numbers in time order, e.g. [100, 105, 110, 108, 115]
    - prediction_length: How many future steps to predict

    OUTPUT:
    - Forecasts at 10th, 50th (median), and 90th percentiles
    - Use median (p50) as point forecast, p10/p90 as confidence bounds

    Args:
        values: Historical values in chronological order (oldest to newest)
        prediction_length: Number of future time steps to forecast
    """
    context_data = [{"item_id": "series_1", "timestamp": i, "target": v} for i, v in enumerate(values)]

    result = forecast(Chronos2Input(
        context_data=context_data,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
    ))

    # Format output for agent readability
    lines = [f"Forecast for next {prediction_length} steps:"]
    lines.append("Step | Low (10%) | Median (50%) | High (90%)")
    lines.append("-" * 45)

    for i, pred in enumerate(result.predictions, 1):
        low = pred.get("target_0.1", pred.get("0.1", "N/A"))
        mid = pred.get("target_0.5", pred.get("0.5", "N/A"))
        high = pred.get("target_0.9", pred.get("0.9", "N/A"))

        if isinstance(low, float):
            lines.append(f"{i:4} | {low:9.2f} | {mid:12.2f} | {high:10.2f}")
        else:
            lines.append(f"{i:4} | {low} | {mid} | {high}")

    return "\n".join(lines)
