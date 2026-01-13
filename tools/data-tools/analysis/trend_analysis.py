"""
trend_analysis - Analyze time-based trends in data.

Answers: "How did sales change month-over-month?" "What's the growth rate?" "Is there seasonality?"
Provides period-over-period changes, growth rates, moving averages, and trend detection.
"""

import sys
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset


def _r(value: float, decimals: int = 2) -> Optional[float]:
    """Round to native float, handle NaN."""
    return None if pd.isna(value) else round(float(value), decimals)


def _fmt(value: Optional[float], show_sign: bool = False) -> str:
    """Format number adaptively based on magnitude."""
    if value is None:
        return "—"
    sign = "+" if show_sign and value > 0 else ""
    abs_val = abs(value)
    if abs_val >= 1000:
        return f"{sign}{value:,.0f}"
    elif abs_val >= 10:
        return f"{sign}{value:,.1f}"
    elif abs_val >= 1:
        return f"{sign}{value:,.2f}"
    else:
        return f"{sign}{value:,.3f}"  # Small values get 3 decimals


def _detect_trend(series: pd.Series) -> dict:
    """Detect trend direction using linear regression."""
    if len(series) < 3:
        return {"direction": "insufficient_data", "slope_per_period": None, "total_change_pct": None, "r_squared": None}
    
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series.values, 1)
    
    # R² via correlation coefficient
    r_squared = np.corrcoef(x, series.values)[0, 1] ** 2
    
    # Total percent change
    pct_change = (series.iloc[-1] - series.iloc[0]) / series.iloc[0] if series.iloc[0] != 0 else 0
    
    # Classify direction
    if abs(pct_change) < 0.05:
        direction = "flat"
    elif pct_change > 0.2:
        direction = "increasing"
    elif pct_change > 0:
        direction = "slightly_increasing"
    elif pct_change < -0.2:
        direction = "decreasing"
    else:
        direction = "slightly_decreasing"
    
    return {
        "direction": direction,
        "slope_per_period": _r(slope, 4),
        "total_change_pct": _r(pct_change * 100, 1),
        "r_squared": _r(r_squared, 3),
    }


def _detect_seasonality(series: pd.Series) -> dict:
    """Detect seasonality using autocorrelation at common lags."""
    n = len(series)
    if n < 12:  # Need at least 12 periods for reliable detection
        return {"detected": False, "reason": "insufficient_data"}
    
    period_names = {7: "weekly", 12: "monthly", 4: "quarterly", 52: "yearly"}
    best_period, best_corr = None, 0.5  # Higher threshold to reduce false positives
    
    for p in [7, 12, 4, 52]:
        # Need at least 2 full cycles to detect seasonality
        if p * 2 <= n:
            corr = series.autocorr(lag=p)
            if pd.notna(corr) and corr > best_corr:
                best_period, best_corr = p, corr
    
    if best_period:
        return {"detected": True, "period": best_period, 
                "period_name": period_names.get(best_period, f"every {best_period} periods"),
                "strength": _r(best_corr, 2)}
    return {"detected": False, "reason": "no strong periodic pattern found"}


def analyze_trend(
    dataset_ref: str,
    date_col: str,
    metric_col: str,
    freq: str = "ME",
    agg_func: str = "sum",
    periods_for_ma: int = 3,
) -> dict:
    """Analyze trends in time series data."""
    df = resolve_dataset(dataset_ref)
    
    if date_col not in df.columns:
        return {"error": f"Date column '{date_col}' not found"}
    if metric_col not in df.columns:
        return {"error": f"Metric column '{metric_col}' not found"}
    
    # Parse dates and filter valid rows
    df = df[[date_col, metric_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna()
    
    if len(df) < 3:
        return {"error": "Insufficient data for trend analysis"}
    
    # Resample and aggregate
    ts = df.set_index(date_col)[metric_col].resample(freq).agg(agg_func).dropna()
    
    if len(ts) < 2:
        return {"error": "Not enough periods after aggregation"}
    
    # Compute changes using pandas
    pct_change = ts.pct_change() * 100
    abs_change = ts.diff()
    ma = ts.rolling(window=periods_for_ma, min_periods=1).mean()
    
    # Build periods list using zip
    periods = [
        {
            "period": idx.strftime("%Y-%m-%d"),
            "value": _r(val),
            "pct_change": _r(pct_change.loc[idx]) if i > 0 else None,
            "abs_change": _r(abs_change.loc[idx]) if i > 0 else None,
            "moving_avg": _r(ma.loc[idx]),
        }
        for i, (idx, val) in enumerate(ts.items())
    ]
    
    # Growth calculation
    first, last = ts.iloc[0], ts.iloc[-1]
    total_growth = _r((last / first - 1) * 100) if first != 0 else None
    
    return {
        "date_col": date_col,
        "metric_col": metric_col,
        "frequency": freq,
        "aggregation": agg_func,
        "summary": {
            "total_periods": len(ts),
            "date_range": {"start": ts.index.min().strftime("%Y-%m-%d"), 
                           "end": ts.index.max().strftime("%Y-%m-%d")},
            "metric_summary": {
                "total": _r(ts.sum()), "mean": _r(ts.mean()),
                "min": _r(ts.min()), "max": _r(ts.max()), "std": _r(ts.std()),
            },
            "growth": {
                "first_period": _r(first), "last_period": _r(last),
                "total_growth_pct": total_growth,
                "avg_period_growth_pct": _r(pct_change.mean()),
                "median_period_growth_pct": _r(pct_change.median()),
            },
            "volatility": {
                "pct_change_std": _r(pct_change.std()),
                "coefficient_of_variation": _r(ts.std() / ts.mean() * 100) if ts.mean() != 0 else None,
            },
        },
        "trend": _detect_trend(ts),
        "seasonality": _detect_seasonality(ts),
        "periods": periods,
    }


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class TrendAnalysisInput(BaseModel):
    """Input schema for trend_analysis_tool."""
    dataset_ref: str = Field(description="Dataset reference from previous operation or table name")
    date_col: str = Field(description="Column containing dates/timestamps")
    metric_col: str = Field(description="Numeric column to analyze for trends")
    freq: Literal["D", "W", "ME", "QE", "YE"] = Field(
        default="ME", description="Frequency: D=daily, W=weekly, ME=monthly, QE=quarterly, YE=yearly")
    agg_func: Literal["sum", "mean", "count", "min", "max"] = Field(
        default="sum", description="How to aggregate the metric within each period")
    periods_for_ma: int = Field(default=3, description="Periods for moving average (2-12)")


@tool(args_schema=TrendAnalysisInput)
def trend_analysis_tool(
    dataset_ref: str,
    date_col: str,
    metric_col: str,
    freq: str = "ME",
    agg_func: str = "sum",
    periods_for_ma: int = 3,
) -> str:
    """
    Analyze time-based trends in data.
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Understand how a metric changes over time
    - Calculate period-over-period growth rates
    - Detect upward/downward trends
    - Identify seasonality patterns
    - Get moving averages for smoothed trends
    
    FREQUENCY: D=daily, W=weekly, ME=monthly, QE=quarterly, YE=yearly
    
    RETURNS: Period-by-period values, growth rates, trend direction, seasonality detection.
    """
    result = analyze_trend(dataset_ref, date_col, metric_col, freq, agg_func, periods_for_ma)
    
    if "error" in result:
        return f"✗ {result['error']}"
    
    freq_names = {"D": "Daily", "W": "Weekly", "ME": "Monthly", "QE": "Quarterly", "YE": "Yearly"}
    s, t, sea = result['summary'], result['trend'], result['seasonality']
    g = s['growth']
    
    # Build output
    lines = [
        "# Trend Analysis",
        f"**Metric:** `{result['metric_col']}` by `{result['date_col']}`",
        f"**Frequency:** {freq_names.get(result['frequency'], result['frequency'])} ({result['aggregation']})",
        f"**Period:** {s['date_range']['start']} to {s['date_range']['end']} ({s['total_periods']} periods)",
        "",
        "## Summary",
        f"- Total: {s['metric_summary']['total']:,.2f} | Mean: {s['metric_summary']['mean']:,.2f}",
        f"- Range: {s['metric_summary']['min']:,.2f} to {s['metric_summary']['max']:,.2f}",
        "",
        "## Growth",
        f"- First → Last: {g['first_period']:,.2f} → {g['last_period']:,.2f}",
        f"- Total: {g['total_growth_pct']:+.1f}% | Avg/period: {g['avg_period_growth_pct']:+.1f}%" 
            if g['total_growth_pct'] and g['avg_period_growth_pct'] else "- Growth: N/A",
        "",
        "## Trend",
    ]
    
    emoji = {"increasing": "📈", "slightly_increasing": "↗️", "flat": "➡️",
             "slightly_decreasing": "↘️", "decreasing": "📉"}.get(t['direction'], "❓")
    lines.append(f"- {emoji} **{t['direction'].replace('_', ' ').upper()}** (R²={t['r_squared']})")
    
    if sea['detected']:
        lines.extend(["", "## Seasonality", f"- ✓ {sea['period_name']} (strength: {sea['strength']})"])
    
    # Period table (last 12)
    periods = result['periods'][-12:]
    lines.extend(["", "## Periods", "| Period | Value | Change | % | MA |", "|--------|-------|--------|---|-----|"])
    
    for p in periods:
        v = _fmt(p['value'])
        c = _fmt(p['abs_change'], show_sign=True)
        pct = f"{p['pct_change']:+.1f}%" if p['pct_change'] is not None else "—"
        m = _fmt(p['moving_avg'])
        lines.append(f"| {p['period']} | {v} | {c} | {pct} | {m} |")
    
    if len(result['periods']) > 12:
        lines.append(f"\n*Last 12 of {len(result['periods'])} periods*")
    
    return "\n".join(lines)


__all__ = ["trend_analysis_tool", "analyze_trend"]
