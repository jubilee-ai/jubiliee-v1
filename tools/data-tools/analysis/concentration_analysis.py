"""
concentration_analysis - Analyze concentration and inequality in data.

Answers: "Do 20% of customers drive 80% of revenue?" "How concentrated is spend?"
Provides Pareto analysis, Gini coefficient, top-N contribution, cumulative distribution.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset


def _r(value: float, decimals: int = 2) -> Optional[float]:
    """Round to native float, handle NaN."""
    return None if pd.isna(value) else round(float(value), decimals)


def _gini(values: pd.Series) -> float:
    """Calculate Gini coefficient using the standard formula."""
    sorted_vals = values.sort_values().values
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1])


def analyze_concentration(
    dataset_ref: str,
    value_col: str,
    entity_col: Optional[str] = None,
    top_n_pcts: list[int] = [1, 5, 10, 20, 50],
) -> dict:
    """
    Analyze concentration/inequality in a numeric column.
    
    Args:
        dataset_ref: Reference to dataset
        value_col: Numeric column to analyze (e.g., revenue, spend)
        entity_col: Optional grouping column (e.g., customer_id). If provided, 
                    aggregates by entity first. If None, treats each row as an entity.
        top_n_pcts: Percentages to calculate top-N contribution for
    
    Returns:
        Dict with concentration metrics
    """
    df = resolve_dataset(dataset_ref)
    
    if value_col not in df.columns:
        return {"error": f"Column '{value_col}' not found"}
    
    # Aggregate by entity if specified
    if entity_col:
        if entity_col not in df.columns:
            return {"error": f"Entity column '{entity_col}' not found"}
        values = df.groupby(entity_col)[value_col].sum().dropna()
        n_entities = len(values)
        entity_name = entity_col
    else:
        values = df[value_col].dropna()
        n_entities = len(values)
        entity_name = "rows"
    
    if len(values) < 2:
        return {"error": "Need at least 2 values for concentration analysis"}
    
    # Filter to positive values only (concentration makes sense for positive quantities)
    values = values[values > 0]
    if len(values) < 2:
        return {"error": "Need at least 2 positive values"}
    
    total = values.sum()
    
    # Sort descending for top-N analysis
    sorted_desc = values.sort_values(ascending=False)
    
    # Calculate top-N contributions
    top_n_results = {}
    for pct in top_n_pcts:
        n = max(1, int(len(sorted_desc) * pct / 100))
        top_sum = sorted_desc.head(n).sum()
        top_n_results[f"top_{pct}pct"] = {
            "n_entities": n,
            "value_sum": _r(top_sum),
            "pct_of_total": _r(top_sum / total * 100, 1),
        }
    
    # Gini coefficient
    gini = _gini(values)
    
    # Interpret Gini
    if gini < 0.2:
        gini_interpretation = "low concentration (fairly equal)"
    elif gini < 0.4:
        gini_interpretation = "moderate concentration"
    elif gini < 0.6:
        gini_interpretation = "high concentration"
    else:
        gini_interpretation = "very high concentration (highly unequal)"
    
    # Cumulative distribution (Lorenz curve data) - 10 buckets
    sorted_asc = values.sort_values().reset_index(drop=True)
    cumsum = sorted_asc.cumsum()
    n = len(sorted_asc)
    
    lorenz_data = []
    for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        # Calculate index: pct% of n entities, clamped to valid range [0, n-1]
        idx = max(0, min(int(n * pct / 100) - 1, n - 1))
        lorenz_data.append({
            "pct_of_entities": pct,
            "pct_of_value": _r(cumsum.iloc[idx] / total * 100, 1),
        })
    
    # Find Pareto point (what % of entities drive ~80% of value)
    cumsum_desc = sorted_desc.cumsum()
    pct_cumsum = cumsum_desc / total * 100
    pareto_idx = (pct_cumsum >= 80).idxmax() if (pct_cumsum >= 80).any() else None
    if pareto_idx is not None:
        pareto_n = list(sorted_desc.index).index(pareto_idx) + 1
        pareto_pct = _r(pareto_n / n * 100, 1)
    else:
        pareto_pct = 100
        pareto_n = n
    
    return {
        "value_col": value_col,
        "entity_col": entity_col,
        "n_entities": n_entities,
        "n_positive": len(values),
        "total_value": _r(total),
        "mean_value": _r(values.mean()),
        "median_value": _r(values.median()),
        "std_value": _r(values.std()),
        "gini_coefficient": _r(gini, 3),
        "gini_interpretation": gini_interpretation,
        "pareto": {
            "pct_entities_for_80pct_value": pareto_pct,
            "n_entities_for_80pct_value": pareto_n,
        },
        "top_n_contribution": top_n_results,
        "lorenz_curve": lorenz_data,
    }


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class ConcentrationAnalysisInput(BaseModel):
    """Input schema for concentration_analysis_tool."""
    dataset_ref: str = Field(description="Dataset reference from previous operation or table name")
    value_col: str = Field(description="Numeric column to analyze (e.g., revenue, spend, amount)")
    entity_col: Optional[str] = Field(
        default=None, 
        description="Optional: Column to group by (e.g., customer_id, region). If omitted, each row is an entity."
    )


@tool(args_schema=ConcentrationAnalysisInput)
def concentration_analysis_tool(
    dataset_ref: str,
    value_col: str,
    entity_col: Optional[str] = None,
) -> str:
    """
    Analyze concentration and inequality in data (Pareto/80-20 analysis).
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Check if 20% of customers drive 80% of revenue
    - Measure inequality/concentration in spend, sales, or any metric
    - Calculate Gini coefficient
    - Understand cumulative distribution (Lorenz curve)
    - Find where to focus effort (top contributors)
    
    RETURNS: Gini coefficient, top-N contributions, Pareto point, Lorenz curve data.
    """
    result = analyze_concentration(dataset_ref, value_col, entity_col)
    
    if "error" in result:
        return f"✗ {result['error']}"
    
    # Build output
    entity_desc = f"by `{result['entity_col']}`" if result['entity_col'] else "(each row)"
    
    lines = [
        "# Concentration Analysis",
        f"**Metric:** `{result['value_col']}` {entity_desc}",
        f"**Entities:** {result['n_entities']:,} ({result['n_positive']:,} with positive values)",
        f"**Total:** {result['total_value']:,.2f} | Mean: {result['mean_value']:,.2f} | Median: {result['median_value']:,.2f}",
        "",
        "## Gini Coefficient",
        f"- **Gini:** {result['gini_coefficient']} → {result['gini_interpretation']}",
        f"- (0 = perfect equality, 1 = maximum concentration)",
        "",
        "## Pareto Analysis",
        f"- **{result['pareto']['pct_entities_for_80pct_value']}%** of entities drive **80%** of total value",
        f"- ({result['pareto']['n_entities_for_80pct_value']:,} out of {result['n_positive']:,} entities)",
        "",
        "## Top-N Contribution",
        "| Top % | # Entities | Value | % of Total |",
        "|-------|------------|-------|------------|",
    ]
    
    for key, data in result['top_n_contribution'].items():
        pct_label = key.replace("top_", "").replace("pct", "%")
        lines.append(f"| {pct_label} | {data['n_entities']:,} | {data['value_sum']:,.0f} | {data['pct_of_total']}% |")
    
    # Lorenz curve (cumulative distribution)
    lines.extend([
        "",
        "## Cumulative Distribution (Lorenz Curve)",
        "| % of Entities | % of Value |",
        "|---------------|------------|",
    ])
    
    for point in result['lorenz_curve']:
        gap = point['pct_of_entities'] - point['pct_of_value']
        indicator = " ⚠️" if gap > 30 else ""
        lines.append(f"| {point['pct_of_entities']}% | {point['pct_of_value']}%{indicator} |")
    
    return "\n".join(lines)


__all__ = ["concentration_analysis_tool", "analyze_concentration"]

