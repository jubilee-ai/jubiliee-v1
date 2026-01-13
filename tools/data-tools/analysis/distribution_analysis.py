"""
distribution_analysis - Analyze the shape and distribution of numeric columns.

Answers: "Is income normally distributed?" "Are there multiple peaks?" "What are key percentiles?"
Reveals hidden patterns through shape analysis, modality detection, and concentration metrics.
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

# =============================================================================
# HELPERS
# =============================================================================

def _r(value: float, decimals: int = 2) -> Optional[float]:
    """Round to native float, handle NaN."""
    return None if pd.isna(value) else round(float(value), decimals)


def _interpret_shape(skew: float, kurt: float) -> str:
    """Generate human-readable shape interpretation."""
    parts = []
    
    if skew > 1:
        parts.append("strongly right-skewed (long tail of high values)")
    elif skew > 0.5:
        parts.append("moderately right-skewed")
    elif skew < -1:
        parts.append("strongly left-skewed (long tail of low values)")
    elif skew < -0.5:
        parts.append("moderately left-skewed")
    else:
        parts.append("approximately symmetric")
    
    if kurt > 3:
        parts.append("heavy tails (many outliers)")
    elif kurt > 1:
        parts.append("somewhat heavy tails")
    elif kurt < -1:
        parts.append("light tails (few extreme values)")
    
    return "; ".join(parts)


def _detect_modality(series: pd.Series) -> dict:
    """Detect unimodal, bimodal, or multimodal distribution using smoothed histogram."""
    if len(series) < 20:
        return {"type": "unknown", "peaks": 0, "peak_locations": []}
    
    # Create and smooth histogram
    counts, edges = np.histogram(series, bins=30)
    smoothed = np.convolve(counts, [1, 2, 3, 2, 1], mode='same') / 9
    max_h = smoothed.max()
    data_range = series.max() - series.min()
    
    # Find significant peaks with prominence
    peaks = []
    for i in range(2, len(smoothed) - 2):
        if smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
            left_min = smoothed[max(0, i-3):i].min()
            right_min = smoothed[i+1:min(len(smoothed), i+4)].min()
            prominence = smoothed[i] - max(left_min, right_min)
            
            if smoothed[i] > 0.2 * max_h and prominence > 0.1 * max_h:
                center = (edges[i] + edges[i+1]) / 2
                peaks.append({"center": _r(center), "height": int(counts[i])})
    
    # Merge nearby peaks (within 15% of range)
    merged = []
    for p in sorted(peaks, key=lambda x: x["center"]):
        if not merged or p["center"] - merged[-1]["center"] >= 0.15 * data_range:
            merged.append(p)
        elif p["height"] > merged[-1]["height"]:
            merged[-1] = p
    
    n = len(merged)
    modality = {0: "uniform", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")
    return {"type": modality, "peaks": n, "peak_locations": merged[:5]}


def _compute_concentration(series: pd.Series) -> dict:
    """Compute Pareto/concentration metrics for positive-valued series."""
    if len(series) < 10 or series.sum() == 0:
        return {}
    
    sorted_vals = series.sort_values(ascending=False).values
    cumsum = np.cumsum(sorted_vals)
    total = cumsum[-1]
    n = len(sorted_vals)
    
    return {
        "top_10pct_share": _r(cumsum[int(n * 0.1)] / total, 2) if n > 10 else None,
        "pct_for_80_of_total": _r(np.searchsorted(cumsum, 0.8 * total) / n * 100, 0),
    }


# =============================================================================
# CORE FUNCTION
# =============================================================================

def analyze_distribution(
    dataset_ref: str,
    columns: Optional[list[str]] = None,
    n_bins: int = 10,
) -> dict:
    """Analyze distribution of numeric columns."""
    df = resolve_dataset(dataset_ref)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    # Filter to requested columns or limit to 10
    cols = [c for c in columns if c in numeric_cols] if columns else numeric_cols[:10]
    if not cols:
        return {"error": f"No valid numeric columns in: {columns}"}
    
    percentiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    results = {}
    
    for col in cols:
        s = df[col].dropna()
        if len(s) == 0:
            results[col] = {"error": "no valid data"}
            continue
        
        # Use pandas built-in functions
        skew = s.skew()
        kurt = s.kurtosis()
        is_normal = abs(skew) < 0.5 and abs(kurt) < 1
        
        # Percentiles using pandas
        pct = {f"p{int(p*100)}": _r(s.quantile(p)) for p in percentiles}
        
        # Histogram using numpy
        counts, edges = np.histogram(s, bins=n_bins)
        histogram = [
            {"range": f"{edges[i]:.2f}-{edges[i+1]:.2f}", 
             "count": int(counts[i]), 
             "pct": _r(counts[i] / len(s) * 100, 1)}
            for i in range(len(counts))
        ]
        
        results[col] = {
            "n": len(s),
            "n_missing": int(df[col].isna().sum()),
            "mean": _r(s.mean()),
            "std": _r(s.std()),
            "min": _r(s.min()),
            "max": _r(s.max()),
            "range": _r(s.max() - s.min()),
            "percentiles": pct,
            "skewness": _r(skew, 3),
            "kurtosis": _r(kurt, 3),
            "is_approximately_normal": is_normal,
            "shape_interpretation": _interpret_shape(skew, kurt),
            "modality": _detect_modality(s),
            "histogram": histogram,
            "concentration": _compute_concentration(s) if (s > 0).all() else {},
        }
    
    return {"columns_analyzed": cols, "total_rows": len(df), "distributions": results}


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class DistributionAnalysisInput(BaseModel):
    """Input schema for distribution_analysis_tool."""
    dataset_ref: str = Field(description="Dataset reference from previous operation or table name")
    columns: Optional[list[str]] = Field(default=None, description="Numeric columns to analyze (default: up to 10)")
    n_bins: int = Field(default=10, description="Number of histogram bins (5-50)")


@tool(args_schema=DistributionAnalysisInput)
def distribution_analysis_tool(
    dataset_ref: str,
    columns: Optional[list[str]] = None,
    n_bins: int = 10,
) -> str:
    """
    Analyze the distribution and shape of numeric columns.
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Check if a column is normally distributed
    - Detect bimodal or multimodal distributions (hidden segments)
    - Understand spread and concentration of values
    - Get key percentiles (p10, p25, p50, p75, p90, p95, p99)
    - Identify skewness, heavy tails, or outlier-prone distributions
    
    RETURNS: Percentiles, histogram, skewness, kurtosis, modality, concentration analysis.
    """
    try:
        result = analyze_distribution(dataset_ref, columns, n_bins)
        if "error" in result:
            return f"✗ {result['error']}"
        
        lines = [
            "# Distribution Analysis",
            "",
            f"**Columns analyzed:** {len(result['columns_analyzed'])}",
            f"**Total rows:** {result['total_rows']:,}",
            "",
        ]
        
        for col, d in result["distributions"].items():
            if "error" in d:
                lines.append(f"## `{col}` — {d['error']}")
                continue
            
            lines.extend([
                f"## `{col}`",
                "",
                f"**Range:** {d['min']:,.2f} to {d['max']:,.2f} (span: {d['range']:,.2f})",
                f"**Mean:** {d['mean']:,.2f} | **Std:** {d['std']:,.2f}",
                f"**N:** {d['n']:,} ({d['n_missing']} missing)",
                "",
                "### Percentiles",
                " | ".join(f"**{k}:** {v:,.2f}" for k, v in d["percentiles"].items()),
                "",
                "### Shape Analysis",
                f"- **Normal:** {'✓' if d['is_approximately_normal'] else '✗'} {d['shape_interpretation']}",
                f"- **Skewness:** {d['skewness']} | **Kurtosis:** {d['kurtosis']}",
            ])
            
            # Modality
            mod = d["modality"]
            if mod["type"] == "bimodal":
                peaks = ", ".join(str(p["center"]) for p in mod["peak_locations"])
                lines.append(f"- **Modality:** ⚠️ BIMODAL — peaks at {peaks}")
            elif mod["type"] == "multimodal":
                lines.append(f"- **Modality:** ⚠️ MULTIMODAL ({mod['peaks']} peaks)")
            else:
                lines.append(f"- **Modality:** {mod['type']}")
            lines.append("")
            
            # Concentration
            if d["concentration"]:
                c = d["concentration"]
                lines.append("### Concentration")
                if c.get("top_10pct_share"):
                    lines.append(f"- Top 10% accounts for {c['top_10pct_share']*100:.0f}% of total")
                if c.get("pct_for_80_of_total"):
                    lines.append(f"- {c['pct_for_80_of_total']:.0f}% of items account for 80% of total")
                lines.append("")
            
            # Histogram table
            lines.extend(["### Histogram", "| Bin Range | Count | Pct | Cumulative |", "|-----------|-------|-----|------------|"])
            cum = 0
            for h in d["histogram"]:
                cum += h["pct"]
                lines.append(f"| {h['range']} | {h['count']:,} | {h['pct']}% | {cum:.1f}% |")
            lines.append("")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"✗ Distribution analysis failed: {type(e).__name__}: {e}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["distribution_analysis_tool", "analyze_distribution"]
