"""
eda_report - Compact dataset profiler for AI agent consumption.

Generates shape, types, missingness, cardinality, distributions, correlations,
and target associations. Emits alerts and actionable recommendations.
Bounded by caps to prevent context explosion.

Run once per dataset after validation passes.
"""

import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from scipy import stats

# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class ShapeInfo(TypedDict):
    """Shape information for the dataset."""
    rows: int
    columns: int


class SchemaColumn(TypedDict, total=False):
    """Schema info for a single column."""
    column: str
    dtype: str
    null_pct: float
    unique: int  # Only present for categorical columns


class NumericColumnSummary(TypedDict):
    """Summary statistics for a numeric column."""
    column: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    p5: float
    p95: float
    skew: float
    outliers_pct: float


class CategoryValue(TypedDict):
    """A single category value with its percentage."""
    value: str
    pct: float


class CategoricalColumnSummary(TypedDict):
    """Summary for a categorical column."""
    column: str
    unique: int
    top_values: list[CategoryValue]
    id_like: bool


class TargetAnalysis(TypedDict, total=False):
    """Analysis of the target variable."""
    column: str
    task: str
    # Classification fields
    class_counts: dict[str, int]
    imbalance_ratio: float
    # Regression fields
    mean: float
    std: float
    skew: float
    # Common
    recommendation: str


class TargetAssociation(TypedDict, total=False):
    """Feature-target association info."""
    column: str
    metric: str  # 'auc' or 'correlation'
    value: float
    direction: str  # Only for AUC


class CorrelationPair(TypedDict):
    """A pair of highly correlated columns."""
    col1: str
    col2: str
    corr: float


class CorrelationsInfo(TypedDict):
    """Correlation analysis results."""
    high_pairs: list[CorrelationPair]


class Alert(TypedDict, total=False):
    """An alert about a data issue."""
    type: str
    message: str
    column: str
    columns: list[str]
    skew: float
    unique: int
    corr: float
    null_pct: float


class EdaSummary(TypedDict):
    """High-level summary for the agent."""
    key_findings: list[str]
    top_actions: list[str]


class EdaReportType(TypedDict, total=False):
    """Complete EDA report structure."""
    shape: ShapeInfo
    schema: list[SchemaColumn]
    numeric_summary: list[NumericColumnSummary]
    categorical_summary: list[CategoricalColumnSummary]
    target_analysis: TargetAnalysis
    target_associations: list[TargetAssociation]
    correlations: CorrelationsInfo
    alerts: list[Alert]
    recommendations: list[str]
    summary: EdaSummary

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset

# =============================================================================
# HELPERS
# =============================================================================

def _r(value, decimals: int = 2) -> float:
    """Round and convert to native float."""
    return float(round(value, decimals))


def _safe_skew(series: pd.Series) -> float:
    """Calculate skewness, handling edge cases."""
    try:
        clean = series.dropna()
        if len(clean) < 3:
            return 0.0
        return float(stats.skew(clean, nan_policy='omit'))
    except Exception:
        return 0.0


def _outlier_pct(series: pd.Series) -> float:
    """Calculate percentage of outliers using IQR method."""
    clean = series.dropna()
    if len(clean) < 4:
        return 0.0
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((clean < lower) | (clean > upper)).sum()
    return float(outliers / len(clean))


def _is_id_like(series: pd.Series, threshold: float = 0.95) -> bool:
    """Check if column looks like an ID (high cardinality, unique)."""
    if series.dtype not in ['object', 'string', 'int64', 'int32']:
        return False
    unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
    return unique_ratio >= threshold


def _compute_auc(feature: pd.Series, target: pd.Series) -> tuple[float, str]:
    """Compute AUC between feature and binary target."""
    try:
        from sklearn.metrics import roc_auc_score

        # Align and drop nulls
        mask = feature.notna() & target.notna()
        x = feature[mask].astype(float).values
        y = target[mask].astype(int).values
        
        if len(np.unique(y)) != 2 or len(x) < 10:
            return 0.5, "n/a"
        
        # Handle non-numeric features
        if not np.issubdtype(x.dtype, np.number):
            return 0.5, "n/a"
        
        auc = roc_auc_score(y, x)
        direction = "higher → class_1" if auc >= 0.5 else "lower → class_1"
        # Normalize AUC to be >= 0.5
        auc = max(auc, 1 - auc)
        return float(auc), direction
    except Exception:
        return 0.5, "n/a"


def _compute_correlation(feature: pd.Series, target: pd.Series) -> float:
    """Compute correlation between feature and target."""
    try:
        mask = feature.notna() & target.notna()
        if mask.sum() < 3:
            return 0.0
        return float(feature[mask].corr(target[mask]))
    except Exception:
        return 0.0


# =============================================================================
# CORE EDA FUNCTIONS
# =============================================================================

def _get_schema(df: pd.DataFrame) -> list[dict]:
    """Get column schema with types and null info."""
    schema = []
    for col in df.columns:
        info = {
            "column": col,
            "dtype": str(df[col].dtype),
            "null_pct": _r(df[col].isna().mean(), 4),
        }
        if df[col].dtype in ['object', 'category', 'string']:
            info["unique"] = int(df[col].nunique())
        schema.append(info)
    return schema


def _numeric_summary(df: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Generate summary statistics for numeric columns."""
    summaries = []
    for col in columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        summaries.append({
            "column": col,
            "mean": _r(series.mean()),
            "median": _r(series.median()),
            "std": _r(series.std()),
            "min": _r(series.min()),
            "max": _r(series.max()),
            "p5": _r(series.quantile(0.05)),
            "p95": _r(series.quantile(0.95)),
            "skew": _r(_safe_skew(series)),
            "outliers_pct": _r(_outlier_pct(series), 4),
        })
    return summaries


def _categorical_summary(df: pd.DataFrame, columns: list[str], top_k: int = 10) -> list[dict]:
    """Generate summary for categorical columns."""
    summaries = []
    for col in columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        
        value_counts = series.value_counts(normalize=True).head(top_k)
        top_values = [
            {"value": str(v), "pct": _r(p, 4)}
            for v, p in value_counts.items()
        ]
        
        summaries.append({
            "column": col,
            "unique": int(df[col].nunique()),
            "top_values": top_values,
            "id_like": _is_id_like(df[col]),
        })
    return summaries


def _analyze_target(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    """Analyze target variable."""
    series = df[target_col].dropna()
    
    result = {
        "column": target_col,
        "task": task_type,
    }
    
    if task_type == "classification":
        class_counts = series.value_counts().to_dict()
        result["class_counts"] = {str(k): int(v) for k, v in class_counts.items()}
        
        if len(class_counts) >= 2:
            counts = sorted(class_counts.values(), reverse=True)
            result["imbalance_ratio"] = _r(counts[0] / counts[-1])
            
            if result["imbalance_ratio"] > 3:
                result["recommendation"] = "Use class_weight='balanced' or SMOTE"
            else:
                result["recommendation"] = "Target is reasonably balanced"
    else:
        # Regression
        result["mean"] = _r(series.mean())
        result["std"] = _r(series.std())
        result["skew"] = _r(_safe_skew(series))
        
        if abs(result["skew"]) > 1:
            result["recommendation"] = "Consider log transform on target"
        else:
            result["recommendation"] = "Target distribution is acceptable"
    
    return result


def _target_associations(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    numeric_cols: list[str],
    max_features: int = 20
) -> list[dict]:
    """Compute feature-target associations."""
    associations = []
    target = df[target_col]
    
    for col in numeric_cols[:max_features]:
        if col == target_col:
            continue
        
        if task_type == "classification":
            auc, direction = _compute_auc(df[col], target)
            if auc > 0.51:  # Only report meaningful associations
                associations.append({
                    "column": col,
                    "metric": "auc",
                    "value": _r(auc, 3),
                    "direction": direction,
                })
        else:
            corr = _compute_correlation(df[col], target)
            if abs(corr) > 0.1:  # Only report meaningful correlations
                associations.append({
                    "column": col,
                    "metric": "correlation",
                    "value": _r(corr, 3),
                })
    
    # Sort by absolute value
    associations.sort(key=lambda x: abs(x["value"]), reverse=True)
    return associations


def _find_high_correlations(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = 0.85,
    max_pairs: int = 20
) -> list[dict]:
    """Find highly correlated column pairs."""
    if len(numeric_cols) < 2:
        return []
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    pairs = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if pd.notna(corr) and abs(corr) >= threshold:
                pairs.append({
                    "col1": col1,
                    "col2": col2,
                    "corr": _r(corr, 3),
                })
    
    # Sort by absolute correlation
    pairs.sort(key=lambda x: abs(x["corr"]), reverse=True)
    return pairs[:max_pairs]


def _generate_alerts(
    schema: list[dict],
    numeric_summary: list[dict],
    categorical_summary: list[dict],
    target_analysis: Optional[dict],
    correlations: list[dict],
) -> list[dict]:
    """Generate alerts based on findings."""
    alerts = []
    
    # Imbalanced target
    if target_analysis and target_analysis.get("imbalance_ratio", 1) > 3:
        ratio = target_analysis["imbalance_ratio"]
        alerts.append({
            "type": "imbalanced_target",
            "message": f"Target is {ratio:.1f}:1 imbalanced",
        })
    
    # High skew
    for num in numeric_summary:
        if abs(num["skew"]) > 2:
            alerts.append({
                "type": "high_skew",
                "column": num["column"],
                "skew": num["skew"],
            })
    
    # ID-like columns
    for cat in categorical_summary:
        if cat["id_like"]:
            alerts.append({
                "type": "id_like",
                "column": cat["column"],
                "unique": cat["unique"],
            })
    
    # Redundant columns
    for pair in correlations:
        if abs(pair["corr"]) >= 0.95:
            alerts.append({
                "type": "redundant",
                "columns": [pair["col1"], pair["col2"]],
                "corr": pair["corr"],
            })
    
    # High cardinality categorical
    for cat in categorical_summary:
        if cat["unique"] > 50 and not cat["id_like"]:
            alerts.append({
                "type": "high_cardinality",
                "column": cat["column"],
                "unique": cat["unique"],
            })
    
    # High null percentage (use pre-computed schema)
    for col_info in schema:
        if col_info["null_pct"] > 0.3:
            alerts.append({
                "type": "high_nulls",
                "column": col_info["column"],
                "null_pct": col_info["null_pct"],
            })
    
    return alerts


def _generate_recommendations(
    alerts: list[dict],
    numeric_summary: list[dict],
    categorical_summary: list[dict],
    target_analysis: Optional[dict],
    correlations: list[dict],
) -> list[str]:
    """Generate actionable recommendations."""
    recs = []
    
    # Target recommendations
    if target_analysis and target_analysis.get("recommendation"):
        recs.append(target_analysis["recommendation"])
    
    # Skew recommendations
    for num in numeric_summary:
        if abs(num["skew"]) > 2:
            recs.append(f"Apply log transform or bin_column to {num['column']} (skew={num['skew']})")
    
    # Redundant column recommendations
    for pair in correlations:
        if abs(pair["corr"]) >= 0.95:
            recs.append(f"Consider dropping {pair['col2']} (redundant with {pair['col1']}, r={pair['corr']})")
    
    # ID-like recommendations
    for cat in categorical_summary:
        if cat["id_like"]:
            recs.append(f"Don't encode {cat['column']} as feature (ID-like, {cat['unique']} unique)")
    
    # High cardinality recommendations
    for cat in categorical_summary:
        if cat["unique"] > 50 and not cat["id_like"]:
            recs.append(f"Use target encoding for {cat['column']} ({cat['unique']} categories)")
    
    # High null recommendations
    for alert in alerts:
        if alert["type"] == "high_nulls":
            pct = alert["null_pct"]
            col = alert["column"]
            if pct > 0.5:
                recs.append(f"Consider dropping {col} ({pct:.0%} null)")
            else:
                recs.append(f"Impute {col} ({pct:.0%} null)")
    
    return recs[:10]  # Cap recommendations


def _generate_summary(
    alerts: list[dict],
    recommendations: list[str],
    target_analysis: Optional[dict],
    numeric_summary: list[dict],
) -> dict:
    """Generate high-level summary for agent."""
    key_findings = []
    top_actions = []
    
    # Target findings
    if target_analysis:
        if target_analysis.get("imbalance_ratio", 1) > 3:
            ratio = target_analysis["imbalance_ratio"]
            key_findings.append(f"Imbalanced target ({ratio:.1f}:1)")
            top_actions.append("class_weight='balanced'")
    
    # Skew findings
    high_skew = [n for n in numeric_summary if abs(n["skew"]) > 2]
    if high_skew:
        cols = [n["column"] for n in high_skew[:3]]
        key_findings.append(f"{', '.join(cols)} {'is' if len(cols)==1 else 'are'} highly skewed")
        top_actions.append(f"bin_column({cols[0]}, strategy='quantile')")
    
    # Redundant findings
    redundant = [a for a in alerts if a["type"] == "redundant"]
    if redundant:
        key_findings.append(f"{len(redundant)} redundant column pair(s) found")
        top_actions.append(f"drop({redundant[0]['columns'][1]})")
    
    # ID-like findings
    id_like = [a for a in alerts if a["type"] == "id_like"]
    if id_like:
        key_findings.append(f"{len(id_like)} ID-like column(s) detected")
    
    return {
        "key_findings": key_findings[:5],
        "top_actions": top_actions[:5],
    }


# =============================================================================
# MAIN EDA FUNCTION
# =============================================================================

def run_eda_report(
    dataset_ref: str,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    sample_n: int = 100000,
    caps: Optional[dict] = None,
) -> EdaReportType:
    """
    Generate comprehensive EDA report for a dataset.
    
    Args:
        dataset_ref: Reference to dataset
        target_col: Optional target column for supervised analysis
        task_type: 'classification' or 'regression'
        sample_n: Max rows to sample
        caps: Limits for output size
    
    Returns:
        EdaReportType with shape, schema, summaries, correlations, alerts, and recommendations
    """
    df = resolve_dataset(dataset_ref)
    
    # Apply caps
    caps = caps or {}
    top_k_categories = caps.get("top_k_categories", 10)
    max_corr_pairs = caps.get("max_corr_pairs", 20)
    max_columns = caps.get("max_columns", 100)
    
    # Sample if needed
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    
    # Limit columns
    if len(df.columns) > max_columns:
        df = df.iloc[:, :max_columns]
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # Build report
    report = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "schema": _get_schema(df),
        "numeric_summary": _numeric_summary(df, numeric_cols),
        "categorical_summary": _categorical_summary(df, categorical_cols, top_k_categories),
    }
    
    # Target analysis
    target_analysis = None
    if target_col and target_col in df.columns:
        # Infer task type if not provided
        if task_type is None:
            if df[target_col].nunique() <= 10:
                task_type = "classification"
            else:
                task_type = "regression"
        
        target_analysis = _analyze_target(df, target_col, task_type)
        report["target_analysis"] = target_analysis
        
        # Target associations
        report["target_associations"] = _target_associations(
            df, target_col, task_type, numeric_cols
        )
    
    # Correlations
    high_corr = _find_high_correlations(df, numeric_cols, max_pairs=max_corr_pairs)
    report["correlations"] = {"high_pairs": high_corr}
    
    # Alerts
    alerts = _generate_alerts(
        report["schema"],
        report["numeric_summary"],
        report["categorical_summary"],
        target_analysis,
        high_corr,
    )
    report["alerts"] = alerts
    
    # Recommendations
    recommendations = _generate_recommendations(
        alerts,
        report["numeric_summary"],
        report["categorical_summary"],
        target_analysis,
        high_corr,
    )
    report["recommendations"] = recommendations
    
    # Summary
    report["summary"] = _generate_summary(
        alerts, recommendations, target_analysis, report["numeric_summary"]
    )
    
    return report


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class EdaCaps(BaseModel):
    """Caps for EDA report output size."""
    top_k_categories: int = Field(default=10, description="Max categories to show per column")
    max_corr_pairs: int = Field(default=20, description="Max correlation pairs to report")
    max_columns: int = Field(default=100, description="Max columns to analyze")


class EdaReportInput(BaseModel):
    """Input schema for eda_report tool."""
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    target_col: Optional[str] = Field(
        default=None,
        description="Target column for supervised analysis (enables target associations)"
    )
    task_type: Optional[Literal["classification", "regression"]] = Field(
        default=None,
        description="Task type. Auto-detected if not provided."
    )
    sample_n: int = Field(
        default=100000,
        description="Max rows to sample for analysis"
    )
    caps: Optional[EdaCaps] = Field(
        default=None,
        description="Caps for output size"
    )


@tool(args_schema=EdaReportInput)
def eda_report_tool(
    dataset_ref: str,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None,
    sample_n: int = 100000,
    caps: Optional[dict] = None,
) -> str:
    """
    Generate a compact dataset profile for understanding shape, types, distributions, and correlations.
    
    WHAT IT PROVIDES:
    - Shape and schema (columns, types, null percentages)
    - Numeric stats (mean, median, std, skew, outliers)
    - Categorical stats (unique counts, top values, ID-like detection)
    - Target analysis (if target_col provided): class balance, associations
    - High correlations between features
    - Alerts (imbalanced target, high skew, ID-like columns, redundant pairs)
    - Actionable recommendations
    
    WHEN TO USE:
    Run once per dataset after validation passes. Provides foundation for all transformation decisions.
    
    EXAMPLES:
    ```
    # Basic profiling
    eda_report(dataset_ref="loan_data")
    
    # With target analysis for classification
    eda_report(dataset_ref="loan_data", target_col="default", task_type="classification")
    
    # With sampling for large datasets
    eda_report(dataset_ref="big_data", sample_n=50000)
    ```
    """
    try:
        caps_dict = caps if isinstance(caps, dict) else (caps.model_dump() if caps else None)
        
        report = run_eda_report(
            dataset_ref=dataset_ref,
            target_col=target_col,
            task_type=task_type,
            sample_n=sample_n,
            caps=caps_dict,
        )
        
        # Format output for agent
        lines = []
        
        # Header
        lines.append("# EDA Report")
        lines.append("")
        
        # Shape
        shape = report["shape"]
        lines.append(f"**Shape:** {shape['rows']:,} rows × {shape['columns']} columns")
        lines.append("")
        
        # Schema (compact)
        lines.append("## Schema")
        for col in report["schema"][:15]:
            null_info = f", {col['null_pct']:.0%} null" if col['null_pct'] > 0 else ""
            unique_info = f", {col.get('unique', '')} unique" if 'unique' in col else ""
            lines.append(f"- `{col['column']}`: {col['dtype']}{null_info}{unique_info}")
        if len(report["schema"]) > 15:
            lines.append(f"- ... +{len(report['schema']) - 15} more columns")
        lines.append("")
        
        # Numeric summary (compact)
        if report["numeric_summary"]:
            lines.append("## Numeric Features")
            for num in report["numeric_summary"][:10]:
                skew_warn = " ⚠️" if abs(num["skew"]) > 2 else ""
                lines.append(
                    f"- `{num['column']}`: mean={num['mean']:,.0f}, "
                    f"median={num['median']:,.0f}, std={num['std']:,.0f}, "
                    f"skew={num['skew']}{skew_warn}"
                )
            lines.append("")
        
        # Categorical summary (compact)
        if report["categorical_summary"]:
            lines.append("## Categorical Features")
            for cat in report["categorical_summary"][:10]:
                id_warn = " 🔑 ID-like" if cat["id_like"] else ""
                top = cat["top_values"][0] if cat["top_values"] else {"value": "n/a", "pct": 0}
                lines.append(
                    f"- `{cat['column']}`: {cat['unique']} unique, "
                    f"top='{top['value']}' ({top['pct']:.0%}){id_warn}"
                )
            lines.append("")
        
        # Target analysis
        if "target_analysis" in report:
            ta = report["target_analysis"]
            lines.append("## Target Analysis")
            lines.append(f"**Column:** `{ta['column']}` ({ta['task']})")
            if ta["task"] == "classification":
                counts = ta.get("class_counts", {})
                lines.append(f"**Class counts:** {counts}")
                if "imbalance_ratio" in ta:
                    ratio = ta["imbalance_ratio"]
                    warn = " ⚠️" if ratio > 3 else ""
                    lines.append(f"**Imbalance ratio:** {ratio:.1f}:1{warn}")
            else:
                lines.append(f"**Mean:** {ta.get('mean', 0):,.2f}, Skew: {ta.get('skew', 0):.2f}")
            lines.append(f"**Recommendation:** {ta.get('recommendation', 'n/a')}")
            lines.append("")
        
        # Target associations
        if report.get("target_associations"):
            lines.append("## Top Feature-Target Associations")
            for assoc in report["target_associations"][:10]:
                direction = f" ({assoc.get('direction', '')})" if 'direction' in assoc else ""
                lines.append(
                    f"- `{assoc['column']}`: {assoc['metric']}={assoc['value']:.3f}{direction}"
                )
            lines.append("")
        
        # Correlations
        if report["correlations"]["high_pairs"]:
            lines.append("## High Correlations")
            for pair in report["correlations"]["high_pairs"][:10]:
                warn = " ⚠️ redundant" if abs(pair["corr"]) >= 0.95 else ""
                lines.append(f"- `{pair['col1']}` ↔ `{pair['col2']}`: r={pair['corr']}{warn}")
            lines.append("")
        
        # Alerts
        if report["alerts"]:
            lines.append("## ⚠️ Alerts")
            for alert in report["alerts"][:10]:
                if alert["type"] == "imbalanced_target":
                    lines.append(f"- 🔴 {alert['message']}")
                elif alert["type"] == "high_skew":
                    lines.append(f"- 🟡 High skew on `{alert['column']}` (skew={alert['skew']})")
                elif alert["type"] == "id_like":
                    lines.append(f"- 🔑 `{alert['column']}` looks like an ID ({alert['unique']} unique)")
                elif alert["type"] == "redundant":
                    lines.append(f"- 🔄 `{alert['columns'][0]}` and `{alert['columns'][1]}` are redundant (r={alert['corr']})")
                elif alert["type"] == "high_cardinality":
                    lines.append(f"- 📊 `{alert['column']}` has high cardinality ({alert['unique']} categories)")
                elif alert["type"] == "high_nulls":
                    lines.append(f"- ❌ `{alert['column']}` has {alert['null_pct']:.0%} nulls")
            lines.append("")
        
        # Recommendations
        if report["recommendations"]:
            lines.append("## 📋 Recommendations")
            for rec in report["recommendations"]:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Summary
        if report["summary"]["key_findings"] or report["summary"]["top_actions"]:
            lines.append("## Summary")
            if report["summary"]["key_findings"]:
                lines.append(f"**Key findings:** {', '.join(report['summary']['key_findings'])}")
            if report["summary"]["top_actions"]:
                lines.append(f"**Top actions:** {', '.join(report['summary']['top_actions'])}")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"✗ EDA report failed: {type(e).__name__}: {e}"


# Export
eda_tools = [eda_report_tool]

__all__ = [
    "eda_report_tool",
    "run_eda_report",
    "eda_tools",
    # Types
    "EdaReportType",
    "ShapeInfo",
    "SchemaColumn",
    "NumericColumnSummary",
    "CategoricalColumnSummary",
    "CategoryValue",
    "TargetAnalysis",
    "TargetAssociation",
    "CorrelationPair",
    "CorrelationsInfo",
    "Alert",
    "EdaSummary",
]

