"""
Feature Engineering: Run all analysis tools, then one LLM call.

Architecture:
1. Runs ALL analysis tools upfront on training data only
2. Compiles all results into a single context
3. Makes ONE LLM call to select and specify features via structured output

Contains the full Formula DSL (operation types, schemas), validation logic,
and the one-shot analysis → LLM pipeline.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Literal, Optional, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..utils.prompts import get_feature_engineering_prompt

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from analysis import (analyze_concentration, analyze_distribution,
                      compute_correlation_matrix, compute_group_summary,
                      run_eda_report, run_feature_diagnostics)
from transformations.tool_utils import resolve_dataset


# =============================================================================
# FORMULA DSL — structured operations for feature engineering
# =============================================================================

OperationType = Literal[
    "passthrough",      # Use column as-is
    "expression",       # Computed expression (pandas eval or fallback eval)
    "bin",              # Discretize numeric into buckets
    "one_hot",          # One-hot encode categorical
    "ordinal",          # Ordinal encode with specified order
    "group_agg",        # Group-by aggregation
    "rolling",          # Rolling window aggregation
    "date_extract",     # Extract datetime parts (year, month, day, etc.)
    "date_diff",        # Difference between two date columns
    "target_encode",    # Smoothed target-mean encoding (fit on train only)
    "frequency_encode", # Replace categories with their frequency proportion
]

AggFunction = Literal["mean", "sum", "min", "max", "count", "std", "median", "nunique", "first", "last"]
DatePart = Literal["year", "month", "day", "dayofweek", "hour", "quarter", "weekofyear"]
BinStrategy = Literal["uniform", "quantile"]


class PassthroughOp(BaseModel):
    """Use a column as-is without transformation."""
    op: Literal["passthrough"] = "passthrough"
    column: str = Field(description="Source column name")


class ExpressionOp(BaseModel):
    """Compute a new column from an expression. Supports: +, -, *, /, **, comparisons, and functions (abs, sqrt, log, round, min, max)."""
    op: Literal["expression"] = "expression"
    expression: str = Field(
        description="Expression using column names. Examples: 'loan_amount / annual_income', 'age * 12', 'sqrt(income)'"
    )
    source_columns: list[str] = Field(description="Columns used in the expression")


class BinOp(BaseModel):
    """Discretize a numeric column into bins/buckets."""
    op: Literal["bin"] = "bin"
    column: str = Field(description="Numeric column to bin")
    bins: Union[int, list[float]] = Field(
        description="Number of bins (int) OR list of bin edges. Example: 5 or [0, 25, 50, 75, 100]"
    )
    strategy: BinStrategy = Field(default="quantile", description="'uniform' (equal-width) or 'quantile' (equal-frequency)")
    labels: Optional[list[str]] = Field(default=None, description="Optional labels for bins")


class OneHotOp(BaseModel):
    """One-hot encode a categorical column into multiple binary columns."""
    op: Literal["one_hot"] = "one_hot"
    column: str = Field(description="Categorical column to encode")
    drop_first: bool = Field(default=True, description="Drop first category to avoid multicollinearity")


class OrdinalOp(BaseModel):
    """Ordinal encode a categorical column with specified order."""
    op: Literal["ordinal"] = "ordinal"
    column: str = Field(description="Categorical column to encode")
    order: list[str] = Field(description="Categories in order from lowest to highest. Example: ['low', 'medium', 'high']")


class GroupAggOp(BaseModel):
    """Create an aggregated feature by computing a statistic within groups."""
    op: Literal["group_agg"] = "group_agg"
    column: str = Field(description="Column to aggregate")
    agg: AggFunction = Field(description="Aggregation function: mean, sum, min, max, count, std, median, nunique")
    group_by: list[str] = Field(description="Column(s) to group by")


class RollingOp(BaseModel):
    """Compute a rolling window aggregation over a time-ordered column."""
    op: Literal["rolling"] = "rolling"
    column: str = Field(description="Column to aggregate")
    agg: Literal["mean", "sum", "min", "max", "std"] = Field(description="Rolling aggregation function")
    window: int = Field(description="Window size (number of rows)")
    partition_by: Optional[list[str]] = Field(default=None, description="Optional column(s) to partition by before rolling")
    order_by: str = Field(description="Column to order by (usually a date/time column)")


class DateExtractOp(BaseModel):
    """Extract a part from a datetime column (year, month, day, etc.)."""
    op: Literal["date_extract"] = "date_extract"
    column: str = Field(description="Datetime column")
    part: DatePart = Field(description="Part to extract: year, month, day, dayofweek, hour, quarter, weekofyear")


class DateDiffOp(BaseModel):
    """Compute the difference between two date columns."""
    op: Literal["date_diff"] = "date_diff"
    start_column: str = Field(description="Start date column")
    end_column: str = Field(description="End date column")
    unit: Literal["days", "months", "years"] = Field(default="days", description="Unit for the difference")


class TargetEncodeOp(BaseModel):
    """Smoothed target-mean encoding for categorical columns. Fit on train only to prevent leakage."""
    op: Literal["target_encode"] = "target_encode"
    column: str = Field(description="Categorical column to encode")
    smoothing: float = Field(default=10.0, description="Smoothing factor — higher values shrink category means toward global mean")


class FrequencyEncodeOp(BaseModel):
    """Replace categorical values with their frequency (proportion) in the dataset."""
    op: Literal["frequency_encode"] = "frequency_encode"
    column: str = Field(description="Categorical column to encode")


FormulaOp = Union[
    PassthroughOp,
    ExpressionOp,
    BinOp,
    OneHotOp,
    OrdinalOp,
    GroupAggOp,
    RollingOp,
    DateExtractOp,
    DateDiffOp,
    TargetEncodeOp,
    FrequencyEncodeOp,
]


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS
# =============================================================================

class AsOfConstraint(BaseModel):
    """Temporal constraint to prevent data leakage."""
    source_date_column: str = Field(
        description="The date/timestamp column that must be checked (e.g., 'transaction_date')"
    )
    operator: Literal["<", "<="] = Field(
        default="<",
        description="Comparison operator: '<' (strictly before) or '<=' (on or before)"
    )


class FeatureDefinition(BaseModel):
    """Definition of a single feature for the ML model."""
    name: str = Field(description="Feature name (e.g., 'credit_score', 'income_to_loan_ratio', 'region_encoded')")
    formula: FormulaOp = Field(description="The operation to compute this feature. Must be one of the structured operation types.")
    grain: str = Field(description="Grain level this feature is computed at (e.g., 'loan_id', 'customer_id')")
    as_of_constraint: Optional[AsOfConstraint] = Field(
        default=None,
        description="Temporal constraint to prevent leakage. Set if this feature uses time-sensitive data. Null if feature is static/non-temporal."
    )


class FeatureSpec(BaseModel):
    """Complete feature specification for the ML model."""
    features: list[FeatureDefinition] = Field(
        description="List of features to use in the model",
        min_length=1
    )
    reasoning: str = Field(
        description="Explanation of why these features were selected and how they relate to the prediction goal"
    )
    excluded_columns: list[str] = Field(
        default_factory=list,
        description="Columns that were intentionally excluded (forbidden, leaky, or redundant) with reasons"
    )


# =============================================================================
# VALIDATION
# =============================================================================

_VALID_OPS = {
    "passthrough", "expression", "bin", "one_hot", "ordinal",
    "group_agg", "rolling", "date_extract", "date_diff",
    "target_encode", "frequency_encode",
}


def _extract_columns_from_formula(formula: dict) -> set[str]:
    """Extract all column references from a formula operation."""
    cols = set()
    op = formula.get("op")

    if op in ("passthrough", "bin", "one_hot", "ordinal", "target_encode", "frequency_encode"):
        cols.add(formula.get("column", ""))
    elif op == "expression":
        cols.update(formula.get("source_columns", []))
    elif op == "group_agg":
        cols.add(formula.get("column", ""))
        cols.update(formula.get("group_by", []))
    elif op == "rolling":
        cols.add(formula.get("column", ""))
        cols.add(formula.get("order_by", ""))
        cols.update(formula.get("partition_by") or [])
    elif op == "date_extract":
        cols.add(formula.get("column", ""))
    elif op == "date_diff":
        cols.add(formula.get("start_column", ""))
        cols.add(formula.get("end_column", ""))

    cols.discard("")
    return cols


def validate_feature_spec(
    feature_spec: dict,
    available_columns: set[str],
    forbidden_columns: set[str] = None,
    target_column: str = None,
) -> dict[str, Any]:
    """Validate that a feature specification is executable against a dataset."""
    errors = []
    warnings = []
    features_valid = {}
    forbidden_columns = forbidden_columns or set()

    features = feature_spec.get("features", [])

    for feat in features:
        feat_name = feat.get("name", "unknown")
        feat_errors = []
        feat_warnings = []

        formula = feat.get("formula", {})
        op = formula.get("op")

        if op not in _VALID_OPS:
            feat_errors.append(f"Invalid operation: '{op}'")

        cols = _extract_columns_from_formula(formula)

        missing = cols - available_columns
        if missing:
            feat_errors.append(f"Missing columns: {missing}")

        used_forbidden = cols & forbidden_columns
        if used_forbidden:
            feat_errors.append(f"Uses forbidden columns: {used_forbidden}")

        if target_column and target_column in cols and op != "target_encode":
            feat_errors.append(f"Uses target column '{target_column}' as input (leakage)")

        as_of = feat.get("as_of_constraint")
        if as_of and isinstance(as_of, dict):
            source_col = as_of.get("source_date_column")
            if source_col and source_col not in available_columns:
                feat_warnings.append(f"as_of_constraint references missing column: '{source_col}'")

        is_valid = len(feat_errors) == 0
        features_valid[feat_name] = {
            "valid": is_valid,
            "errors": feat_errors,
            "warnings": feat_warnings,
            "columns_used": list(cols),
        }

        for err in feat_errors:
            errors.append(f"Feature '{feat_name}': {err}")
        for warn in feat_warnings:
            warnings.append(f"Feature '{feat_name}': {warn}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "features_valid": features_valid,
        "total_features": len(features),
        "valid_features": sum(1 for f in features_valid.values() if f["valid"]),
    }


# =============================================================================
# ANALYSIS RUNNER
# =============================================================================

def _extract_key_stats(analysis_results: dict[str, Any], target_column: str) -> dict[str, Any]:
    """Extract key statistics from raw analysis results for frontend display."""
    key_stats = {
        "dataset_overview": {},
        "target_analysis": {},
        "feature_correlations": [],
        "mutual_information": [],
        "correlation_matrix": {},
        "high_correlation_pairs": [],
        "distribution_stats": [],
        "numeric_summaries": [],
        "leakage_warnings": [],
        "feature_health": [],
        "categorical_summaries": [],
        "cardinality_analysis": [],
        "group_summaries": [],
        "concentration_analysis": [],
        "schema": [],
        "summary_text": "",
    }

    # 1. EDA Report stats
    eda = analysis_results.get("eda_report")
    if isinstance(eda, dict):
        shape = eda.get("shape", {})
        schema = eda.get("schema", [])
        numeric_summary = eda.get("numeric_summary", [])
        categorical_summary = eda.get("categorical_summary", [])

        if isinstance(schema, list):
            key_stats["schema"] = [
                {
                    "column": s.get("column"),
                    "dtype": s.get("dtype"),
                    "null_pct": s.get("null_pct", 0),
                    "unique": s.get("unique"),
                }
                for s in schema if isinstance(s, dict)
            ]

        num_numeric = 0
        num_categorical = 0
        target_stats = None

        if isinstance(numeric_summary, list):
            num_numeric = len(numeric_summary)
            key_stats["numeric_summaries"] = []
            for item in numeric_summary:
                if isinstance(item, dict):
                    col = item.get("column")
                    stat = {
                        "column": col,
                        "mean": round(item.get("mean", 0), 2) if item.get("mean") is not None else None,
                        "median": round(item.get("median", 0), 2) if item.get("median") is not None else None,
                        "std": round(item.get("std", 0), 2) if item.get("std") is not None else None,
                        "min": item.get("min"),
                        "max": item.get("max"),
                        "p5": item.get("p5"),
                        "p95": item.get("p95"),
                        "skew": round(item.get("skew", 0), 3) if item.get("skew") is not None else None,
                        "outliers_pct": round(item.get("outliers_pct", 0), 2) if item.get("outliers_pct") is not None else None,
                    }
                    key_stats["numeric_summaries"].append(stat)
                    if col == target_column:
                        target_stats = stat
        elif isinstance(numeric_summary, dict):
            num_numeric = len(numeric_summary)
            for col, item in numeric_summary.items():
                if isinstance(item, dict):
                    stat = {
                        "column": col,
                        "mean": round(item.get("mean", 0), 2) if item.get("mean") is not None else None,
                        "median": round(item.get("median", 0), 2) if item.get("median") is not None else None,
                        "std": round(item.get("std", 0), 2) if item.get("std") is not None else None,
                        "min": item.get("min"),
                        "max": item.get("max"),
                        "skew": round(item.get("skew", 0), 3) if item.get("skew") is not None else None,
                    }
                    key_stats["numeric_summaries"].append(stat)
                    if col == target_column:
                        target_stats = stat

        if isinstance(categorical_summary, (list, dict)):
            num_categorical = len(categorical_summary)

        key_stats["dataset_overview"] = {
            "rows": shape.get("rows") if isinstance(shape, dict) else None,
            "columns": shape.get("columns") if isinstance(shape, dict) else None,
            "numeric_columns": num_numeric,
            "categorical_columns": num_categorical,
        }

        if target_stats:
            key_stats["target_analysis"] = {"type": "numeric", **target_stats}

    # 1b. Override target analysis with dedicated result
    target_info = analysis_results.get("target_analysis")
    if isinstance(target_info, dict) and "column" in target_info:
        key_stats["target_analysis"] = target_info

    # 2. Correlation info
    corr = analysis_results.get("correlation_matrix")
    if isinstance(corr, dict):
        matrix = corr.get("matrix", {})
        columns = corr.get("columns", [])

        if matrix and columns:
            key_stats["correlation_matrix"] = {"columns": columns, "matrix": matrix}

        if isinstance(matrix, dict) and target_column in matrix:
            target_corrs = matrix[target_column]
            if isinstance(target_corrs, dict):
                sorted_corrs = sorted(
                    [(col, val) for col, val in target_corrs.items() if col != target_column and isinstance(val, (int, float))],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                key_stats["feature_correlations"] = [
                    {"feature": col, "correlation": round(val, 4)}
                    for col, val in sorted_corrs[:15]
                ]

        high_corr = corr.get("high_correlations", [])
        if high_corr and isinstance(high_corr, list):
            key_stats["high_correlation_pairs"] = [
                {
                    "feature1": pair.get("col1") or pair.get("feature1"),
                    "feature2": pair.get("col2") or pair.get("feature2"),
                    "correlation": round(pair.get("correlation", 0), 4) if isinstance(pair.get("correlation"), (int, float)) else 0,
                }
                for pair in high_corr[:10]
                if isinstance(pair, dict)
            ]

    # 3. Feature diagnostics
    diagnostics = analysis_results.get("feature_diagnostics")
    if isinstance(diagnostics, dict):
        feature_health = diagnostics.get("feature_health", [])
        if isinstance(feature_health, list):
            key_stats["feature_health"] = [
                {
                    "column": f.get("column"),
                    "null_pct": f.get("null_pct", 0),
                    "skew": round(f.get("skew", 0), 3) if f.get("skew") is not None else None,
                    "leakage_risk": f.get("leakage_risk", "none"),
                    "redundancy_group": f.get("redundancy_group"),
                }
                for f in feature_health if isinstance(f, dict)
            ]

            leakage_features = [
                f.get("column") for f in feature_health
                if isinstance(f, dict) and f.get("leakage_risk") not in [None, "none", "low"]
            ]
            if leakage_features:
                key_stats["leakage_warnings"] = leakage_features

        summary = diagnostics.get("summary", {})
        if isinstance(summary, dict):
            key_stats["diagnostics_summary"] = {
                "features_checked": summary.get("features_checked", 0),
                "drop": summary.get("drop", 0),
                "transform": summary.get("transform", 0),
                "keep_as_is": summary.get("keep_as_is", 0),
            }

    # 4. Distribution stats
    dist = analysis_results.get("distribution_analysis")
    if isinstance(dist, dict):
        distributions = dist.get("distributions", dist)
        if isinstance(distributions, dict):
            dist_stats = []
            for col, stats in distributions.items():
                if isinstance(stats, dict):
                    hist = stats.get("histogram", [])
                    histogram_data = [
                        {"range": h.get("range"), "count": h.get("count"), "pct": h.get("pct")}
                        for h in hist if isinstance(h, dict)
                    ] if isinstance(hist, list) else []

                    dist_stats.append({
                        "column": col,
                        "n": stats.get("n"),
                        "n_missing": stats.get("n_missing", 0),
                        "mean": round(stats.get("mean", 0), 2) if stats.get("mean") is not None else None,
                        "std": round(stats.get("std", 0), 2) if stats.get("std") is not None else None,
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                        "skewness": round(stats.get("skewness", 0), 3) if stats.get("skewness") is not None else None,
                        "kurtosis": round(stats.get("kurtosis", 0), 3) if stats.get("kurtosis") is not None else None,
                        "is_normal": stats.get("is_approximately_normal"),
                        "shape": stats.get("shape_interpretation"),
                        "modality": stats.get("modality", {}).get("type") if isinstance(stats.get("modality"), dict) else None,
                        "histogram": histogram_data,
                        "percentiles": stats.get("percentiles", {}),
                    })
            key_stats["distribution_stats"] = dist_stats

    # 5. Group summaries
    group = analysis_results.get("group_summary")
    if isinstance(group, dict):
        group_summaries = []
        for col, summary in group.items():
            if isinstance(summary, dict):
                groups = summary.get("groups", [])
                group_data = [
                    {
                        "value": g.get(col),
                        "count": g.get(f"{target_column}_count"),
                        "mean": round(g.get(f"{target_column}_mean", 0), 3) if g.get(f"{target_column}_mean") is not None else None,
                    }
                    for g in groups if isinstance(g, dict)
                ] if isinstance(groups, list) else []

                group_summaries.append({
                    "column": col,
                    "n_groups": summary.get("n_groups", len(group_data)),
                    "total_rows": summary.get("total_rows"),
                    "groups": group_data,
                    "overall_mean": summary.get("overall", {}).get(target_column, {}).get("mean") if isinstance(summary.get("overall"), dict) else None,
                })
        key_stats["group_summaries"] = group_summaries
        key_stats["categorical_summaries"] = [
            {"column": s["column"], "groups": [g["value"] for g in s["groups"][:10]], "group_count": s["n_groups"]}
            for s in group_summaries
        ]

    # 6. Concentration analysis
    conc = analysis_results.get("concentration_analysis")
    if isinstance(conc, dict):
        conc_stats = []
        for col, stats in conc.items():
            if isinstance(stats, dict):
                lorenz = stats.get("lorenz_curve", [])
                lorenz_data = [
                    {"pct_entities": l.get("pct_of_entities"), "pct_value": l.get("pct_of_value")}
                    for l in lorenz if isinstance(l, dict)
                ] if isinstance(lorenz, list) else []

                top_n = stats.get("top_n_contribution", {})
                conc_stats.append({
                    "column": col,
                    "gini": round(stats.get("gini_coefficient", 0), 4) if stats.get("gini_coefficient") is not None else None,
                    "gini_interpretation": stats.get("gini_interpretation"),
                    "mean": round(stats.get("mean_value", 0), 2) if stats.get("mean_value") is not None else None,
                    "median": round(stats.get("median_value", 0), 2) if stats.get("median_value") is not None else None,
                    "top_1pct_share": round(top_n.get("top_1pct", {}).get("pct_of_total", 0), 1) if isinstance(top_n.get("top_1pct"), dict) else None,
                    "top_10pct_share": round(top_n.get("top_10pct", {}).get("pct_of_total", 0), 1) if isinstance(top_n.get("top_10pct"), dict) else None,
                    "top_50pct_share": round(top_n.get("top_50pct", {}).get("pct_of_total", 0), 1) if isinstance(top_n.get("top_50pct"), dict) else None,
                    "pareto_80pct": round(stats.get("pareto", {}).get("pct_entities_for_80pct_value", 0), 1) if isinstance(stats.get("pareto"), dict) else None,
                    "lorenz_curve": lorenz_data,
                })
        key_stats["concentration_analysis"] = conc_stats

    # 7. Mutual information scores
    mi = analysis_results.get("mutual_information")
    if isinstance(mi, list):
        key_stats["mutual_information"] = mi

    # 8. Cardinality analysis
    card = analysis_results.get("cardinality_analysis")
    if isinstance(card, list):
        key_stats["cardinality_analysis"] = card

    # 9. Summary text
    summary_parts = []
    if key_stats["dataset_overview"].get("rows"):
        summary_parts.append(f"Dataset has {key_stats['dataset_overview']['rows']:,} rows and {key_stats['dataset_overview']['columns']} columns")

    ta = key_stats.get("target_analysis", {})
    if ta.get("task_type") == "classification" and ta.get("minority_class_count"):
        minority_pct = ta["minority_class_count"] / ta.get("n_rows", 1)
        summary_parts.append(
            f"Target '{ta.get('column')}': {ta['n_classes']} classes, "
            f"minority={ta['minority_class_count']:,} ({minority_pct:.1%}), "
            f"imbalance ratio={ta.get('imbalance_ratio', '?')}"
        )

    if key_stats.get("numeric_summaries"):
        summary_parts.append(f"{len(key_stats['numeric_summaries'])} numeric features analyzed")
    if key_stats.get("feature_correlations"):
        top_corr = key_stats["feature_correlations"][0]
        summary_parts.append(f"Top linear correlation: {top_corr['feature']} (r={top_corr['correlation']})")
    if isinstance(mi, list) and mi:
        top_mi = mi[0]
        summary_parts.append(f"Top mutual information: {top_mi['feature']} (MI={top_mi['mutual_information']})")
    if key_stats.get("leakage_warnings"):
        summary_parts.append(f"{len(key_stats['leakage_warnings'])} features flagged for potential leakage")
    if key_stats.get("high_correlation_pairs"):
        summary_parts.append(f"{len(key_stats['high_correlation_pairs'])} highly correlated pairs found")
    if isinstance(card, list):
        high_card = [c for c in card if c.get("n_unique", 0) > 50]
        if high_card:
            summary_parts.append(f"{len(high_card)} high-cardinality categoricals (>50 values)")

    key_stats["summary_text"] = ". ".join(summary_parts) if summary_parts else "Analysis complete"

    return key_stats


def _compute_target_analysis(df, target_column: str, task_type: str) -> dict[str, Any]:
    """Compute detailed target column analysis including class balance and EPV info."""
    target = df[target_column]
    n_rows = len(df)
    result = {"column": target_column, "task_type": task_type, "n_rows": n_rows}

    if n_rows == 0:
        return result

    if task_type == "classification":
        counts = target.value_counts().to_dict()
        proportions = target.value_counts(normalize=True).to_dict()
        n_classes = len(counts)
        minority_count = min(counts.values())
        majority_count = max(counts.values())
        imbalance_ratio = round(majority_count / minority_count, 2) if minority_count > 0 else float("inf")

        result.update({
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "class_proportions": {str(k): round(float(v), 4) for k, v in proportions.items()},
            "n_classes": n_classes,
            "minority_class_count": int(minority_count),
            "majority_class_count": int(majority_count),
            "imbalance_ratio": imbalance_ratio,
            "is_imbalanced": imbalance_ratio > 3,
        })
    else:
        result.update({
            "mean": round(float(target.mean()), 4),
            "median": round(float(target.median()), 4),
            "std": round(float(target.std()), 4),
            "min": float(target.min()),
            "max": float(target.max()),
            "skew": round(float(target.skew()), 3),
            "zero_pct": round(float((target == 0).mean()), 4),
        })

    return result


def _compute_mutual_information(df, feature_cols: list[str], target_column: str, task_type: str) -> list[dict]:
    """Compute mutual information between each feature and the target."""
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    target = df[target_column].copy()
    if target.isna().any():
        mask = target.notna()
        df = df.loc[mask]
        target = target.loc[mask]

    mi_func = mutual_info_classif if task_type == "classification" else mutual_info_regression

    if target.dtype == "object" or target.dtype.name == "category":
        le_target = LabelEncoder()
        target = pd.Series(le_target.fit_transform(target.astype(str)), index=target.index)

    X = df[feature_cols].copy()
    discrete_mask = []
    usable_cols = []

    for col in feature_cols:
        if X[col].isna().all():
            continue
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            discrete_mask.append(True)
        else:
            X[col] = X[col].fillna(X[col].median())
            discrete_mask.append(False)
        usable_cols.append(col)

    if not usable_cols:
        return []

    try:
        mi_scores = mi_func(X[usable_cols], target, discrete_features=discrete_mask, random_state=42)
    except Exception:
        return []

    results = [
        {"feature": col, "mutual_information": round(float(score), 4), "is_discrete": disc}
        for col, score, disc in zip(usable_cols, mi_scores, discrete_mask)
    ]
    results.sort(key=lambda x: x["mutual_information"], reverse=True)
    return results


def _compute_cardinality_analysis(df, cat_cols: list[str]) -> list[dict]:
    """Analyze cardinality and value distribution for categorical columns."""
    results = []
    n_rows = len(df)

    for col in cat_cols:
        series = df[col].dropna()
        n_unique = series.nunique()
        value_counts = series.value_counts()

        top_5 = [
            {"value": str(v), "count": int(c), "pct": round(c / n_rows, 4)}
            for v, c in value_counts.head(5).items()
        ]

        rare_threshold = max(10, n_rows * 0.01)
        n_rare = int((value_counts < rare_threshold).sum())
        rare_pct = round(n_rare / n_unique, 3) if n_unique > 0 else 0

        if n_unique <= 5:
            suggested_encoding = "one_hot"
        elif n_unique <= 15:
            suggested_encoding = "one_hot_or_target_encode"
        elif n_unique <= 50:
            suggested_encoding = "target_encode_or_frequency_encode"
        else:
            suggested_encoding = "target_encode_or_frequency_encode"

        results.append({
            "column": col,
            "n_unique": n_unique,
            "null_pct": round(float(df[col].isna().mean()), 4),
            "top_values": top_5,
            "n_rare_categories": n_rare,
            "rare_pct": rare_pct,
            "suggested_encoding": suggested_encoding,
        })

    results.sort(key=lambda x: x["n_unique"], reverse=True)
    return results


def _run_all_analysis(dataset_ref: str, target_column: str, task_type: str = "classification") -> dict[str, Any]:
    """Run all analysis tools on the dataset and collect results."""
    results = {}
    df = resolve_dataset(dataset_ref)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_feature_cols = [c for c in df.columns if c != target_column]
    numeric_feature_cols = [c for c in numeric_cols if c != target_column]

    n_cols = len(all_feature_cols)
    max_numeric = min(n_cols, 30)
    max_cat = min(len(cat_cols), 15)
    max_diag = min(n_cols, 25)
    max_conc = min(len(numeric_feature_cols), 5)

    print(f"[feature_analysis] Analyzing {len(numeric_cols)} numeric, {len(cat_cols)} categorical columns")

    try:
        print(f"[feature_analysis] Analyzing target column '{target_column}'...")
        results["target_analysis"] = _compute_target_analysis(df, target_column, task_type)
    except Exception as e:
        results["target_analysis"] = f"Error: {e}"

    try:
        print(f"[feature_analysis] Running EDA report...")
        results["eda_report"] = run_eda_report(dataset_ref)
    except Exception as e:
        results["eda_report"] = f"Error: {e}"

    try:
        print(f"[feature_analysis] Computing correlation matrix ({min(len(numeric_cols), max_numeric)} columns)...")
        results["correlation_matrix"] = compute_correlation_matrix(
            dataset_ref=dataset_ref,
            columns=numeric_cols[:max_numeric],
            threshold=0.5,
        )
    except Exception as e:
        results["correlation_matrix"] = f"Error: {e}"

    try:
        diag_cols = all_feature_cols[:max_diag]
        if diag_cols:
            print(f"[feature_analysis] Running feature diagnostics on {len(diag_cols)} features...")
            results["feature_diagnostics"] = run_feature_diagnostics(
                dataset_ref=dataset_ref,
                feature_cols=diag_cols,
                target_col=target_column,
                task_type=task_type,
            )
        else:
            results["feature_diagnostics"] = "No features to analyze"
    except Exception as e:
        results["feature_diagnostics"] = f"Error: {e}"

    try:
        cols_to_analyze = numeric_feature_cols[:max_numeric]
        if cols_to_analyze:
            print(f"[feature_analysis] Analyzing distributions for {len(cols_to_analyze)} columns...")
            results["distribution_analysis"] = analyze_distribution(
                dataset_ref=dataset_ref,
                columns=cols_to_analyze,
            )
        else:
            results["distribution_analysis"] = "No numeric columns to analyze"
    except Exception as e:
        results["distribution_analysis"] = f"Error: {e}"

    try:
        if cat_cols:
            group_cats = cat_cols[:max_cat]
            print(f"[feature_analysis] Computing group summaries for {len(group_cats)} categorical columns...")
            group_results = {}
            for col in group_cats:
                try:
                    summary = compute_group_summary(
                        dataset_ref=dataset_ref,
                        group_by=col,
                        metrics=[target_column] if target_column in numeric_cols else None,
                        agg_funcs=["mean", "count"],
                    )
                    group_results[col] = summary
                except Exception as e:
                    group_results[col] = f"Error: {e}"
            results["group_summary"] = group_results
        else:
            results["group_summary"] = "No categorical columns to analyze"
    except Exception as e:
        results["group_summary"] = f"Error: {e}"

    try:
        conc_cols = numeric_feature_cols[:max_conc]
        if conc_cols:
            print(f"[feature_analysis] Computing concentration for {len(conc_cols)} columns...")
        concentration_results = {}
        for col in conc_cols:
            try:
                conc = analyze_concentration(dataset_ref=dataset_ref, value_col=col)
                concentration_results[col] = conc
            except Exception as e:
                concentration_results[col] = f"Error: {e}"
        results["concentration_analysis"] = concentration_results if concentration_results else "No columns to analyze"
    except Exception as e:
        results["concentration_analysis"] = f"Error: {e}"

    try:
        mi_cols = all_feature_cols[:max_numeric]
        if mi_cols:
            print(f"[feature_analysis] Computing mutual information for {len(mi_cols)} features...")
            results["mutual_information"] = _compute_mutual_information(df, mi_cols, target_column, task_type)
        else:
            results["mutual_information"] = "No features to analyze"
    except Exception as e:
        results["mutual_information"] = f"Error: {e}"

    try:
        if cat_cols:
            print(f"[feature_analysis] Analyzing cardinality for {len(cat_cols)} categorical columns...")
            results["cardinality_analysis"] = _compute_cardinality_analysis(df, cat_cols)
        else:
            results["cardinality_analysis"] = "No categorical columns to analyze"
    except Exception as e:
        results["cardinality_analysis"] = f"Error: {e}"

    print(f"[feature_analysis] Analysis complete")
    return results


# =============================================================================
# FORMATTING
# =============================================================================

def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _format_analysis_results(results: dict[str, Any]) -> str:
    """Format analysis results into a readable string for the LLM."""
    SECTION_LIMITS = {
        "target_analysis": 2000,
        "mutual_information": 4000,
        "cardinality_analysis": 3000,
        "correlation_matrix": 5000,
        "feature_diagnostics": 5000,
        "eda_report": 5000,
    }
    DEFAULT_LIMIT = 4000

    sections = []

    for tool_name, result in results.items():
        limit = SECTION_LIMITS.get(tool_name, DEFAULT_LIMIT)
        sections.append(f"## {tool_name.replace('_', ' ').title()}")
        if isinstance(result, str):
            sections.append(result)
        elif isinstance(result, (dict, list)):
            sanitized = _sanitize_for_json(result)
            formatted = json.dumps(sanitized, indent=2, default=str)
            if len(formatted) > limit:
                item_info = f"{len(result)} items" if isinstance(result, list) else f"{len(formatted)} chars"
                formatted = formatted[:limit] + f"\n... (truncated, {item_info} total)"
            sections.append(f"```json\n{formatted}\n```")
        else:
            sections.append(str(result)[:2000])
        sections.append("")

    return "\n".join(sections)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

TaskType = Literal["classification", "regression"]


def run_feature_engineering_simple(
    train_ref: str,
    goal: str,
    target_column: str,
    grain: str,
    recomendation: Optional[str] = None,
    val_ref: Optional[str] = None,
    test_ref: Optional[str] = None,
    task_type: TaskType = "classification",
    forbidden_columns: list[str] = None,
    as_of_cutoff: Optional[str] = None,
    prediction_horizon: Optional[str] = None,
    selected_model: Optional[str] = None,
    model: str = "openai:gpt-5.1",
) -> dict[str, Any]:
    """
    Run feature engineering with all analysis upfront, then one LLM call.

    Analysis is run ONLY on training data to prevent data leakage.
    Generates a feature_spec. Use execute_feature_spec_split() to apply it.
    """
    forbidden_columns = forbidden_columns or []

    print(f"Running analysis tools on training data ({train_ref})...")
    analysis_results = _run_all_analysis(train_ref, target_column, task_type)

    print(f"Extracting key statistics for display...")
    key_stats = _extract_key_stats(analysis_results, target_column)

    analysis_context = _format_analysis_results(analysis_results)

    context_parts = [
        f"## Goal\n{goal}",
        f"## Task Type\n**{task_type.upper()}**" + (
            " (predict a class/category)" if task_type == "classification"
            else " (predict a continuous value)"
        ),
        f"## Selected Model\n**{selected_model or 'unknown'}**",
        f"## Training Dataset\n`{train_ref}`",
        f"## Target Column\n`{target_column}`",
        f"## Grain\n{grain} (what one row represents)",
    ]

    split_info = [f"- Training: `{train_ref}`"]
    if val_ref:
        split_info.append(f"- Validation: `{val_ref}`")
    if test_ref:
        split_info.append(f"- Test: `{test_ref}`")
    context_parts.append(f"## Data Split\n" + "\n".join(split_info))
    context_parts.append("*Note: Analysis above is from training data only to prevent leakage.*")

    if forbidden_columns:
        cols_str = ", ".join([f"`{c}`" for c in forbidden_columns])
        context_parts.append(f"## Forbidden Columns (DO NOT USE)\n{cols_str}")

    if as_of_cutoff:
        context_parts.append(f"## As-of Cutoff Column\n`{as_of_cutoff}`")

    if prediction_horizon:
        context_parts.append(f"## Prediction Horizon\n{prediction_horizon}")

    context_parts.append(f"## Analysis Results (Training Data)\n{analysis_context}")

    if recomendation:
        context_parts.append(f"""
## IMPORTANT: Feature Engineering Redo Request

This is a REDO of feature engineering based on feedback from the training agent.
The previous feature set did not produce satisfactory model performance.

**Recommendation from training agent:**
{recomendation}

**Action Required:**
- Carefully consider the recommendation above
- Modify your feature selection and engineering approach accordingly
- Focus on addressing the specific issues mentioned
- Generate an IMPROVED feature specification that addresses the feedback
""")

    context_parts.append("\n## Instructions\nBased on the analysis above, select features and specify how to build them using the formula DSL.")

    user_message = "\n\n".join(context_parts)

    system_prompt = get_feature_engineering_prompt(selected_model)
    print(f"Making LLM call for feature selection (model-specific guidance: {selected_model or 'generic'})...")
    llm = init_chat_model(model)
    llm_with_structure = llm.with_structured_output(FeatureSpec)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    feature_spec = llm_with_structure.invoke(messages)

    if hasattr(feature_spec, "model_dump"):
        feature_spec_dict = feature_spec.model_dump()
    else:
        feature_spec_dict = feature_spec

    validation = None
    if feature_spec_dict:
        try:
            df = resolve_dataset(train_ref)
            available_columns = set(df.columns)
        except Exception:
            available_columns = set()

        validation = validate_feature_spec(
            feature_spec=feature_spec_dict,
            available_columns=available_columns,
            forbidden_columns=set(forbidden_columns),
            target_column=target_column,
        )

    return {
        "feature_spec": feature_spec_dict,
        "validation": validation,
        "analysis_results": analysis_results,
        "key_stats": key_stats,
        "dataset_refs": {
            "train": train_ref,
            "val": val_ref,
            "test": test_ref,
        },
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_feature_engineering_simple",
    "validate_feature_spec",
    "FeatureSpec",
    "FeatureDefinition",
    "FeatureDefinition",
    "AsOfConstraint",
    "FormulaOp",
    "PassthroughOp",
    "ExpressionOp",
    "BinOp",
    "OneHotOp",
    "OrdinalOp",
    "GroupAggOp",
    "RollingOp",
    "DateExtractOp",
    "DateDiffOp",
    "TargetEncodeOp",
    "FrequencyEncodeOp",
    "_extract_columns_from_formula",
]
