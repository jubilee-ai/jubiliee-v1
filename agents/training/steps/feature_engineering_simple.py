"""
Simple Feature Engineering: Run all analysis tools, then one LLM call.

Instead of an agent loop that decides which tools to call, this version:
1. Runs ALL analysis tools upfront (in parallel conceptually, sequentially in practice)
2. Compiles all results into a single context
3. Makes ONE LLM call to select and specify features

Same input/output format as feature_engineering.py, but simpler and more deterministic.
"""

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Literal, Optional, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ..utils.prompts import get_feature_engineering_prompt

load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

# Add data-tools to path
# Path: steps -> training -> agents -> root -> tools/data-tools
_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from analysis import (analyze_concentration, analyze_distribution,
                      compute_correlation_matrix, compute_group_summary,
                      run_eda_report, run_feature_diagnostics,
                      run_inferential_feature_scan)
from transformations.tool_utils import resolve_dataset

from .feature_engineering import (AsOfConstraint, FeatureDefinition,
                                  FeatureSpec, FormulaOp,
                                  _extract_columns_from_formula,
                                  validate_feature_spec)

# =============================================================================
# IMPORT SHARED TYPES FROM MAIN MODULE
# =============================================================================


# =============================================================================
# ANALYSIS RUNNER
# =============================================================================

def _extract_key_stats(analysis_results: dict[str, Any], target_column: str) -> dict[str, Any]:
    """
    Extract key statistics from raw analysis results for display.
    Returns a structured dict optimized for frontend display.
    
    Handles the actual data formats returned by the analysis tools:
    - EDA Report: numeric_summary is a LIST of dicts, not a dict
    - Correlation Matrix: has 'matrix' key with nested correlations
    - Distribution Analysis: has 'distributions' key with nested data
    - Concentration Analysis: has gini_coefficient, lorenz_curve, etc.
    """
    key_stats = {
        "dataset_overview": {},
        "target_analysis": {},
        "feature_correlations": [],
        "mutual_information": [],  # Non-linear feature-target association
        "correlation_matrix": {},  # Full matrix for heatmap
        "high_correlation_pairs": [],
        "distribution_stats": [],
        "numeric_summaries": [],  # Detailed numeric column stats
        "leakage_warnings": [],
        "feature_health": [],  # From feature diagnostics
        "categorical_summaries": [],
        "cardinality_analysis": [],  # Categorical encoding guidance
        "group_summaries": [],  # Detailed group summaries with default rates
        "concentration_analysis": [],
        "schema": [],  # Column schema
        "statistical_tests": [],  # Inferential: chi2, t-tests, ANOVA-style
        "effect_sizes": [],  # Cramér's V, Cohen's d where available
        "summary_text": "",
    }
    
    # 1. Extract EDA Report stats
    eda = analysis_results.get("eda_report")
    if isinstance(eda, dict):
        shape = eda.get("shape", {})
        schema = eda.get("schema", [])
        numeric_summary = eda.get("numeric_summary", [])
        categorical_summary = eda.get("categorical_summary", [])
        
        # Schema is a list of column info
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
        
        # Numeric summary is a LIST of dicts: [{column: "Age", mean: 43.51, ...}, ...]
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
        
        if isinstance(categorical_summary, list):
            num_categorical = len(categorical_summary)
        elif isinstance(categorical_summary, dict):
            num_categorical = len(categorical_summary)
        
        key_stats["dataset_overview"] = {
            "rows": shape.get("rows") if isinstance(shape, dict) else None,
            "columns": shape.get("columns") if isinstance(shape, dict) else None,
            "numeric_columns": num_numeric,
            "categorical_columns": num_categorical,
        }
        
        # Target column analysis from EDA (fallback)
        if target_stats:
            key_stats["target_analysis"] = {
                "type": "numeric",
                **target_stats
            }
    
    # 1b. Override target analysis with dedicated result (has class balance info)
    target_info = analysis_results.get("target_analysis")
    if isinstance(target_info, dict) and "column" in target_info:
        key_stats["target_analysis"] = target_info
    
    # 2. Extract correlation info from matrix
    corr = analysis_results.get("correlation_matrix")
    if isinstance(corr, dict):
        matrix = corr.get("matrix", {})
        columns = corr.get("columns", [])
        
        # Store full matrix for heatmap
        if matrix and columns:
            key_stats["correlation_matrix"] = {
                "columns": columns,
                "matrix": matrix,
            }
        
        # Get correlations with target from the matrix
        if isinstance(matrix, dict) and target_column in matrix:
            target_corrs = matrix[target_column]
            if isinstance(target_corrs, dict):
                sorted_corrs = sorted(
                    [(col, val) for col, val in target_corrs.items() if col != target_column and isinstance(val, (int, float))],
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                key_stats["feature_correlations"] = [
                    {"feature": col, "correlation": round(val, 4)}
                    for col, val in sorted_corrs[:15]
                ]
        
        # Get high correlation pairs (if available)
        high_corr = corr.get("high_correlations", [])
        if high_corr and isinstance(high_corr, list):
            pairs = []
            for pair in high_corr[:10]:
                if isinstance(pair, dict):
                    pairs.append({
                        "feature1": pair.get("col1") or pair.get("feature1"),
                        "feature2": pair.get("col2") or pair.get("feature2"),
                        "correlation": round(pair.get("correlation", 0), 4) if isinstance(pair.get("correlation"), (int, float)) else 0
                    })
            key_stats["high_correlation_pairs"] = pairs
    
    # 3. Extract feature diagnostics
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
        
        # Extract leakage warnings
        leakage_features = [f.get("column") for f in feature_health 
                           if isinstance(f, dict) and f.get("leakage_risk") not in [None, "none", "low"]]
        if leakage_features:
            key_stats["leakage_warnings"] = leakage_features
        
        # Summary info
        summary = diagnostics.get("summary", {})
        if isinstance(summary, dict):
            key_stats["diagnostics_summary"] = {
                "features_checked": summary.get("features_checked", 0),
                "drop": summary.get("drop", 0),
                "transform": summary.get("transform", 0),
                "keep_as_is": summary.get("keep_as_is", 0),
            }
    
    # 4. Extract distribution stats - note the 'distributions' nested key
    dist = analysis_results.get("distribution_analysis")
    if isinstance(dist, dict):
        distributions = dist.get("distributions", dist)  # May be nested under 'distributions'
        columns_analyzed = dist.get("columns_analyzed", [])
        
        if isinstance(distributions, dict):
            dist_stats = []
            for col, stats in distributions.items():
                if isinstance(stats, dict):
                    hist = stats.get("histogram", [])
                    histogram_data = []
                    if isinstance(hist, list):
                        histogram_data = [
                            {"range": h.get("range"), "count": h.get("count"), "pct": h.get("pct")}
                            for h in hist if isinstance(h, dict)
                        ]
                    
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
    
    # 5. Extract group summaries - detailed with default rates per group
    group = analysis_results.get("group_summary")
    if isinstance(group, dict):
        group_summaries = []
        for col, summary in group.items():
            if isinstance(summary, dict):
                groups = summary.get("groups", [])
                group_data = []
                if isinstance(groups, list):
                    for g in groups:
                        if isinstance(g, dict):
                            group_data.append({
                                "value": g.get(col),
                                "count": g.get(f"{target_column}_count"),
                                "mean": round(g.get(f"{target_column}_mean", 0), 3) if g.get(f"{target_column}_mean") is not None else None,
                            })
                
                group_summaries.append({
                    "column": col,
                    "n_groups": summary.get("n_groups", len(group_data)),
                    "total_rows": summary.get("total_rows"),
                    "groups": group_data,
                    "overall_mean": summary.get("overall", {}).get(target_column, {}).get("mean") if isinstance(summary.get("overall"), dict) else None,
                })
        key_stats["group_summaries"] = group_summaries
        
        # Simplified categorical summaries for backward compat
        key_stats["categorical_summaries"] = [
            {
                "column": s["column"],
                "groups": [g["value"] for g in s["groups"][:10]],
                "group_count": s["n_groups"],
            }
            for s in group_summaries
        ]
    
    # 6. Extract concentration analysis with full details
    conc = analysis_results.get("concentration_analysis")
    if isinstance(conc, dict):
        conc_stats = []
        for col, stats in conc.items():
            if isinstance(stats, dict):
                lorenz = stats.get("lorenz_curve", [])
                lorenz_data = []
                if isinstance(lorenz, list):
                    lorenz_data = [
                        {"pct_entities": l.get("pct_of_entities"), "pct_value": l.get("pct_of_value")}
                        for l in lorenz if isinstance(l, dict)
                    ]
                
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
    
    # 7. Extract mutual information scores
    mi = analysis_results.get("mutual_information")
    if isinstance(mi, list):
        key_stats["mutual_information"] = mi
    
    # 8. Extract cardinality analysis
    card = analysis_results.get("cardinality_analysis")
    if isinstance(card, list):
        key_stats["cardinality_analysis"] = card

    # 8b. Inferential scan (chi-square / t-tests / ANOVA-style)
    inf = analysis_results.get("inferential_scan")
    if isinstance(inf, dict) and "error" not in inf:
        cat_tests = inf.get("categorical_tests") or []
        num_tests = inf.get("numeric_tests") or []
        stat_rows = []
        eff_rows = []
        for t in cat_tests[:25]:
            if not isinstance(t, dict):
                continue
            feat = t.get("feature") or t.get("col_b")
            if t.get("cramers_v") is not None or t.get("chi2") is not None:
                stat_rows.append(
                    {
                        "feature": feat,
                        "test": "categorical_association",
                        "p_value": t.get("p_value"),
                        "cramers_v": t.get("cramers_v"),
                        "fisher_p": t.get("fisher_exact_p"),
                        "note": t.get("method_note") or t.get("warning"),
                    }
                )
                if t.get("cramers_v") is not None:
                    eff_rows.append(
                        {
                            "feature": feat,
                            "metric": "cramers_v",
                            "value": round(float(t["cramers_v"]), 4),
                        }
                    )
            elif t.get("p_value") is not None and t.get("test"):
                stat_rows.append(
                    {
                        "feature": feat or t.get("group_col"),
                        "test": t.get("test"),
                        "p_value": t.get("p_value"),
                        "note": t.get("method_note"),
                    }
                )
        for t in num_tests[:20]:
            if not isinstance(t, dict):
                continue
            p_main = t.get("primary_p")
            if p_main is None:
                p_main = t.get("welch_t_p")
            stat_rows.append(
                {
                    "feature": t.get("feature"),
                    "test": t.get("test", "binary_numeric_comparison"),
                    "p_value": p_main,
                    "cohens_d": t.get("cohens_d"),
                    "note": t.get("method_note"),
                }
            )
            if t.get("cohens_d") is not None:
                eff_rows.append(
                    {
                        "feature": t.get("feature"),
                        "metric": "cohens_d",
                        "value": round(float(t["cohens_d"]), 4),
                    }
                )
        key_stats["statistical_tests"] = stat_rows[:40]
        key_stats["effect_sizes"] = eff_rows[:30]
    
    # 9. Generate summary text
    summary_parts = []
    if key_stats["dataset_overview"].get("rows"):
        summary_parts.append(f"Dataset has {key_stats['dataset_overview']['rows']:,} rows and {key_stats['dataset_overview']['columns']} columns")
    
    # Target class balance summary
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
        summary_parts.append(f"⚠️ {len(key_stats['leakage_warnings'])} features flagged for potential leakage")
    if key_stats.get("high_correlation_pairs"):
        summary_parts.append(f"{len(key_stats['high_correlation_pairs'])} highly correlated pairs found")
    if isinstance(card, list):
        high_card = [c for c in card if c.get("n_unique", 0) > 50]
        if high_card:
            summary_parts.append(f"{len(high_card)} high-cardinality categoricals (>50 values)")

    if key_stats.get("statistical_tests"):
        n_st = len(key_stats["statistical_tests"])
        sig = [
            s
            for s in key_stats["statistical_tests"]
            if s.get("p_value") is not None and float(s["p_value"]) < 0.05
        ]
        summary_parts.append(
            f"Inferential tests: {n_st} run, {len(sig)} with p<0.05 (exploratory; not adjusted for multiple comparisons)"
        )
    
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
    """
    Compute mutual information between each feature and the target.
    Works for both numeric and categorical features, captures non-linear associations.
    """
    from sklearn.feature_selection import (mutual_info_classif,
                                           mutual_info_regression)
    from sklearn.preprocessing import LabelEncoder

    target = df[target_column].copy()
    if target.isna().any():
        mask = target.notna()
        df = df.loc[mask]
        target = target.loc[mask]

    mi_func = mutual_info_classif if task_type == "classification" else mutual_info_regression

    if target.dtype == "object" or target.dtype.name == "category":
        import pandas as pd
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
            suggested_encoding = "one_hot_or_ordinal"
        elif n_unique <= 50:
            suggested_encoding = "ordinal_or_group_agg"
        else:
            suggested_encoding = "group_agg"

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
    """
    Run all analysis tools on the dataset and collect results.
    
    Returns a dict with results from each tool (or error message if failed).
    """
    results = {}
    df = resolve_dataset(dataset_ref)
    
    # Get column lists
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    all_feature_cols = [c for c in df.columns if c != target_column]
    numeric_feature_cols = [c for c in numeric_cols if c != target_column]
    
    # Adaptive limits based on dataset size — analyze more columns for smaller datasets
    n_cols = len(all_feature_cols)
    max_numeric = min(n_cols, 30)
    max_cat = min(len(cat_cols), 15)
    max_diag = min(n_cols, 25)
    max_conc = min(len(numeric_feature_cols), 5)
    
    print(f"[feature_analysis] Analyzing {len(numeric_cols)} numeric, {len(cat_cols)} categorical columns")
    
    # 0. TARGET ANALYSIS — class balance, imbalance ratio, EPV context
    try:
        print(f"[feature_analysis] Analyzing target column '{target_column}'...")
        results["target_analysis"] = _compute_target_analysis(df, target_column, task_type)
    except Exception as e:
        results["target_analysis"] = f"Error: {e}"
    
    # 1. EDA Report - always run
    try:
        print(f"[feature_analysis] Running EDA report...")
        results["eda_report"] = run_eda_report(dataset_ref)
    except Exception as e:
        results["eda_report"] = f"Error: {e}"
    
    # 2. Correlation Matrix (adaptive limit)
    try:
        print(f"[feature_analysis] Computing correlation matrix ({min(len(numeric_cols), max_numeric)} columns)...")
        results["correlation_matrix"] = compute_correlation_matrix(
            dataset_ref=dataset_ref,
            columns=numeric_cols[:max_numeric],
            threshold=0.5,
        )
    except Exception as e:
        results["correlation_matrix"] = f"Error: {e}"
    
    # 3. Feature Diagnostics - leakage, redundancy, skew (adaptive limit)
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
    
    # 4. Distribution Analysis - for numeric feature columns
    try:
        cols_to_analyze = [c for c in numeric_feature_cols[:max_numeric]]
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
    
    # 5. Group Summary - for categorical columns (adaptive limit)
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
    
    # 6. Concentration Analysis
    try:
        conc_cols = numeric_feature_cols[:max_conc]
        if conc_cols:
            print(f"[feature_analysis] Computing concentration for {len(conc_cols)} columns...")
        concentration_results = {}
        for col in conc_cols:
            try:
                conc = analyze_concentration(
                    dataset_ref=dataset_ref,
                    value_col=col,
                )
                concentration_results[col] = conc
            except Exception as e:
                concentration_results[col] = f"Error: {e}"
        results["concentration_analysis"] = concentration_results if concentration_results else "No columns to analyze"
    except Exception as e:
        results["concentration_analysis"] = f"Error: {e}"
    
    # 7. MUTUAL INFORMATION — non-linear feature-target association
    try:
        mi_cols = all_feature_cols[:max_numeric]
        if mi_cols:
            print(f"[feature_analysis] Computing mutual information for {len(mi_cols)} features...")
            results["mutual_information"] = _compute_mutual_information(df, mi_cols, target_column, task_type)
        else:
            results["mutual_information"] = "No features to analyze"
    except Exception as e:
        results["mutual_information"] = f"Error: {e}"
    
    # 8. CARDINALITY ANALYSIS — encoding strategy input
    try:
        if cat_cols:
            print(f"[feature_analysis] Analyzing cardinality for {len(cat_cols)} categorical columns...")
            results["cardinality_analysis"] = _compute_cardinality_analysis(df, cat_cols)
        else:
            results["cardinality_analysis"] = "No categorical columns to analyze"
    except Exception as e:
        results["cardinality_analysis"] = f"Error: {e}"

    # 9. INFERENTIAL STATISTICS — p-values / effect sizes for key feature–target relationships
    try:
        print(f"[feature_analysis] Running inferential feature scan (task_type={task_type})...")
        results["inferential_scan"] = run_inferential_feature_scan(
            dataset_ref,
            target_column,
            task_type=task_type,
            max_categorical=8,
            max_numeric=6,
        )
    except Exception as e:
        results["inferential_scan"] = f"Error: {e}"
    
    print(f"[feature_analysis] Analysis complete")
    
    return results


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf floats with None so json.dumps produces valid JSON."""
    import math
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _format_analysis_results(results: dict[str, Any]) -> str:
    """Format analysis results into a readable string for the LLM.
    
    Uses per-section limits that prioritize high-value analyses (target, MI,
    correlations, diagnostics) over less critical ones (concentration, distribution).
    """
    SECTION_LIMITS = {
        "target_analysis": 2000,
        "mutual_information": 4000,
        "cardinality_analysis": 3000,
        "correlation_matrix": 5000,
        "feature_diagnostics": 5000,
        "eda_report": 5000,
        "inferential_scan": 6000,
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
    recomendation: Optional[str] = None, # This is if we need to go back to the feature engineering to regenerate the features from training
    val_ref: Optional[str] = None,
    test_ref: Optional[str] = None,
    task_type: TaskType = "classification",
    forbidden_columns: list[str] = None,
    as_of_cutoff: Optional[str] = None,
    prediction_horizon: Optional[str] = None,
    selected_model: Optional[str] = None,
    model: str = "openai:gpt-5.4",
) -> dict[str, Any]:
    """
    Run feature engineering with all analysis upfront, then one LLM call.
    
    IMPORTANT: Analysis is run ONLY on training data to prevent data leakage.
    The feature spec generated is based solely on training data statistics.
    
    This function generates the feature_spec. Use execute_feature_spec_split()
    to apply the spec to train/val/test sets (fitting on train, transforming all).
    
    Args:
        train_ref: Reference to the TRAINING dataset (analysis runs on this only)
        goal: Description of the ML goal
        target_column: The target column for prediction
        grain: What one row represents (e.g., policy_id)
        val_ref: Optional reference to validation dataset (for info only, not analyzed)
        test_ref: Optional reference to test dataset (for info only, not analyzed)
        task_type: Either "classification" or "regression"
        forbidden_columns: Columns not available at prediction time
        as_of_cutoff: Column representing the observation timestamp
        prediction_horizon: How far into the future we're predicting
        selected_model: ML model type (e.g., "logistic_regression", "xgboost") —
            used to tailor feature engineering strategy to the model's strengths
        model: Model to use for the LLM call
    
    Returns:
        Dict with:
        - feature_spec: The feature specification
        - validation: Validation results
        - analysis_results: Raw analysis tool outputs (for debugging)
        - dataset_refs: {"train": train_ref, "val": val_ref, "test": test_ref}
    """
    forbidden_columns = forbidden_columns or []
    
    # Step 1: Run all analysis tools ON TRAINING DATA ONLY
    print(f"Running analysis tools on training data ({train_ref})...")
    analysis_results = _run_all_analysis(train_ref, target_column, task_type)
    
    # Step 1.5: Extract key statistics for frontend display
    print(f"Extracting key statistics for display...")
    key_stats = _extract_key_stats(analysis_results, target_column)
    
    # Step 2: Format results into context
    analysis_context = _format_analysis_results(analysis_results)
    
    # Step 3: Build the prompt
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
    
    # Include info about split structure
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
    
    # Include recommendation from training agent if this is a redo iteration
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
    
    # Step 4: Make ONE LLM call with structured output
    system_prompt = get_feature_engineering_prompt(selected_model)
    print(f"Making LLM call for feature selection (model-specific guidance: {selected_model or 'generic'})...")
    llm = init_chat_model(model)
    llm_with_structure = llm.with_structured_output(FeatureSpec)
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    
    feature_spec = llm_with_structure.invoke(messages)
    
    # Convert to dict
    if hasattr(feature_spec, "model_dump"):
        feature_spec_dict = feature_spec.model_dump()
    else:
        feature_spec_dict = feature_spec
    
    # Step 5: Validate against training data columns
    validation = None
    if feature_spec_dict:
        try:
            df = resolve_dataset(train_ref)
            available_columns = set(df.columns)
        except:
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
        "key_stats": key_stats,  # Extracted statistics for frontend display
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
]
