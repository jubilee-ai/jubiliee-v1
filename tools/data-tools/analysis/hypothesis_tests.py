"""
Hypothesis tests — inferential statistics for group comparisons and categorical association.

Uses scipy.stats (no statsmodels required). Returns structured payloads for sidecars and key_stats.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset

from .analysis_sidecar import attach_analysis_sidecar

# Minimum cell count for chi-square reliability
_MIN_CHI2_CELL = 5
_MIN_GROUP_N = 5
_MAX_CATEGORIES = 20
_MAX_FEATURES_SCAN = 12


def _r(x: Any, d: int = 4) -> Optional[float]:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    try:
        return float(round(float(x), d))
    except (TypeError, ValueError):
        return None


def cramer_v(confusion: np.ndarray) -> float:
    """Cramér's V for r x c contingency table."""
    chi2, _, _, _ = stats.chi2_contingency(confusion, correction=False)
    n = confusion.sum()
    if n == 0:
        return 0.0
    r, c = confusion.shape
    k = min(r - 1, c - 1)
    if k <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / k))


def cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Cohen's d for two independent samples (pooled std)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return None
    m1, m2 = np.mean(a), np.mean(b)
    s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0 or not np.isfinite(pooled):
        return None
    return float((m1 - m2) / pooled)


def run_group_comparison(
    df: pd.DataFrame,
    outcome_col: str,
    group_col: str,
    test: Literal["auto", "welch_t", "mannwhitney", "anova", "kruskal"] = "auto",
) -> dict[str, Any]:
    """
    Compare numeric outcome across groups in group_col.
    2 groups: Welch t or Mann-Whitney. 3+: ANOVA or Kruskal-Wallis.
    """
    d = df[[outcome_col, group_col]].dropna()
    if len(d) < 2 * _MIN_GROUP_N:
        return {
            "error": f"Too few rows after dropna ({len(d)}). Need at least {2 * _MIN_GROUP_N}.",
        }
    groups = d.groupby(group_col, observed=True)[outcome_col]
    gnames = [str(x) for x in groups.groups.keys()]
    parts = [groups.get_group(k).values.astype(float) for k in groups.groups]
    n_groups = len(parts)
    if n_groups < 2:
        return {"error": f"Need at least 2 groups; found {n_groups} in '{group_col}'."}
    for i, p in enumerate(parts):
        if len(p) < _MIN_GROUP_N:
            return {
                "error": f"Group '{gnames[i]}' has n={len(p)} (minimum {_MIN_GROUP_N} per group).",
            }

    result: dict[str, Any] = {
        "outcome_col": outcome_col,
        "group_col": group_col,
        "n_groups": n_groups,
        "group_sizes": {str(g): int(len(parts[i])) for i, g in enumerate(gnames)},
    }

    if n_groups == 2:
        a, b = parts[0], parts[1]
        d_est = cohens_d(a, b)
        if test == "mannwhitney":
            u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
            result.update(
                {
                    "test": "mann_whitney",
                    "statistic": _r(u, 4),
                    "p_value": _r(p_mw, 6),
                    "cohens_d": _r(d_est, 4) if d_est is not None else None,
                    "method_note": "Non-parametric; robust to non-normality.",
                }
            )
        elif test == "welch_t":
            t, p_t = stats.ttest_ind(a, b, equal_var=False)
            result.update(
                {
                    "test": "welch_t",
                    "statistic": _r(t, 4),
                    "p_value": _r(p_t, 6),
                    "cohens_d": _r(d_est, 4) if d_est is not None else None,
                    "method_note": "Welch's t-test (unequal variances).",
                }
            )
        else:
            # auto: skew-based choice
            skew_a, skew_b = float(stats.skew(a)), float(stats.skew(b))
            if abs(skew_a) > 1.0 or abs(skew_b) > 1.0:
                u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
                result.update(
                    {
                        "test": "mann_whitney",
                        "statistic": _r(u, 4),
                        "p_value": _r(p_mw, 6),
                        "cohens_d": _r(d_est, 4) if d_est is not None else None,
                        "method_note": "auto: Mann-Whitney (high skew in one or both groups).",
                    }
                )
            else:
                t, p_t = stats.ttest_ind(a, b, equal_var=False)
                result.update(
                    {
                        "test": "welch_t",
                        "statistic": _r(t, 4),
                        "p_value": _r(p_t, 6),
                        "cohens_d": _r(d_est, 4) if d_est is not None else None,
                        "method_note": "auto: Welch t-test (moderate skew).",
                    }
                )
    else:
        if test == "kruskal":
            h, p_k = stats.kruskal(*parts)
            result.update(
                {
                    "test": "kruskal_wallis",
                    "statistic": _r(h, 4),
                    "p_value": _r(p_k, 6),
                    "method_note": "Kruskal-Wallis (non-parametric k-group).",
                }
            )
        elif test == "anova":
            f, p_f = stats.f_oneway(*parts)
            result.update(
                {
                    "test": "one_way_anova",
                    "statistic": _r(f, 4),
                    "p_value": _r(p_f, 6),
                    "method_note": "One-way ANOVA (independent groups).",
                }
            )
        else:
            f, p_f = stats.f_oneway(*parts)
            result.update(
                {
                    "test": "one_way_anova",
                    "statistic": _r(f, 4),
                    "p_value": _r(p_f, 6),
                    "method_note": "auto: one-way ANOVA for 3+ groups.",
                }
            )

    return result


def run_categorical_association(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict[str, Any]:
    """
    Test association between two categorical columns (e.g. target vs feature).
    Chi-square with Cramér's V; Fisher exact for 2x2 when expected counts are small.
    """
    d = df[[col_a, col_b]].dropna()
    if len(d) < 10:
        return {"error": f"Too few rows after dropna ({len(d)})."}
    # Limit categories
    if d[col_a].nunique() > _MAX_CATEGORIES or d[col_b].nunique() > _MAX_CATEGORIES:
        return {
            "error": f"Too many categories (max {_MAX_CATEGORIES}). Group or filter levels first.",
        }
    ct = pd.crosstab(d[col_a], d[col_b])
    table = ct.values.astype(int)
    if table.size == 0:
        return {"error": "Empty contingency table."}

    # Chi-square
    chi2, p_chi, dof, expected = stats.chi2_contingency(table, correction=False)
    min_exp = float(expected.min()) if expected.size else 0.0
    v = cramer_v(table)

    out: dict[str, Any] = {
        "col_a": col_a,
        "col_b": col_b,
        "n": int(len(d)),
        "table_shape": [int(table.shape[0]), int(table.shape[1])],
        "chi2": _r(chi2, 4),
        "p_value": _r(p_chi, 6),
        "degrees_of_freedom": int(dof),
        "cramers_v": _r(v, 4),
        "min_expected_cell": _r(min_exp, 4),
    }

    if table.shape == (2, 2) and min_exp < _MIN_CHI2_CELL:
        oddsratio, p_f = stats.fisher_exact(table)
        out["fisher_exact_p"] = _r(p_f, 6)
        out["odds_ratio"] = _r(oddsratio, 4) if np.isfinite(oddsratio) else None
        out["method_note"] = "2x2 table with small expected counts: also report Fisher exact p."
    else:
        if min_exp < _MIN_CHI2_CELL:
            out["warning"] = "Some expected cell counts are small; interpret chi-square with caution."
        out["method_note"] = "Chi-square test of independence with Cramér's V effect size."

    return out


def run_numeric_binary_comparison(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
) -> dict[str, Any]:
    """Binary target: compare numeric feature between the two classes."""
    d = df[[feature_col, target_col]].dropna()
    if d[target_col].nunique() != 2:
        return {"error": "Target must be binary (exactly 2 values)."}
    classes = sorted(d[target_col].unique().tolist(), key=str)
    m0 = d[d[target_col] == classes[0]][feature_col].values.astype(float)
    m1 = d[d[target_col] == classes[1]][feature_col].values.astype(float)
    if len(m0) < _MIN_GROUP_N or len(m1) < _MIN_GROUP_N:
        return {
            "error": f"Insufficient samples per class (n0={len(m0)}, n1={len(m1)}).",
        }
    t, p_t = stats.ttest_ind(m0, m1, equal_var=False)
    u, p_mw = stats.mannwhitneyu(m0, m1, alternative="two-sided")
    d_cohen = cohens_d(m0, m1)
    return {
        "feature": feature_col,
        "target": target_col,
        "class_0": str(classes[0]),
        "class_1": str(classes[1]),
        "n0": len(m0),
        "n1": len(m1),
        "welch_t_p": _r(p_t, 6),
        "mann_whitney_p": _r(p_mw, 6),
        "cohens_d": _r(d_cohen, 4) if d_cohen is not None else None,
        "primary_p": _r(p_t, 6),
        "method_note": "Binary target: Welch t p (primary) and Mann-Whitney for robustness.",
    }


def run_inferential_feature_scan(
    dataset_ref: str,
    target_column: str,
    task_type: str = "classification",
    max_categorical: int = 8,
    max_numeric: int = 6,
) -> dict[str, Any]:
    """
    Deterministic batch: test strongest categorical and numeric relationships with target.
    For use in training pipeline key_stats (not a LangChain tool).
    """
    df = resolve_dataset(dataset_ref)
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not in dataset."}

    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns.tolist()
        if c != target_column
    ]
    num_cols = [
        c
        for c in df.select_dtypes(include=["number"]).columns.tolist()
        if c != target_column
    ]

    out: dict[str, Any] = {
        "target_column": target_column,
        "task_type": task_type,
        "categorical_tests": [],
        "numeric_tests": [],
        "notes": [],
    }

    if task_type == "classification":
        # Categorical vs target: chi2 / fisher (any finite # of target classes)
        for col in cat_cols[:max_categorical]:
            if df[col].nunique() < 2:
                continue
            r = run_categorical_association(df, target_column, col)
            if "error" not in r:
                r["feature"] = col
                out["categorical_tests"].append(r)
        if df[target_column].nunique() == 2:
            for col in num_cols[:max_numeric]:
                r = run_numeric_binary_comparison(df, col, target_column)
                if "error" not in r:
                    out["numeric_tests"].append(r)
        elif num_cols:
            out["notes"].append(
                "Numeric vs-target tests skipped: only implemented for binary classification."
            )
    elif task_type == "regression":
        # Numeric target: ANOVA / Kruskal for each categorical
        for col in cat_cols[:max_categorical]:
            if df[col].nunique() < 2 or df[col].nunique() > _MAX_CATEGORIES:
                continue
            r = run_group_comparison(df, target_column, col, test="auto")
            if "error" not in r:
                r["feature"] = col
                out["categorical_tests"].append(r)
    else:
        out["notes"].append("Scan skipped or limited: use compare_groups_test_tool for custom tests.")

    # Sort by p-value when present
    def pkey(d: dict) -> float:
        p = d.get("p_value") or d.get("primary_p")
        return float(p) if p is not None else 1.0

    out["categorical_tests"].sort(key=pkey)
    out["numeric_tests"].sort(key=pkey)
    return out


# ---------------------------------------------------------------------------
# LangChain tools
# ---------------------------------------------------------------------------


class GroupComparisonInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference id")
    outcome_col: str = Field(description="Numeric outcome column to compare across groups")
    group_col: str = Field(description="Categorical column defining groups")
    test: Literal["auto", "welch_t", "mannwhitney", "anova", "kruskal"] = Field(
        default="auto",
        description="Statistical test to use; auto picks based on number of groups and skew",
    )


@tool(args_schema=GroupComparisonInput)
def group_comparison_test_tool(
    dataset_ref: str,
    outcome_col: str,
    group_col: str,
    test: str = "auto",
) -> str:
    """
    Compare a numeric column across groups (inferential test).

    Use for: "Is mean revenue different by region?", "Do segments differ on this metric?".
    For two groups: Welch t-test or Mann-Whitney. For 3+ groups: ANOVA or Kruskal-Wallis.
    Reports p-values and effect sizes (Cohen's d for two groups).
    """
    try:
        df = resolve_dataset(dataset_ref)
        for c in (outcome_col, group_col):
            if c not in df.columns:
                return f"✗ Column not found: {c}"
        if not pd.api.types.is_numeric_dtype(df[outcome_col]):
            return f"✗ outcome_col '{outcome_col}' must be numeric."
        res = run_group_comparison(df, outcome_col, group_col, test=test)  # type: ignore[arg-type]
        if "error" in res:
            return f"✗ {res['error']}"
        lines = [
            "# Group comparison (inferential)",
            f"**Outcome:** `{outcome_col}`  **Groups:** `{group_col}`",
            f"**Test:** {res.get('test', 'n/a')}",
        ]
        if "p_value" in res:
            lines.append(f"**p-value:** {res.get('p_value')}")
        if res.get("cohens_d") is not None:
            lines.append(f"**Cohen's d:** {res.get('cohens_d')}")
        if res.get("method_note"):
            lines.append(f"*{res['method_note']}*")
        md = "\n".join(lines)
        p = res.get("p_value")
        summary = f"{res.get('test', 'test')}; p={p if p is not None else 'n/a'}"
        return attach_analysis_sidecar(
            md,
            kind="inferential_group",
            tool="group_comparison_test_tool",
            summary=summary,
            payload=res,
        )
    except Exception as e:
        return f"✗ group_comparison_test_tool failed: {type(e).__name__}: {e}"


class CategoricalAssociationInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference id")
    col_a: str = Field(
        description="First categorical column (e.g. target or segment)"
    )
    col_b: str = Field(
        description="Second categorical column (e.g. feature or another segment)"
    )


@tool(args_schema=CategoricalAssociationInput)
def categorical_association_test_tool(
    dataset_ref: str,
    col_a: str,
    col_b: str,
) -> str:
    """
    Test association between two categorical variables (chi-square, Cramér's V; Fisher for 2x2 when appropriate).

    Use for: "Is this feature associated with the target class?", "Independence of two flags".
    """
    try:
        df = resolve_dataset(dataset_ref)
        for c in (col_a, col_b):
            if c not in df.columns:
                return f"✗ Column not found: {c}"
        res = run_categorical_association(df, col_a, col_b)
        if "error" in res:
            return f"✗ {res['error']}"
        lines = [
            "# Categorical association",
            f"**Columns:** `{col_a}` × `{col_b}`",
            f"**Chi-square:** {res.get('chi2')}, **df** {res.get('degrees_of_freedom')}, **p** {res.get('p_value')}",
            f"**Cramér's V:** {res.get('cramers_v')}",
        ]
        if res.get("fisher_exact_p") is not None:
            lines.append(f"**Fisher exact p (2x2):** {res.get('fisher_exact_p')}")
        if res.get("warning"):
            lines.append(f"**Warning:** {res['warning']}")
        md = "\n".join(lines)
        summary = f"chi2 p={res.get('p_value')}, Cramér V={res.get('cramers_v')}"
        return attach_analysis_sidecar(
            md,
            kind="inferential_categorical",
            tool="categorical_association_test_tool",
            summary=summary,
            payload=res,
        )
    except Exception as e:
        return f"✗ categorical_association_test_tool failed: {type(e).__name__}: {e}"


hypothesis_tools = [group_comparison_test_tool, categorical_association_test_tool]

__all__ = [
    "group_comparison_test_tool",
    "categorical_association_test_tool",
    "run_group_comparison",
    "run_categorical_association",
    "run_numeric_binary_comparison",
    "run_inferential_feature_scan",
    "cramer_v",
    "cohens_d",
    "hypothesis_tools",
]
