"""
Statistical models — OLS and logistic regression summaries (statsmodels).

Optional dependency: statsmodels. If missing, tools return a clear error.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset

from .analysis_sidecar import attach_analysis_sidecar


def _try_import_statsmodels():
    try:
        import statsmodels.api as sm  # noqa: F401
        return sm
    except ImportError:
        return None


def run_ols_summary(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    sm = _try_import_statsmodels()
    if sm is None:
        return {"error": "statsmodels is not installed. Add `statsmodels` to the environment."}

    cols = [target_col] + list(feature_cols)
    d = df[cols].dropna()
    if len(d) < max(20, 2 * len(feature_cols) + 5):
        return {"error": f"Too few complete rows ({len(d)}) for stable regression."}

    y = d[target_col].astype(float)
    X = d[feature_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    ols = sm.OLS(y, X).fit()

    params = ols.params.to_dict()
    pvalues = ols.pvalues.to_dict()
    bse = ols.bse.to_dict()
    conf = ols.conf_int()
    coef_rows = []
    for name in X.columns:
        lo, hi = conf.loc[name, 0], conf.loc[name, 1]
        coef_rows.append(
            {
                "name": str(name),
                "coef": float(ols.params[name]),
                "std_err": float(ols.bse[name]) if name in bse else None,
                "p_value": float(ols.pvalues[name]) if name in pvalues else None,
                "ci_lower": float(lo),
                "ci_upper": float(hi),
            }
        )

    return {
        "model": "ols",
        "target": target_col,
        "n_obs": int(ols.nobs),
        "r_squared": float(ols.rsquared),
        "adj_r_squared": float(ols.rsquared_adj),
        "f_pvalue": float(ols.f_pvalue) if ols.f_pvalue is not None and np.isfinite(ols.f_pvalue) else None,
        "coefficients": coef_rows,
        "method_note": "Ordinary least squares. Check linearity, independence, and residual normality for validity.",
    }


def run_logit_summary(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    sm = _try_import_statsmodels()
    if sm is None:
        return {"error": "statsmodels is not installed. Add `statsmodels` to the environment."}

    cols = [target_col] + list(feature_cols)
    d = df[cols].dropna().copy()
    t = d[target_col]
    if t.nunique() != 2:
        return {"error": "Logistic model requires a binary target (exactly 2 values)."}
    u = sorted(t.unique().tolist(), key=str)
    # Map to 0/1
    d["_ybin"] = (d[target_col] == u[1]).astype(int)
    y = d["_ybin"]
    X = d[feature_cols].astype(float)
    X = sm.add_constant(X, has_constant="add")
    if len(d) < max(30, 2 * len(feature_cols) + 10):
        return {"error": f"Too few complete rows ({len(d)}) for stable logistic regression."}

    logit = sm.Logit(y, X).fit(disp=0)
    conf = logit.conf_int()
    coef_rows = []
    for name in X.columns:
        lo, hi = conf.loc[name, 0], conf.loc[name, 1]
        coef_rows.append(
            {
                "name": str(name),
                "coef": float(logit.params[name]),
                "std_err": float(logit.bse[name]),
                "p_value": float(logit.pvalues[name]) if name in logit.pvalues else None,
                "odds_ratio": float(np.exp(logit.params[name])),
                "ci_lower": float(np.exp(lo)),
                "ci_upper": float(np.exp(hi)),
            }
        )
    return {
        "model": "logit",
        "target": target_col,
        "positive_class": str(u[1]),
        "n_obs": int(logit.nobs),
        "pseudo_r_squared": float(logit.prsquared),
        "llr_pvalue": float(logit.llr_pvalue) if logit.llr_pvalue is not None else None,
        "coefficients": coef_rows,
        "method_note": "Logistic regression (MLE). Coefficients on log-odds; odds_ratio = exp(coef).",
    }


class RegressionSummaryInput(BaseModel):
    dataset_ref: str = Field(description="Dataset reference id")
    target_col: str = Field(description="Outcome column")
    feature_cols: list[str] = Field(
        min_length=1,
        description="Numeric predictor column names (no categoricals in v1; encode or bin first).",
    )
    model: Literal["ols", "logit"] = Field(
        default="ols",
        description="ols for numeric target; logit for binary classification target",
    )


@tool(args_schema=RegressionSummaryInput)
def regression_summary_tool(
    dataset_ref: str,
    target_col: str,
    feature_cols: list[str],
    model: str = "ols",
) -> str:
    """
    Fit OLS (numeric target) or binary logistic regression and return coefficients, p-values, and CIs.

    Use for inferential "which factors matter after controlling for others" — not just bivariate correlation.
    Only numeric feature columns are supported; one-hot or ordinal-encode categoricals first.
    """
    try:
        sm = _try_import_statsmodels()
        if sm is None:
            return "✗ regression_summary_tool requires the `statsmodels` package. Install with: pip install statsmodels"

        df = resolve_dataset(dataset_ref)
        if target_col not in df.columns:
            return f"✗ target_col '{target_col}' not found."
        for c in feature_cols:
            if c not in df.columns:
                return f"✗ feature column not found: {c}"
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in feature_cols + [target_col]):
            return "✗ All of target and features must be numeric for this tool."

        if model == "logit":
            res = run_logit_summary(df, target_col, feature_cols)
        else:
            res = run_ols_summary(df, target_col, feature_cols)
        if "error" in res:
            return f"✗ {res['error']}"
        lines = [
            f"# Regression summary ({res.get('model', model)})",
            f"**Target:** `{target_col}`  **N:** {res.get('n_obs')}",
        ]
        if res.get("r_squared") is not None:
            lines.append(f"**R² / adj-R²:** {res.get('r_squared'):.4f} / {res.get('adj_r_squared'):.4f}")
        if res.get("pseudo_r_squared") is not None:
            lines.append(f"**McFadden pseudo R²:** {res.get('pseudo_r_squared'):.4f}")
        lines.append("\n| Term | Coef | p |")
        lines.append("|---|---|---|")
        for row in (res.get("coefficients") or [])[:20]:
            lines.append(
                f"| `{row['name']}` | {row.get('coef', 0):.4g} | {row.get('p_value', '—')} |"
            )
        if len((res.get("coefficients") or [])) > 20:
            lines.append("| ... | ... | ... |")
        md = "\n".join(lines)
        top = (res.get("coefficients") or [None])[1] if len(res.get("coefficients") or []) > 1 else (res.get("coefficients") or [None])[0]
        p0 = top.get("p_value") if isinstance(top, dict) else None
        summary = f"{res.get('model')} n={res.get('n_obs')}, top_p={p0 if p0 is not None else 'n/a'}"
        return attach_analysis_sidecar(
            md,
            kind="inferential_regression",
            tool="regression_summary_tool",
            summary=summary,
            payload=res,
        )
    except Exception as e:
        return f"✗ regression_summary_tool failed: {type(e).__name__}: {e}"


regression_tools = [regression_summary_tool]

__all__ = [
    "regression_summary_tool",
    "run_ols_summary",
    "run_logit_summary",
    "regression_tools",
]
