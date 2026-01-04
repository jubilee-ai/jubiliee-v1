"""
feature_diagnostics - Audit feature set for safety and quality before training.

Checks missingness patterns, redundancy groups, leakage risks, and stability.
Produces explicit drop_list and transform_suggestions with rationales.
Rerun after adding engineered features.

Use before finalizing training features. Critical when time columns or 
post-outcome joins are involved.
"""

import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset

# =============================================================================
# HELPERS
# =============================================================================

def _r(value, decimals: int = 2) -> float:
    """Round and convert to native float."""
    if pd.isna(value):
        return None
    return float(round(value, decimals))


def _cv_to_stability(series: pd.Series) -> Optional[float]:
    """Convert coefficient of variation to stability score (0-1, higher = more stable)."""
    if len(series) < 2:
        return None
    mean = series.mean()
    if mean == 0:
        return None
    cv = series.std() / (mean + 1e-10)
    return _r(max(0, 1 - cv))


def _safe_skew(series: pd.Series) -> float:
    """Calculate skewness, handling edge cases."""
    try:
        clean = series.dropna()
        if len(clean) < 3:
            return 0.0
        return float(stats.skew(clean, nan_policy='omit'))
    except Exception:
        return 0.0


def _null_target_correlation(feature: pd.Series, target: pd.Series) -> Optional[float]:
    """
    Check if missingness in feature is correlated with target.
    Returns correlation between is_null indicator and target.
    """
    try:
        if feature.isna().sum() == 0:
            return None  # No nulls to analyze
        
        is_null = feature.isna().astype(int)
        mask = target.notna()
        
        if mask.sum() < 10:
            return None
        
        corr = is_null[mask].corr(target[mask].astype(float))
        return _r(corr, 3) if pd.notna(corr) else None
    except Exception:
        return None


def _compute_stability(
    df: pd.DataFrame,
    feature_col: str,
    time_col: str,
    n_periods: int = 4
) -> Optional[float]:
    """
    Compute stability score by comparing feature distribution across time periods.
    Returns a score 0-1 where 1 = perfectly stable.
    """
    try:
        if time_col not in df.columns or feature_col not in df.columns:
            return None
        
        # Create time periods
        time_series = pd.to_datetime(df[time_col], errors='coerce')
        if time_series.isna().all():
            return None
        
        # Split into n_periods
        df_temp = df[[feature_col]].copy()
        df_temp['_time_'] = time_series
        df_temp = df_temp.dropna(subset=['_time_'])
        
        if len(df_temp) < n_periods * 10:
            return None
        
        df_temp['_period_'] = pd.qcut(df_temp['_time_'].rank(method='first'), 
                                       q=n_periods, labels=False, duplicates='drop')
        
        # Compare mean/std across periods for numeric
        if pd.api.types.is_numeric_dtype(df[feature_col]):
            period_means = df_temp.groupby('_period_')[feature_col].mean()
            return _cv_to_stability(period_means)
        else:
            # For categorical, compare top category proportion
            period_top = df_temp.groupby('_period_')[feature_col].apply(
                lambda x: x.value_counts(normalize=True).iloc[0] if len(x) > 0 else 0
            )
            return _cv_to_stability(period_top)
    except Exception:
        return None


def _check_leakage(
    df: pd.DataFrame,
    feature_col: str,
    target_col: str,
    suspect_cols: list[str],
    target_date_col: Optional[str] = None,
    time_col: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Check if a feature has potential leakage.
    Returns (risk_level, reason).
    """
    # Check if feature is in suspect list
    if feature_col in suspect_cols:
        return "high", "Marked as suspect — may be derived from outcome"
    
    # Check if feature name suggests post-outcome data
    leaky_keywords = ['final', 'outcome', 'result', 'approved', 'rejected', 
                      'closed', 'end_', 'last_', 'total_after', 'post_']
    for keyword in leaky_keywords:
        if keyword in feature_col.lower():
            return "medium", f"Name suggests post-outcome data ('{keyword}')"
    
    # Check temporal leakage if dates available
    if target_date_col and time_col and target_date_col in df.columns:
        try:
            target_dates = pd.to_datetime(df[target_date_col], errors='coerce')
            feature_dates = pd.to_datetime(df[time_col], errors='coerce') if time_col in df.columns else None
            
            if feature_dates is not None:
                # Check if feature data comes after target date
                mask = target_dates.notna() & feature_dates.notna()
                if mask.sum() > 10:
                    after_target = (feature_dates[mask] > target_dates[mask]).mean()
                    if after_target > 0.1:
                        return "high", f"Feature date is after target date in {after_target:.0%} of cases"
        except Exception:
            pass
    
    # Check if feature has suspiciously high correlation with target
    try:
        if pd.api.types.is_numeric_dtype(df[feature_col]):
            corr = df[feature_col].corr(df[target_col].astype(float))
            if pd.notna(corr) and abs(corr) > 0.95:
                return "high", f"Suspiciously high correlation with target (r={corr:.2f})"
    except Exception:
        pass
    
    return "none", None


def _find_redundancy_groups(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.85,
    max_pairs: int = 50
) -> tuple[list[dict], dict]:
    """
    Find groups of highly correlated features.
    Returns (redundancy_groups, column_to_group mapping).
    """
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(numeric_cols) < 2:
        return [], {}
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr().abs()
    
    # Find pairs above threshold
    pairs = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            corr = corr_matrix.loc[col1, col2]
            if pd.notna(corr) and corr >= threshold:
                pairs.append((col1, col2, corr))
    
    # Build groups using union-find approach
    parent = {col: col for col in numeric_cols}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    for col1, col2, _ in pairs[:max_pairs]:
        union(col1, col2)
    
    # Build groups
    groups_dict = {}
    for col in numeric_cols:
        root = find(col)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(col)
    
    # Filter to groups with >1 member and format output
    redundancy_groups = []
    col_to_group = {}
    group_id = 1
    
    for root, cols in groups_dict.items():
        if len(cols) > 1:
            # Find max correlation within group
            max_corr = 0
            for i, c1 in enumerate(cols):
                for c2 in cols[i+1:]:
                    c = corr_matrix.loc[c1, c2]
                    if pd.notna(c) and c > max_corr:
                        max_corr = c
            
            group_name = f"group_{group_id}"
            redundancy_groups.append({
                "group": group_name,
                "columns": cols,
                "max_corr": _r(max_corr, 3)
            })
            
            for col in cols:
                col_to_group[col] = group_name
            
            group_id += 1
    
    return redundancy_groups, col_to_group


def _generate_transform_suggestion(
    column: str,
    skew: float,
    null_pct: float,
    target_corr: Optional[float] = None,
) -> Optional[dict]:
    """Generate transform suggestion for a feature."""
    
    # High skew → log transform
    if abs(skew) > 2:
        return {
            "column": column,
            "action": "log1p",
            "reason": f"right-skewed ({skew:.1f})" if skew > 0 else f"left-skewed ({skew:.1f})"
        }
    
    # Moderate skew → binning
    if abs(skew) > 1:
        return {
            "column": column,
            "action": "bin",
            "reason": f"moderately skewed ({skew:.1f})",
            "params": {"bins": 5, "strategy": "quantile"}
        }
    
    return None


# =============================================================================
# MAIN DIAGNOSTICS FUNCTION
# =============================================================================

def run_feature_diagnostics(
    dataset_ref: str,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    time_col: Optional[str] = None,
    entity_id: Optional[str] = None,
    leakage_checks: Optional[dict] = None,
    sample_n: int = 100000,
    caps: Optional[dict] = None,
) -> dict:
    """
    Run comprehensive feature diagnostics.
    
    Args:
        dataset_ref: Reference to dataset
        feature_cols: List of feature columns to analyze
        target_col: Target column name
        task_type: 'classification' or 'regression'
        time_col: Optional time column for stability analysis
        entity_id: Optional entity ID column
        leakage_checks: Optional dict with suspect_cols and target_date_col
        sample_n: Max rows to sample
        caps: Limits for output size
    
    Returns:
        Diagnostics report dictionary
    """
    df = resolve_dataset(dataset_ref)
    
    # Apply caps
    caps = caps or {}
    max_redundancy_pairs = caps.get("max_redundancy_pairs", 50)
    
    # Sample if needed
    if len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42)
    
    # Validate columns exist
    available_cols = df.columns.tolist()
    feature_cols = [c for c in feature_cols if c in available_cols]
    
    if target_col not in available_cols:
        return {"error": f"Target column '{target_col}' not found"}
    
    # Leakage check config
    leakage_checks = leakage_checks or {}
    suspect_cols = leakage_checks.get("suspect_cols", [])
    target_date_col = leakage_checks.get("target_date_col")
    
    # Find redundancy groups
    redundancy_groups, col_to_group = _find_redundancy_groups(
        df, feature_cols, max_pairs=max_redundancy_pairs
    )
    
    # Analyze each feature
    feature_health = []
    drop_list = []
    transform_suggestions = []
    keep_list = []
    
    target_series = df[target_col]
    
    for col in feature_cols:
        if col == target_col:
            continue
        
        series = df[col]
        
        # Basic stats
        null_pct = _r(series.isna().mean(), 3)
        null_target_corr = _null_target_correlation(series, target_series)
        
        # Skew (numeric only)
        skew = None
        if pd.api.types.is_numeric_dtype(series):
            skew = _r(_safe_skew(series))
        
        # Redundancy group
        redundancy_group = col_to_group.get(col)
        
        # Leakage check
        leakage_risk, leakage_reason = _check_leakage(
            df, col, target_col, suspect_cols, target_date_col, time_col
        )
        
        # Stability (if time column provided)
        stability = None
        if time_col:
            stability = _compute_stability(df, col, time_col)
        
        # Build health record
        health = {
            "column": col,
            "null_pct": null_pct,
            "null_target_corr": null_target_corr,
            "skew": skew,
            "redundancy_group": redundancy_group,
            "leakage_risk": leakage_risk,
            "stability": stability,
        }
        
        if leakage_reason:
            health["leakage_reason"] = leakage_reason
        
        feature_health.append(health)
        
        # Determine if should drop
        should_drop = False
        drop_reason = None
        
        # Drop for leakage
        if leakage_risk == "high":
            should_drop = True
            drop_reason = f"leakage — {leakage_reason}"
        
        # Drop for high nulls
        elif null_pct > 0.7:
            should_drop = True
            drop_reason = f"too many nulls ({null_pct:.0%})"
        
        # Drop redundant (keep first in group)
        elif redundancy_group:
            group = next((g for g in redundancy_groups if g["group"] == redundancy_group), None)
            if group and col != group["columns"][0]:
                should_drop = True
                primary = group["columns"][0]
                drop_reason = f"redundant with {primary} (r={group['max_corr']})"
        
        if should_drop:
            drop_list.append({"column": col, "reason": drop_reason})
        else:
            # Check for transform suggestions
            if skew is not None:
                suggestion = _generate_transform_suggestion(col, skew, null_pct)
                if suggestion:
                    transform_suggestions.append(suggestion)
            
            # Add to keep list if not dropped
            keep_list.append(col)
    
    # Generate summary
    top_actions = []
    
    # Leakage drops
    leakage_drops = [d for d in drop_list if "leakage" in d["reason"]]
    if leakage_drops:
        cols = [d["column"] for d in leakage_drops]
        top_actions.append(f"DROP {', '.join(cols[:3])} (leakage)")
    
    # Redundant drops
    redundant_drops = [d for d in drop_list if "redundant" in d["reason"]]
    if redundant_drops:
        cols = [d["column"] for d in redundant_drops]
        top_actions.append(f"DROP {', '.join(cols[:3])} (redundant)")
    
    # Transforms
    if transform_suggestions:
        for t in transform_suggestions[:2]:
            top_actions.append(f"TRANSFORM {t['column']} → {t['action']}")
    
    summary = {
        "features_checked": len(feature_cols),
        "drop": len(drop_list),
        "transform": len(transform_suggestions),
        "keep_as_is": len(keep_list) - len(transform_suggestions),
        "top_actions": top_actions[:5],
    }
    
    return {
        "feature_health": feature_health,
        "redundancy_groups": redundancy_groups,
        "drop_list": drop_list,
        "transform_suggestions": transform_suggestions,
        "keep_list": keep_list,
        "summary": summary,
    }


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class LeakageChecks(BaseModel):
    """Configuration for leakage detection."""
    suspect_cols: list[str] = Field(
        default_factory=list,
        description="Columns suspected of leakage (e.g., post-outcome values)"
    )
    target_date_col: Optional[str] = Field(
        default=None,
        description="Column containing target/outcome date for temporal leakage check"
    )


class DiagnosticsCaps(BaseModel):
    """Caps for diagnostics output size."""
    max_redundancy_pairs: int = Field(default=50, description="Max correlation pairs to analyze")


class FeatureDiagnosticsInput(BaseModel):
    """Input schema for feature_diagnostics tool."""
    dataset_ref: str = Field(description="Dataset reference from previous operation")
    feature_cols: list[str] = Field(description="List of feature columns to analyze")
    target_col: str = Field(description="Target column name")
    task_type: Literal["classification", "regression"] = Field(
        description="Task type for analysis"
    )
    time_col: Optional[str] = Field(
        default=None,
        description="Time column for stability analysis"
    )
    entity_id: Optional[str] = Field(
        default=None,
        description="Entity ID column (for group-aware analysis)"
    )
    leakage_checks: Optional[LeakageChecks] = Field(
        default=None,
        description="Configuration for leakage detection"
    )
    sample_n: int = Field(
        default=100000,
        description="Max rows to sample for analysis"
    )
    caps: Optional[DiagnosticsCaps] = Field(
        default=None,
        description="Caps for output size"
    )


@tool(args_schema=FeatureDiagnosticsInput)
def feature_diagnostics_tool(
    dataset_ref: str,
    feature_cols: list[str],
    target_col: str,
    task_type: str,
    time_col: Optional[str] = None,
    entity_id: Optional[str] = None,
    leakage_checks: Optional[dict] = None,
    sample_n: int = 100000,
    caps: Optional[dict] = None,
) -> str:
    """
    Audit feature set for safety and quality before training.
    
    WHAT IT CHECKS:
    - Missingness patterns (is null correlated with target?)
    - Redundancy groups (highly correlated feature clusters)
    - Leakage risks (features derived from target or future data)
    - Stability across time (if time_col provided)
    - Skewness and distribution issues
    
    WHAT IT PRODUCES:
    - feature_health: Per-feature diagnostics
    - drop_list: Columns to remove with reasons
    - transform_suggestions: Recommended transformations
    - keep_list: Safe features to use
    - summary: Top actions to take
    
    WHEN TO USE:
    - Before finalizing training features
    - After adding new engineered features
    - When time columns or post-outcome joins are involved
    
    EXAMPLES:
    ```
    feature_diagnostics(
        dataset_ref="loan_data",
        feature_cols=["income", "age", "debt_ratio", "final_balance"],
        target_col="default",
        task_type="classification",
        leakage_checks={"suspect_cols": ["final_balance"]}
    )
    ```
    """
    try:
        leakage_dict = leakage_checks if isinstance(leakage_checks, dict) else (
            leakage_checks.model_dump() if leakage_checks else None
        )
        caps_dict = caps if isinstance(caps, dict) else (
            caps.model_dump() if caps else None
        )
        
        result = run_feature_diagnostics(
            dataset_ref=dataset_ref,
            feature_cols=feature_cols,
            target_col=target_col,
            task_type=task_type,
            time_col=time_col,
            entity_id=entity_id,
            leakage_checks=leakage_dict,
            sample_n=sample_n,
            caps=caps_dict,
        )
        
        if "error" in result:
            return f"✗ {result['error']}"
        
        # Format output for agent
        lines = []
        
        # Header
        lines.append("# Feature Diagnostics Report")
        lines.append("")
        
        # Summary
        summary = result["summary"]
        lines.append(f"**Features checked:** {summary['features_checked']}")
        lines.append(f"**Drop:** {summary['drop']} | **Transform:** {summary['transform']} | **Keep as-is:** {summary['keep_as_is']}")
        lines.append("")
        
        # Drop list (most important)
        if result["drop_list"]:
            lines.append("## ❌ Drop List")
            for item in result["drop_list"]:
                lines.append(f"- `{item['column']}`: {item['reason']}")
            lines.append("")
        
        # Transform suggestions
        if result["transform_suggestions"]:
            lines.append("## 🔧 Transform Suggestions")
            for t in result["transform_suggestions"]:
                params = f" {t.get('params', '')}" if t.get('params') else ""
                lines.append(f"- `{t['column']}` → **{t['action']}**{params}: {t['reason']}")
            lines.append("")
        
        # Redundancy groups
        if result["redundancy_groups"]:
            lines.append("## 🔄 Redundancy Groups")
            for g in result["redundancy_groups"]:
                cols = ", ".join([f"`{c}`" for c in g["columns"]])
                lines.append(f"- **{g['group']}**: {cols} (max r={g['max_corr']})")
            lines.append("")
        
        # Feature health (compact)
        if result["feature_health"]:
            lines.append("## 📊 Feature Health")
            
            # Show problematic features first
            problematic = [f for f in result["feature_health"] 
                          if f["leakage_risk"] != "none" or f["null_pct"] > 0.3 
                          or (f["skew"] and abs(f["skew"]) > 2)]
            healthy = [f for f in result["feature_health"] 
                      if f not in problematic]
            
            for f in problematic[:10]:
                issues = []
                if f["leakage_risk"] != "none":
                    issues.append(f"⚠️ {f['leakage_risk']} leakage")
                if f["null_pct"] > 0.3:
                    issues.append(f"❌ {f['null_pct']:.0%} null")
                if f["skew"] and abs(f["skew"]) > 2:
                    issues.append(f"📈 skew={f['skew']}")
                
                lines.append(f"- `{f['column']}`: {', '.join(issues)}")
            
            if healthy:
                lines.append(f"- ... +{len(healthy)} healthy features")
            lines.append("")
        
        # Keep list (compact)
        if result["keep_list"]:
            lines.append("## ✅ Keep List")
            keep_str = ", ".join([f"`{c}`" for c in result["keep_list"][:15]])
            lines.append(keep_str)
            if len(result["keep_list"]) > 15:
                lines.append(f"... +{len(result['keep_list']) - 15} more")
            lines.append("")
        
        # Top actions
        if summary["top_actions"]:
            lines.append("## 📋 Top Actions")
            for action in summary["top_actions"]:
                lines.append(f"- {action}")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"✗ Feature diagnostics failed: {type(e).__name__}: {e}"


# Export
diagnostics_tools = [feature_diagnostics_tool]

__all__ = [
    "feature_diagnostics_tool",
    "run_feature_diagnostics",
    "diagnostics_tools",
]

