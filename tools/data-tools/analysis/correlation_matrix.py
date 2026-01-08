"""
correlation_matrix - Compute correlation matrix between numeric features.

Provides full correlation matrix, top correlated pairs, and multicollinearity flags.
Use to understand feature relationships and detect redundancy before modeling.
"""

import sys
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformations.tool_utils import resolve_dataset

# =============================================================================
# HELPERS
# =============================================================================

def _round(value: float, decimals: int = 3) -> float:
    """Round and convert to native float."""
    if pd.isna(value):
        return None
    return float(round(value, decimals))


# =============================================================================
# CORE FUNCTION
# =============================================================================

def compute_correlation_matrix(
    dataset_ref: str,
    columns: Optional[list[str]] = None,
    method: str = "pearson",
    threshold: float = 0.7,
    top_n: int = 20,
) -> dict:
    """
    Compute correlation matrix for numeric columns in a dataset.
    
    Args:
        dataset_ref: Reference to dataset
        columns: Optional list of columns to include. If None, uses all numeric columns.
        method: Correlation method ('pearson', 'spearman', 'kendall')
        threshold: Threshold for flagging high correlations
        top_n: Number of top correlated pairs to return
    
    Returns:
        Dict with correlation matrix, top pairs, and multicollinearity flags
    """
    df = resolve_dataset(dataset_ref)
    
    # Select numeric columns
    numeric_df = df.select_dtypes(include=["number"])
    
    # Filter to requested columns if specified
    if columns:
        valid_cols = [c for c in columns if c in numeric_df.columns]
        if not valid_cols:
            return {"error": f"No valid numeric columns found in: {columns}"}
        numeric_df = numeric_df[valid_cols]
    
    if numeric_df.shape[1] < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis"}
    
    # Compute correlation matrix using pandas
    corr_matrix = numeric_df.corr(method=method)
    
    # Extract all pairs with correlations
    pairs = []
    cols = corr_matrix.columns.tolist()
    for i, col1 in enumerate(cols):
        for col2 in cols[i + 1:]:
            corr_value = corr_matrix.loc[col1, col2]
            if pd.notna(corr_value):
                pairs.append({
                    "col1": col1,
                    "col2": col2,
                    "correlation": _round(corr_value),
                    "abs_correlation": _round(abs(corr_value)),
                })
    
    # Sort by absolute correlation descending
    pairs.sort(key=lambda x: x["abs_correlation"], reverse=True)
    
    # Get top N pairs
    top_pairs = pairs[:top_n]
    
    # Flag high correlations (potential multicollinearity)
    high_correlations = [p for p in pairs if p["abs_correlation"] >= threshold]
    
    # Build condensed matrix (as nested dict for agent readability)
    matrix_dict = {}
    for col in cols:
        matrix_dict[col] = {
            other: _round(corr_matrix.loc[col, other])
            for other in cols
        }
    
    return {
        "method": method,
        "n_features": len(cols),
        "columns": cols,
        "matrix": matrix_dict,
        "top_pairs": top_pairs,
        "high_correlations": high_correlations,
        "threshold": threshold,
    }


# =============================================================================
# LANGCHAIN TOOL
# =============================================================================

class CorrelationMatrixInput(BaseModel):
    """Input schema for correlation_matrix_tool."""
    dataset_ref: str = Field(
        description="Dataset reference from previous operation or table name"
    )
    columns: Optional[list[str]] = Field(
        default=None,
        description="Optional list of columns to include. If not provided, uses all numeric columns."
    )
    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson",
        description="Correlation method: 'pearson' (linear), 'spearman' (rank), or 'kendall' (rank)"
    )
    threshold: float = Field(
        default=0.7,
        description="Threshold for flagging high correlations (0-1). Pairs above this are flagged."
    )
    top_n: int = Field(
        default=20,
        description="Number of top correlated pairs to return"
    )


@tool(args_schema=CorrelationMatrixInput)
def correlation_matrix_tool(
    dataset_ref: str,
    columns: Optional[list[str]] = None,
    method: str = "pearson",
    threshold: float = 0.7,
    top_n: int = 20,
) -> str:
    """
    Compute correlation matrix between numeric features in a dataset.
    
    USE THIS TOOL WHEN YOU NEED TO:
    - Understand relationships between numeric features
    - Detect multicollinearity before model training
    - Find redundant features that can be dropped
    - Identify feature pairs that move together
    
    METHODS:
    - pearson: Linear correlation (default, assumes normal distribution)
    - spearman: Rank correlation (robust to outliers and non-linear relationships)
    - kendall: Rank correlation (good for small samples or many ties)
    
    RETURNS:
    - Full correlation matrix
    - Top correlated pairs (sorted by absolute correlation)
    - High correlation flags (pairs above threshold)
    
    EXAMPLES:
    ```
    # All numeric columns with default settings
    correlation_matrix(dataset_ref="loan_data")
    
    # Specific columns with Spearman correlation
    correlation_matrix(dataset_ref="loan_data", columns=["income", "age", "debt"], method="spearman")
    
    # Lower threshold for stricter multicollinearity detection
    correlation_matrix(dataset_ref="loan_data", threshold=0.5)
    ```
    """
    try:
        result = compute_correlation_matrix(
            dataset_ref=dataset_ref,
            columns=columns,
            method=method,
            threshold=threshold,
            top_n=top_n,
        )
        
        if "error" in result:
            return f"✗ {result['error']}"
        
        # Format output for agent
        lines = []
        
        lines.append("# Correlation Matrix Analysis")
        lines.append("")
        lines.append(f"**Method:** {result['method']}")
        lines.append(f"**Features:** {result['n_features']} numeric columns")
        lines.append(f"**Threshold:** {result['threshold']}")
        lines.append("")
        
        # Top correlated pairs
        if result["top_pairs"]:
            lines.append("## Top Correlated Pairs")
            for pair in result["top_pairs"][:15]:
                sign = "+" if pair["correlation"] > 0 else ""
                flag = " ⚠️" if pair["abs_correlation"] >= threshold else ""
                lines.append(
                    f"- `{pair['col1']}` ↔ `{pair['col2']}`: "
                    f"r = {sign}{pair['correlation']}{flag}"
                )
            if len(result["top_pairs"]) > 15:
                lines.append(f"- ... +{len(result['top_pairs']) - 15} more pairs")
            lines.append("")
        
        # High correlations (multicollinearity flags)
        if result["high_correlations"]:
            lines.append("## ⚠️ High Correlations (Multicollinearity Risk)")
            lines.append(f"Found {len(result['high_correlations'])} pairs above threshold {threshold}:")
            for pair in result["high_correlations"][:10]:
                lines.append(
                    f"- `{pair['col1']}` ↔ `{pair['col2']}`: r = {pair['correlation']}"
                )
            lines.append("")
            lines.append("**Recommendation:** Consider dropping one feature from each highly correlated pair, "
                        "or use dimensionality reduction (PCA) to combine them.")
        else:
            lines.append("## ✓ No High Correlations")
            lines.append(f"No feature pairs exceed the {threshold} threshold.")
        lines.append("")
        
        # Compact matrix (show first 8x8 if large)
        cols = result["columns"]
        max_display = 8
        if len(cols) <= max_display:
            lines.append("## Correlation Matrix")
            # Header row
            header = "| | " + " | ".join(f"`{c[:8]}`" for c in cols) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(cols) + 1))
            # Data rows
            for row_col in cols:
                row_vals = [
                    f"{result['matrix'][row_col][c]:.2f}" if result['matrix'][row_col][c] is not None else "—"
                    for c in cols
                ]
                lines.append(f"| `{row_col[:8]}` | " + " | ".join(row_vals) + " |")
        else:
            lines.append(f"## Matrix Preview (first {max_display} of {len(cols)} features)")
            preview_cols = cols[:max_display]
            header = "| | " + " | ".join(f"`{c[:6]}`" for c in preview_cols) + " |"
            lines.append(header)
            lines.append("|" + "---|" * (len(preview_cols) + 1))
            for row_col in preview_cols:
                row_vals = [
                    f"{result['matrix'][row_col][c]:.2f}" if result['matrix'][row_col].get(c) is not None else "—"
                    for c in preview_cols
                ]
                lines.append(f"| `{row_col[:6]}` | " + " | ".join(row_vals) + " |")
            lines.append(f"\n*Full matrix has {len(cols)} features. Use `columns` parameter to focus on specific features.*")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"✗ Correlation matrix failed: {type(e).__name__}: {e}"


# =============================================================================
# EXPORTS
# =============================================================================

correlation_tools = [correlation_matrix_tool]

__all__ = [
    "correlation_matrix_tool",
    "compute_correlation_matrix",
    "correlation_tools",
]

