"""Declarative chart data for Jubilee chat UI (Recharts)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from transformations.tool_utils import resolve_dataset

from .analysis_sidecar import attach_analysis_sidecar

_ChartKind = Literal["bar", "grouped_bar", "histogram", "scatter", "line", "box"]


class ChartToolInput(BaseModel):
    dataset_ref: str = Field(description="Registered dataset reference")
    chart_type: _ChartKind = Field(description="Visualization type")
    x: str = Field(description="Primary X column (category or numeric for histogram)")
    y: Optional[str] = Field(
        default=None,
        description="Numeric Y column (not used for histogram of x)",
    )
    group_by: Optional[str] = Field(
        default=None,
        description="Second categorical column for grouped_bar",
    )
    agg: str = Field(
        default="mean",
        description="Aggregation for bar/grouped_bar: mean, median, sum, count",
    )
    title: Optional[str] = Field(default=None, description="Chart title")
    top_n: int = Field(default=20, ge=1, le=200)
    n_bins: int = Field(default=15, ge=5, le=50, description="Bins for histogram")


_AGG_FUNCS = {
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "count": "count",
}


def _agg_series(s: pd.Series, how: str) -> float:
    how = how.lower().strip()
    if how == "mean":
        return float(s.mean())
    if how == "median":
        return float(s.median())
    if how == "sum":
        return float(s.sum())
    if how == "count":
        return float(s.count())
    return float(s.mean())


def build_chart_spec(
    dataset_ref: str,
    chart_type: _ChartKind,
    x: str,
    y: Optional[str],
    group_by: Optional[str],
    agg: str,
    title: Optional[str],
    top_n: int,
    n_bins: int,
) -> tuple[str, dict[str, Any]]:
    df = resolve_dataset(dataset_ref)
    if x not in df.columns:
        raise ValueError(f"Column '{x}' not in dataset")
    spec: dict[str, Any] = {
        "type": chart_type,
        "title": title or f"{chart_type}: {x}",
        "dataset_ref": dataset_ref,
        "xKey": x,
        "yKey": y or "",
    }

    if chart_type == "histogram":
        series = pd.to_numeric(df[x], errors="coerce").dropna()
        if series.empty:
            raise ValueError(f"No numeric values in column '{x}'")
        counts, edges = np.histogram(series.values, bins=n_bins)
        data = []
        for i, c in enumerate(counts):
            lo, hi = edges[i], edges[i + 1]
            label = f"{lo:.3g}–{hi:.3g}"
            data.append({"bin": label, "count": int(c), "x": label})
        spec["type"] = "histogram"
        spec["data"] = data
        spec["xKey"] = "x"
        spec["yKey"] = "count"
        md = (
            f"Histogram of `{x}` ({len(series)} values, {n_bins} bins). "
            f"Peaks indicate where values cluster."
        )
        return md, spec

    if chart_type == "scatter":
        if not y or y not in df.columns:
            raise ValueError("scatter requires numeric y column")
        d = df[[x, y]].dropna()
        d[x] = pd.to_numeric(d[x], errors="coerce")
        d[y] = pd.to_numeric(d[y], errors="coerce")
        d = d.dropna()
        sample = d.head(top_n * 50)
        spec["data"] = [
            {str(x): float(row[x]), str(y): float(row[y])}
            for _, row in sample.iterrows()
        ]
        md = f"Scatter `{x}` vs `{y}` ({len(sample)} points shown)."
        return md, spec

    if chart_type == "line":
        if not y or y not in df.columns:
            raise ValueError("line requires y column")
        d = df[[x, y]].copy()
        d[y] = pd.to_numeric(d[y], errors="coerce")
        d = d.dropna(subset=[y])
        try:
            d[x] = pd.to_datetime(d[x])
            d = d.sort_values(x)
        except Exception:
            d = d.sort_values(x)
        sample = d.head(top_n * 100)
        spec["data"] = []
        for _, row in sample.iterrows():
            xv = row[x]
            xv_out = xv.isoformat() if hasattr(xv, "isoformat") else str(xv)
            spec["data"].append({"x": xv_out, str(y): float(row[y])})
        md = f"Line trend of `{y}` over `{x}` ({len(sample)} points)."
        return md, spec

    if chart_type == "box":
        if not y or y not in df.columns:
            raise ValueError("box plot requires numeric y column")
        if x not in df.columns:
            raise ValueError(f"Column '{x}' not in dataset")
        grp = df.groupby(df[x].astype(str), dropna=False)[y].apply(
            lambda s: pd.to_numeric(s, errors="coerce").dropna()
        )
        rows = []
        for cat, ser in grp.items():
            if len(ser) == 0:
                continue
            rows.append(
                {
                    "name": str(cat)[:80],
                    "min": float(ser.min()),
                    "q1": float(ser.quantile(0.25)),
                    "median": float(ser.median()),
                    "q3": float(ser.quantile(0.75)),
                    "max": float(ser.max()),
                }
            )
        rows = sorted(rows, key=lambda r: r["median"], reverse=True)[:top_n]
        spec["type"] = "box"
        spec["data"] = rows
        spec["xKey"] = "name"
        md = f"Box-style summary of `{y}` by `{x}` ({len(rows)} groups)."
        return md, spec

    if chart_type == "grouped_bar":
        if not y or y not in df.columns:
            raise ValueError("grouped_bar requires y column")
        if not group_by or group_by not in df.columns:
            raise ValueError("grouped_bar requires group_by column")
        ser_y = pd.to_numeric(df[y], errors="coerce")
        tmp = pd.DataFrame({x: df[x].astype(str), group_by: df[group_by].astype(str), "_y": ser_y})
        tmp = tmp.dropna(subset=["_y"])
        pivot = tmp.groupby([x, group_by], observed=False)["_y"].mean().reset_index()
        pivot = pivot.sort_values("_y", ascending=False).head(top_n * 12)
        # wide format for Recharts: one row per x, dynamic series keys = group_by values
        cats = pivot[x].unique().tolist()
        groups = pivot[group_by].unique().tolist()
        wide: dict[Any, dict[str, Any]] = {}
        for _, row in pivot.iterrows():
            xv = str(row[x])
            if xv not in wide:
                wide[xv] = {str(x): xv}
            wide[xv][str(row[group_by])] = float(row["_y"])
        spec["type"] = "grouped_bar"
        spec["bars"] = [str(g) for g in groups][:12]
        spec["data"] = list(wide.values())
        spec["groupKey"] = str(group_by)
        md = (
            f"Grouped bars: mean `{y}` by `{x}` × `{group_by}` "
            f"({len(spec['data'])} categories on X)."
        )
        return md, spec

    # bar (default)
    if not y or y not in df.columns:
        raise ValueError("bar chart requires numeric y column")
    ser_y = pd.to_numeric(df[y], errors="coerce")
    tmp = pd.DataFrame({x: df[x].astype(str), "_y": ser_y}).dropna(subset=["_y"])
    g = tmp.groupby(x, observed=False)["_y"].agg(lambda s: _agg_series(s, agg))
    g = g.sort_values(ascending=False).head(top_n)
    spec["type"] = "bar"
    spec["data"] = [{str(x): str(idx), str(y): float(val)} for idx, val in g.items()]
    spec["xKey"] = str(x)
    spec["yKey"] = str(y)
    md = f"Bar chart: `{agg}` of `{y}` by `{x}` ({len(spec['data'])} groups)."
    return md, spec


@tool(args_schema=ChartToolInput)
def chart_tool(
    dataset_ref: str,
    chart_type: _ChartKind = "bar",
    x: str = "",
    y: Optional[str] = None,
    group_by: Optional[str] = None,
    agg: str = "mean",
    title: Optional[str] = None,
    top_n: int = 20,
    n_bins: int = 15,
) -> str:
    """Build chart-ready JSON for the UI from a registered dataset.

    Pick **one** visualization that best answers the user's question.
    Supported types: bar, grouped_bar, histogram, scatter, line, box.
    """
    try:
        md, spec = build_chart_spec(
            dataset_ref=dataset_ref,
            chart_type=chart_type,
            x=x,
            y=y,
            group_by=group_by,
            agg=agg,
            title=title,
            top_n=top_n,
            n_bins=n_bins,
        )
        body = f"## Chart\n\n{md}\n"
        return attach_analysis_sidecar(
            body,
            kind="chart",
            tool="chart_tool",
            summary=(title or spec.get("title") or chart_type),
            payload={"spec": spec},
        )
    except Exception as e:
        return f"✗ chart_tool failed: {type(e).__name__}: {e}"


__all__ = ["chart_tool", "build_chart_spec"]
