"""
Custom tools for the dataset curator agent.

- download_kaggle_dataset: Download a Kaggle dataset locally and register its files
- export_csv: Export a registered dataset to CSV
"""

import json
import os
import re
import subprocess
import sys
import uuid
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

_DATA_TOOLS_DIR = Path(__file__).parent.parent.parent / "tools" / "data-tools"
if str(_DATA_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_TOOLS_DIR))

from utils import (
    DATASETS_DIR,
    get_registered_dataset,
    register_dataset,
    truncate_columns_display,
)

KAGGLE_DIR = DATASETS_DIR / "kaggle"
CURATED_DIR = DATASETS_DIR / "curated"
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
CURATED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _safe_ref(name: str) -> str:
    """Sanitize a string into a lowercase identifier safe for registry keys."""
    return re.sub(r"[^\w]", "_", name).lower()


def _classify_dtype(dtype_str: str) -> str:
    """Map a pandas dtype string to a human-readable category."""
    d = str(dtype_str)
    if "int" in d:
        return "integer"
    if "float" in d:
        return "float"
    if "bool" in d:
        return "boolean"
    if "datetime" in d:
        return "datetime"
    return "string"


def _get_or_error(ref: str) -> tuple[pd.DataFrame | None, str | None]:
    """Look up a registered dataset, returning (df, None) or (None, error_msg)."""
    df = get_registered_dataset(ref)
    if df is None:
        return None, f"**Error:** Dataset '{ref}' not found in registry."
    return df, None


def _load_tabular_file(path: Path) -> pd.DataFrame:
    """Read a CSV, TSV, or Parquet file into a DataFrame."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


# =============================================================================
# download_kaggle_dataset
# =============================================================================


class DownloadKaggleDatasetInput(BaseModel):
    owner_slug: str = Field(
        description="The Kaggle username or organization that owns the dataset. "
        "Example: 'uciml' from 'uciml/iris'"
    )
    dataset_slug: str = Field(
        description="The dataset slug on Kaggle. "
        "Example: 'iris' from 'uciml/iris'"
    )


@tool(args_schema=DownloadKaggleDatasetInput)
def download_kaggle_dataset(owner_slug: str, dataset_slug: str) -> str:
    """
    Download a Kaggle dataset to local storage and register all CSV/Parquet files.

    After downloading, each file is loaded into the dataset registry so it can be
    referenced by other tools (join_merge, transformations, etc.).

    Use search_datasets (Kaggle MCP) to find datasets first, then call this with
    the owner and slug from the results.

    Example: download_kaggle_dataset(owner_slug="uciml", dataset_slug="iris")
    """
    dataset_id = f"{owner_slug}/{dataset_slug}"
    dest_dir = KAGGLE_DIR / f"{owner_slug}_{dataset_slug}"

    env = os.environ.copy()
    kaggle_token = os.environ.get("KAGGLE_API_KEY", "")
    if kaggle_token:
        env["KAGGLE_API_TOKEN"] = kaggle_token

    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", dataset_id,
                "--unzip",
                "-p", str(dest_dir),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "403" in stderr or "401" in stderr:
                return (
                    f"**Auth Error:** Kaggle credentials not configured. "
                    f"Set KAGGLE_API_KEY in .env or place kaggle.json in ~/.kaggle/\n"
                    f"Details: {stderr}"
                )
            return f"**Download Error:** {stderr or result.stdout}"
    except FileNotFoundError:
        return "**Error:** kaggle CLI not found. Install with: pip install kaggle"
    except subprocess.TimeoutExpired:
        return f"**Error:** Download timed out after 300s for {dataset_id}"

    data_files = (
        list(dest_dir.rglob("*.csv"))
        + list(dest_dir.rglob("*.parquet"))
        + list(dest_dir.rglob("*.tsv"))
    )

    if not data_files:
        all_files = list(dest_dir.rglob("*"))
        file_list = ", ".join(f.name for f in all_files[:10])
        return (
            f"Downloaded {dataset_id} but found no CSV/Parquet/TSV files.\n"
            f"Files present: {file_list}"
        )

    lines = [f"## Downloaded: {dataset_id}", f"Location: `{dest_dir}`", ""]
    registered_refs = []

    for fp in sorted(data_files):
        ref = _safe_ref(f"kaggle_{owner_slug}_{dataset_slug}_{fp.stem}")

        try:
            df = _load_tabular_file(fp)
            register_dataset(ref, df, persist=True, register_sql=True)
            registered_refs.append(ref)

            cols = truncate_columns_display(list(df.columns), max_cols=8)
            lines.append(
                f"- **{fp.name}** → `{ref}` ({len(df):,} rows, {len(df.columns)} cols)"
            )
            lines.append(f"  Columns: {', '.join(cols)}")
        except Exception as e:
            lines.append(f"- **{fp.name}** — failed to load: {e}")

    lines.append("")
    lines.append(
        f"**{len(registered_refs)} file(s) registered.** "
        f"Use these refs with join_merge, transformations, or export_csv."
    )
    return "\n".join(lines)


# =============================================================================
# export_csv
# =============================================================================


class ExportCSVInput(BaseModel):
    dataset_ref: str = Field(
        description="Reference ID of the dataset to export (from download, join, or transformation)."
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Output file name (without path). Defaults to auto-generated name. "
        "Example: 'training_data.csv'",
    )


@tool(args_schema=ExportCSVInput)
def export_csv(dataset_ref: str, file_name: Optional[str] = None) -> str:
    """
    Export a registered dataset to a CSV file in the curated datasets directory.

    Use this as the final step to produce the training-ready CSV.
    The dataset_ref can come from download_kaggle_dataset, join_merge, or any transformation.

    Returns the file path and summary statistics.
    """
    df, err = _get_or_error(dataset_ref)
    if err:
        return err

    if file_name is None:
        file_name = f"{_safe_ref(dataset_ref)}.csv"

    if not file_name.endswith(".csv"):
        file_name += ".csv"

    output_path = CURATED_DIR / file_name
    df.to_csv(output_path, index=False)

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    lines = [
        f"## Exported: `{file_name}`",
        f"**Path:** `{output_path}`",
        f"**Shape:** {len(df):,} rows x {len(df.columns)} columns",
        f"**Size:** {output_path.stat().st_size / 1024:.1f} KB",
        "",
        "### Columns",
    ]
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = int(null_counts.get(col, 0))
        null_str = f" ({nulls} nulls)" if nulls > 0 else ""
        lines.append(f"- `{col}` ({dtype}){null_str}")

    if len(cols_with_nulls) > 0:
        lines.append(
            f"\n**Warning:** {len(cols_with_nulls)} column(s) contain null values."
        )

    return "\n".join(lines)


# =============================================================================
# normalize_columns
# =============================================================================


class NormalizeColumnsInput(BaseModel):
    dataset_ref: str = Field(
        description="Reference ID of the dataset whose columns to normalize."
    )


def _to_snake_case(name: str) -> str:
    """Convert any casing to clean snake_case."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[\s\-\.]+", "_", s)
    s = re.sub(r"[^\w]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()


@tool(args_schema=NormalizeColumnsInput)
def normalize_columns(dataset_ref: str) -> str:
    """
    Normalize all column names to clean snake_case
    (e.g. 'TransAmt' → 'trans_amt', 'First Name' → 'first_name').

    Call this after downloading a dataset and before export to ensure
    consistent, readable column names for downstream model training.
    """
    df, err = _get_or_error(dataset_ref)
    if err:
        return err

    old_names = list(df.columns)
    rename_map = {}

    for col in old_names:
        new_name = _to_snake_case(col)
        if new_name != col:
            rename_map[col] = new_name

    if not rename_map:
        return f"All {len(old_names)} columns already normalized. Ref: `{dataset_ref}`"

    # Handle collisions after renaming
    seen: dict[str, int] = {}
    final_map = {}
    for old, new in rename_map.items():
        if new in seen:
            seen[new] += 1
            final_map[old] = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
            final_map[old] = new

    result = df.rename(columns=final_map)
    ref = f"norm_{_safe_ref(dataset_ref)[:20]}_{uuid.uuid4().hex[:6]}"
    register_dataset(ref, result, persist=True, register_sql=True)

    lines = [
        f"## Normalized columns → `{ref}`",
        f"**{len(final_map)} of {len(old_names)} columns renamed:**",
    ]
    for old, new in list(final_map.items())[:15]:
        lines.append(f"  `{old}` → `{new}`")
    if len(final_map) > 15:
        lines.append(f"  ... +{len(final_map) - 15} more")

    return "\n".join(lines)


# =============================================================================
# profile_dataset
# =============================================================================


class ProfileDatasetInput(BaseModel):
    dataset_ref: str = Field(
        description="Reference ID of the dataset to profile."
    )


@tool(args_schema=ProfileDatasetInput)
def profile_dataset(dataset_ref: str) -> str:
    """
    Profile a registered dataset and report statistics useful for training:
    dtype distribution, null counts, cardinality, numeric ranges, and
    potential quality issues (constant columns, high-cardinality strings,
    columns that are mostly null).

    Call this after downloading/loading a dataset to understand its shape
    and quality before transforming and exporting.
    """
    df, err = _get_or_error(dataset_ref)
    if err:
        return err

    n_rows, n_cols = df.shape
    lines = [
        f"## Profile: `{dataset_ref}`",
        f"**Shape:** {n_rows:,} rows × {n_cols} columns",
        "",
    ]

    dtype_counts: dict[str, int] = {}
    for dt in df.dtypes:
        key = _classify_dtype(dt)
        dtype_counts[key] = dtype_counts.get(key, 0) + 1
    lines.append("### Dtype distribution")
    for dt, cnt in sorted(dtype_counts.items()):
        lines.append(f"  {dt}: {cnt}")
    lines.append("")

    lines.append("### Column details")
    lines.append(f"{'Column':<30} {'Type':<12} {'Nulls':>8} {'Unique':>8} {'Notes'}")
    lines.append(f"{'─' * 30} {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 30}")

    issues = []
    for col in df.columns:
        dtype_label = _classify_dtype(df[col].dtype)
        nulls = int(df[col].isnull().sum())
        null_pct = nulls / n_rows * 100 if n_rows > 0 else 0
        nunique = df[col].nunique()

        notes_parts = []
        if null_pct > 50:
            notes_parts.append(f"⚠ {null_pct:.0f}% null")
            issues.append(f"`{col}` is {null_pct:.0f}% null")
        elif null_pct > 0:
            notes_parts.append(f"{null_pct:.1f}% null")

        if nunique == 1:
            notes_parts.append("⚠ constant")
            issues.append(f"`{col}` is constant (single value)")
        elif nunique == n_rows and dtype_label == "string":
            notes_parts.append("unique-id?")

        if dtype_label in ("integer", "float") and nulls < n_rows:
            series = df[col].dropna()
            lo, hi = series.min(), series.max()
            notes_parts.append(f"range [{lo}, {hi}]")

        null_str = f"{nulls}" if nulls > 0 else "0"
        notes_str = " | ".join(notes_parts) if notes_parts else ""
        col_display = col[:28] + ".." if len(col) > 30 else col
        lines.append(f"{col_display:<30} {dtype_label:<12} {null_str:>8} {nunique:>8} {notes_str}")

    # Duplicates
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        dupe_pct = n_dupes / n_rows * 100
        lines.append(f"\n**Duplicates:** {n_dupes:,} ({dupe_pct:.1f}%)")
        issues.append(f"{n_dupes:,} duplicate rows ({dupe_pct:.1f}%)")

    # Summary issues
    if issues:
        lines.append(f"\n### Quality issues ({len(issues)})")
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("\n**No quality issues detected.**")

    return "\n".join(lines)


# =============================================================================
# validate_target
# =============================================================================


class ValidateTargetInput(BaseModel):
    dataset_ref: str = Field(
        description="Reference ID of the dataset to validate."
    )
    target_column: str = Field(
        description="Name of the column intended as the training target/label."
    )
    task_type: Optional[str] = Field(
        default=None,
        description=(
            "Expected ML task type: 'classification' or 'regression'. "
            "If omitted, the tool auto-detects from the column values."
        ),
    )


@tool(args_schema=ValidateTargetInput)
def validate_target(
    dataset_ref: str,
    target_column: str,
    task_type: Optional[str] = None,
) -> str:
    """
    Validate that a target column is suitable for model training.

    Checks:
    - Column exists and is not entirely null
    - Value distribution (class balance for classification, spread for regression)
    - Auto-detects classification vs regression if task_type is omitted
    - Warns about severe class imbalance, zero-variance, or suspicious patterns

    Call this before export to confirm the target variable is training-ready.
    """
    df, err = _get_or_error(dataset_ref)
    if err:
        return err

    if target_column not in df.columns:
        available = ", ".join(f"`{c}`" for c in df.columns[:15])
        return (
            f"**Error:** Column `{target_column}` not found.\n"
            f"Available: {available}"
        )

    series = df[target_column]
    n_total = len(series)
    n_null = int(series.isnull().sum())
    null_pct = n_null / n_total * 100 if n_total > 0 else 0
    valid = series.dropna()
    n_valid = len(valid)
    n_unique = valid.nunique()

    lines = [
        f"## Target validation: `{target_column}`",
        f"**Total rows:** {n_total:,} | **Nulls:** {n_null:,} ({null_pct:.1f}%)",
        f"**Unique values:** {n_unique:,}",
    ]

    warnings = []

    if n_valid == 0:
        lines.append("\n**FAIL:** Column is entirely null — cannot be used as target.")
        return "\n".join(lines)

    if null_pct > 5:
        warnings.append(
            f"Target has {null_pct:.1f}% null values — consider dropping or imputing."
        )

    # Auto-detect task type
    detected_type = None
    is_numeric = pd.api.types.is_numeric_dtype(valid)
    if is_numeric and n_unique <= 20:
        detected_type = "classification"
    elif is_numeric and n_unique > 20:
        detected_type = "regression"
    elif not is_numeric:
        detected_type = "classification"

    effective_type = task_type or detected_type
    lines.append(
        f"**Detected task:** {detected_type} "
        f"{'(matches specified)' if task_type == detected_type else ''}"
    )
    if task_type and task_type != detected_type:
        warnings.append(
            f"Specified task_type='{task_type}' but data looks like '{detected_type}'."
        )

    if effective_type == "classification":
        vc = valid.value_counts()
        lines.append(f"\n### Class distribution ({n_unique} classes)")
        for label, count in vc.head(10).items():
            pct = count / n_valid * 100
            bar = "█" * max(1, int(pct / 5))
            lines.append(f"  {label}: {count:,} ({pct:.1f}%) {bar}")
        if n_unique > 10:
            lines.append(f"  ... +{n_unique - 10} more classes")

        # Imbalance check
        majority_pct = vc.iloc[0] / n_valid * 100
        minority_pct = vc.iloc[-1] / n_valid * 100
        ratio = vc.iloc[0] / vc.iloc[-1] if vc.iloc[-1] > 0 else float("inf")

        if ratio > 100:
            warnings.append(
                f"Severe class imbalance ({ratio:.0f}:1). "
                f"Majority class is {majority_pct:.1f}%. "
                f"Consider stratified sampling, SMOTE, or class weights."
            )
        elif ratio > 10:
            warnings.append(
                f"Moderate class imbalance ({ratio:.1f}:1). "
                f"Majority: {majority_pct:.1f}%, minority: {minority_pct:.1f}%."
            )

    elif effective_type == "regression":
        desc = valid.describe()
        lines.append("\n### Value distribution")
        lines.append(f"  Mean:   {desc['mean']:.4g}")
        lines.append(f"  Std:    {desc['std']:.4g}")
        lines.append(f"  Min:    {desc['min']:.4g}")
        lines.append(f"  25%:    {desc['25%']:.4g}")
        lines.append(f"  Median: {desc['50%']:.4g}")
        lines.append(f"  75%:    {desc['75%']:.4g}")
        lines.append(f"  Max:    {desc['max']:.4g}")

        if desc["std"] == 0:
            warnings.append("Zero variance — column has a single value. Not useful as target.")
        elif desc["std"] / abs(desc["mean"]) > 10 if desc["mean"] != 0 else False:
            warnings.append("Very high coefficient of variation — check for outliers.")

    # Final verdict
    if not warnings:
        lines.append(f"\n**PASS:** `{target_column}` looks good for {effective_type}.")
    else:
        lines.append(f"\n### Warnings ({len(warnings)})")
        for w in warnings:
            lines.append(f"  ⚠ {w}")
        severity = "INFO" if len(warnings) <= 1 else "REVIEW"
        lines.append(f"\n**{severity}:** Target is usable but review warnings above.")

    return "\n".join(lines)


# =============================================================================
# suggest_join_keys
# =============================================================================


class SuggestJoinKeysInput(BaseModel):
    left_ref: str = Field(description="Reference ID of the first dataset.")
    right_ref: str = Field(description="Reference ID of the second dataset.")


@tool(args_schema=SuggestJoinKeysInput)
def suggest_join_keys(left_ref: str, right_ref: str) -> str:
    """
    Compare two registered datasets and suggest columns suitable for joining.

    Checks for exact column name matches, similar names (fuzzy), and
    compatible dtypes/value overlap between candidate key pairs.
    Use this before join_merge_tool to pick the right key columns.
    """
    left, err = _get_or_error(left_ref)
    if err:
        return err
    right, err = _get_or_error(right_ref)
    if err:
        return err

    lines = [
        f"## Join key suggestions: `{left_ref}` ↔ `{right_ref}`",
        f"Left: {len(left):,} rows × {len(left.columns)} cols | "
        f"Right: {len(right):,} rows × {len(right.columns)} cols",
        "",
    ]

    exact = sorted(set(left.columns) & set(right.columns))
    candidates = []

    if exact:
        lines.append(f"### Exact column matches ({len(exact)})")
        for col in exact:
            l_uniq = left[col].nunique()
            r_uniq = right[col].nunique()
            l_dtype = str(left[col].dtype)
            r_dtype = str(right[col].dtype)

            l_set = set(left[col].dropna().unique())
            r_set = set(right[col].dropna().unique())
            overlap = len(l_set & r_set)
            smaller = min(len(l_set), len(r_set))
            overlap_pct = overlap / smaller * 100 if smaller > 0 else 0

            quality = "strong" if overlap_pct > 50 else ("weak" if overlap_pct > 10 else "poor")
            candidates.append((col, col, overlap_pct, quality))

            lines.append(
                f"  `{col}` — {l_dtype}/{r_dtype} | "
                f"unique {l_uniq:,}/{r_uniq:,} | "
                f"overlap {overlap:,} ({overlap_pct:.0f}%) → **{quality}**"
            )
        lines.append("")

    left_only = set(left.columns) - set(exact)
    right_only = set(right.columns) - set(exact)
    fuzzy_pairs = []
    for lc in left_only:
        matches = get_close_matches(lc.lower(), [rc.lower() for rc in right_only], n=1, cutoff=0.7)
        if matches:
            rc = next(rc for rc in right_only if rc.lower() == matches[0])
            fuzzy_pairs.append((lc, rc))

    if fuzzy_pairs:
        lines.append(f"### Similar column names ({len(fuzzy_pairs)})")
        for lc, rc in fuzzy_pairs:
            l_dtype = str(left[lc].dtype)
            r_dtype = str(right[rc].dtype)
            lines.append(f"  `{lc}` ↔ `{rc}` ({l_dtype}/{r_dtype})")
        lines.append("")

    # Recommendation
    strong = [c for c in candidates if c[3] == "strong"]
    if strong:
        best = strong[0]
        lines.append(f"**Recommendation:** Join on `{best[0]}` ({best[2]:.0f}% value overlap).")
        if len(strong) > 1:
            lines.append(f"Also strong: {', '.join(f'`{c[0]}`' for c in strong[1:])}")
    elif candidates:
        best = max(candidates, key=lambda c: c[2])
        lines.append(
            f"**Best available:** `{best[0]}` ({best[2]:.0f}% overlap, {best[3]}). "
            f"Low overlap may indicate these datasets don't share the same entities."
        )
    elif fuzzy_pairs:
        lc, rc = fuzzy_pairs[0]
        lines.append(
            f"**No exact matches.** Try joining on `{lc}` (left) = `{rc}` (right) "
            f"using a key mapping dict."
        )
    else:
        lines.append("**No joinable columns found.** These datasets likely can't be joined directly.")

    return "\n".join(lines)


# =============================================================================
# download_hf_dataset
# =============================================================================


class DownloadHFDatasetInput(BaseModel):
    dataset_id: str = Field(
        description="HuggingFace dataset identifier (e.g. 'imdb', 'stanfordnlp/imdb', 'scikit-learn/iris')."
    )
    split: Optional[str] = Field(
        default="train",
        description="Which split to load (e.g. 'train', 'test', 'validation'). Defaults to 'train'.",
    )
    config_name: Optional[str] = Field(
        default=None,
        description="Dataset configuration/subset name, if the dataset has multiple configs.",
    )


@tool(args_schema=DownloadHFDatasetInput)
def download_hf_dataset(
    dataset_id: str,
    split: Optional[str] = "train",
    config_name: Optional[str] = None,
) -> str:
    """
    Download a HuggingFace dataset, convert it to a pandas DataFrame, and
    register it for use with other tools (join, transform, export, etc.).

    Use the HuggingFace MCP tools (hf_dataset_search, hf_list_datasets) to
    discover datasets first, then call this with the dataset ID from the results.

    Example: download_hf_dataset(dataset_id="scikit-learn/iris", split="train")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return "**Error:** `datasets` library not installed. Run: pip install datasets"

    try:
        kwargs: dict = {"trust_remote_code": True}
        if config_name:
            kwargs["name"] = config_name
        if split:
            kwargs["split"] = split

        ds = load_dataset(dataset_id, **kwargs)

        if hasattr(ds, "to_pandas"):
            df = ds.to_pandas()
        else:
            first_key = list(ds.keys())[0]
            df = ds[first_key].to_pandas()
    except Exception as e:
        return f"**Error downloading** `{dataset_id}`: {e}"

    for col in df.columns:
        if df[col].apply(type).isin({dict, list}).any():
            df[col] = df[col].apply(
                lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list)) else v
            )

    ref = f"hf_{_safe_ref(dataset_id)}_{split or 'all'}"

    register_dataset(ref, df, persist=True, register_sql=True)

    cols = truncate_columns_display(list(df.columns), max_cols=8)
    lines = [
        f"## Downloaded from HuggingFace: {dataset_id}",
        f"Split: {split or 'all'} | Config: {config_name or 'default'}",
        f"**Registered as:** `{ref}` ({len(df):,} rows, {len(df.columns)} cols)",
        f"Columns: {', '.join(cols)}",
    ]
    return "\n".join(lines)


curator_tools = [
    download_kaggle_dataset,
    download_hf_dataset,
    export_csv,
    normalize_columns,
    profile_dataset,
    validate_target,
    suggest_join_keys,
]
