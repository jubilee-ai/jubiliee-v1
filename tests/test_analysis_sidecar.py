"""Round-trip tests for analysis tool JSON sidecars (no LLM)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load_analysis_sidecar():
    """Load ``analysis_sidecar.py`` without importing the ``analysis`` package."""
    path = _ROOT / "tools/data-tools/analysis/analysis_sidecar.py"
    spec = importlib.util.spec_from_file_location("analysis_sidecar_standalone", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_attach_and_extract_roundtrip():
    m = _load_analysis_sidecar()
    md = m.attach_analysis_sidecar(
        "# Report\n\nhello",
        kind="correlation",
        tool="correlation_matrix_tool",
        summary="top pairs",
        payload={"top_pairs": [{"col1": "a", "col2": "b", "correlation": 0.5}]},
    )
    cleaned, envs = m.extract_analysis_json_sidecars(md)
    assert "<ANALYSIS_JSON>" not in cleaned
    assert "# Report" in cleaned
    assert len(envs) == 1
    assert envs[0]["kind"] == "correlation"
    assert envs[0]["tool"] == "correlation_matrix_tool"
    assert envs[0]["payload"]["top_pairs"][0]["col1"] == "a"


def test_chart_tool_build_spec_histogram():
    pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    pytest.importorskip("langchain_core")

    pd = pytest.importorskip("pandas")
    sys.path.insert(0, str(_ROOT))
    from utils import clear_dataset_registry, register_dataset

    _DT = _ROOT / "tools/data-tools"
    sys.path.insert(0, str(_DT))

    chart_path = _ROOT / "tools/data-tools/analysis/chart.py"
    spec = importlib.util.spec_from_file_location("chart_tool_standalone", chart_path)
    assert spec and spec.loader
    chart_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = chart_mod
    spec.loader.exec_module(chart_mod)

    clear_dataset_registry()
    register_dataset(
        "t_insurance",
        pd.DataFrame(
            {
                "region": ["ne", "ne", "sw"],
                "charges": [100.0, 200.0, 150.0],
                "age": [30, 40, 35],
            }
        ),
    )
    md, spec_out = chart_mod.build_chart_spec(
        dataset_ref="t_insurance",
        chart_type="histogram",
        x="charges",
        y=None,
        group_by=None,
        agg="mean",
        title="T",
        top_n=10,
        n_bins=5,
    )
    assert md
    assert spec_out.get("type") == "histogram"
