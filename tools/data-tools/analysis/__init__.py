"""Analysis tools for AI agents - validation, EDA, diagnostics, correlation, inferential tests, regression summaries, grouping, distribution, trends, concentration."""

from .chart import chart_tool
from .concentration_analysis import (analyze_concentration,
                                     concentration_analysis_tool)
from .correlation_matrix import (compute_correlation_matrix,
                                 correlation_matrix_tool)
from .data_validation import data_validation_tool, validate_dataset
from .distribution_analysis import (analyze_distribution,
                                    distribution_analysis_tool)
from .eda_report import eda_report_tool, run_eda_report
from .feature_diagnostics import (feature_diagnostics_tool,
                                  run_feature_diagnostics)
from .group_summary import compute_group_summary, group_summary_tool
from .hypothesis_tests import (categorical_association_test_tool,
                               group_comparison_test_tool,
                               run_categorical_association,
                               run_group_comparison, run_inferential_feature_scan)
from .statistical_models import regression_summary_tool, run_ols_summary
from .trend_analysis import analyze_trend, trend_analysis_tool

analysis_tools = [
    data_validation_tool,
    eda_report_tool,
    feature_diagnostics_tool,
    correlation_matrix_tool,
    group_summary_tool,
    distribution_analysis_tool,
    trend_analysis_tool,
    concentration_analysis_tool,
    group_comparison_test_tool,
    categorical_association_test_tool,
    regression_summary_tool,
    chart_tool,
]

__all__ = [
    "analysis_tools",
    "data_validation_tool",
    "validate_dataset",
    "eda_report_tool",
    "run_eda_report",
    "feature_diagnostics_tool",
    "run_feature_diagnostics",
    "correlation_matrix_tool",
    "compute_correlation_matrix",
    "group_summary_tool",
    "compute_group_summary",
    "distribution_analysis_tool",
    "analyze_distribution",
    "trend_analysis_tool",
    "analyze_trend",
    "concentration_analysis_tool",
    "analyze_concentration",
    "group_comparison_test_tool",
    "categorical_association_test_tool",
    "regression_summary_tool",
    "run_group_comparison",
    "run_categorical_association",
    "run_inferential_feature_scan",
    "run_ols_summary",
    "chart_tool",
]

