"""Agents package - Analysis and orchestration agents."""

from .analysis_agent import run_analysis
from .model_index import get_model_by_name, get_model_index
from .subagent import SubagentState, build_subagent_graph, run_subagent

from .data_retrieval import (
    DATA_RETRIEVAL_TOOLS,
    DataRetrievalResult,
    build_data_retrieval_agent,
    get_dataset,
    is_dataset_available,
    list_available_datasets,
    retrieve_data,
)

__all__ = [
    # Analysis
    "run_analysis",
    "run_subagent",
    "build_subagent_graph",
    "SubagentState",
    # Model index
    "get_model_index",
    "get_model_by_name",
    # Data retrieval
    "retrieve_data",
    "build_data_retrieval_agent",
    "DATA_RETRIEVAL_TOOLS",
    "DataRetrievalResult",
    "get_dataset",
    "list_available_datasets",
    "is_dataset_available",
]
