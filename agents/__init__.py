"""Agents package - Analysis and orchestration agents."""

from .model_index import get_model_by_name, get_model_index

# Note: data_retrieval is in data-retrieval/ folder (with hyphen)
# Import it lazily or directly from the submodule when needed

# Main chat orchestrator lives in ``agent.py`` at the project root.

__all__ = [
    # Model index
    "get_model_index",
    "get_model_by_name",
]
