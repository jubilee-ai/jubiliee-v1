"""
ML Model Training Agent Package

This package provides a LangGraph-based training agent for ML model development.

Structure:
- agent.py: Main entry point and public API
- core/: Graph infrastructure (state, edges, hitl, graph)
- steps/: Pipeline step implementations
- utils/: Utilities (streaming, prompts)
"""

# Re-export everything from agent.py for backward compatibility
from .agent import *
