"""
Utilities for the ML Training Agent.

This module contains:
- streaming.py: Streaming functions for real-time updates
- prompts.py: System prompts for agents
"""

# Import prompts directly (no circular dependency)
from .prompts import (
    TRAINING_SYSTEM_PROMPT,
    FEATURE_ENGINEERING_SIMPLE_SYSTEM_PROMPT,
)

# Note: streaming module imports are done lazily to avoid circular imports
# Use: from agents.training.utils.streaming import stream_training_agent
# Or: from agents.training.agent import stream_training_agent

__all__ = [
    # Prompts
    "TRAINING_SYSTEM_PROMPT",
    "FEATURE_ENGINEERING_SIMPLE_SYSTEM_PROMPT",
]
