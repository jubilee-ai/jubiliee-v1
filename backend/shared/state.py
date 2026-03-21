"""
In-memory state kept only for ephemeral per-process caches and constant mappings.
Persistent state now lives in PostgreSQL via backend.shared.models.
"""

TOOL_TO_STEP = {
    "tool_data_collection": "data_collection",
    "tool_select_model": "select_model",
    "tool_cleaning": "cleaning",
    "tool_label_split_definition": "label_split_definition",
    "tool_feature_selection_specification": "feature_selection_specification",
    "tool_feature_engineering_executor": "feature_engineering_executor",
    "tool_training_approval": "training_approval",
    "tool_training": "training",
    "tool_generate_report": "generate_report",
}

# Kept for backward-compat imports; code should migrate to DB-backed repository.
training_jobs: dict[str, dict[str, object]] = {}
simple_agent_store: dict[str, dict[str, object]] = {}
last_training_context: dict[str, object] = {}
chat_threads_with_context: set[str] = set()
