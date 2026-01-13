"""
Data tools for catalog search, retrieval, data loading, SQL queries, and transformations.
"""

from .data_loader import (ColumnSchema, DatasetResult, data_loader_tools,
                          dataset_get, dataset_get_tool)
from .embedder import OpenAIEmbedder
from .join_merge import (JoinReport, JoinResult, join_merge, join_merge_tool,
                         join_merge_tools)
from .retrieval import (CatalogSearcher, SearchFilters, SearchResult,
                        catalog_search, catalog_search_tool, catalog_tools,
                        hybrid_search, list_datasets_tool, search)
from .sql_query import (QueryResult, SQLWarehouse, describe_table, get_schema,
                        get_sql_schema_tool, list_tables, sql_query,
                        sql_query_tool, sql_tools)
from .utils import (  # Text utilities; Fuzzy matching; Type conversions; ID and timestamp; Catalog access; Schema utilities; Paths; Dataset registry (shared across all tools); Unified dataset discovery
    CATALOG_PATH, DATASETS_DIR, DERIVED_DATASETS_DIR, SQL_DIR,
    build_schema_from_dataframe, clear_registry, cosine_similarity,
    fuzzy_suggest, generate_unique_id, get_all_available_datasets,
    get_catalog, get_dataset_info, get_openai_client, get_registered_dataset,
    get_registered_dataset_info, get_similar_matches, infer_simple_dtype,
    list_all_dataset_refs, list_catalog_assets, list_derived_datasets,
    list_registered_datasets, pandas_dtype_to_sql_type, register_dataset,
    tokenize_text, truncate_columns_display, utc_timestamp)

# Transformations
from .transformations import (
    # Base types
    BaseTransform,
    TransformResult,
    TransformAudit,
    TransformWarning,
    SchemaChange,
    # Transform classes
    SelectTransform,
    DropTransform,
    RenameTransform,
    CastTransform,
    ParseDatetimeTransform,
    AddColumnTransform,
    # Factory functions
    select,
    drop,
    rename,
    cast,
    parse_datetime,
    add_column,
    # Registry
    TRANSFORM_REGISTRY,
    TRANSFORM_DOCS,
    get_transform,
    list_transforms,
    get_transform_docs,
)
from .transformations.transform_apply import (
    transform_apply,
    transform_apply_tool,
    list_transforms_tool,
    transform_tools,
    TransformApplyResult,
    TransformLineage,
)

__all__ = [
    # Utils - Text
    "tokenize_text",
    "cosine_similarity",
    "get_openai_client",
    # Utils - Fuzzy matching
    "fuzzy_suggest",
    "get_similar_matches",
    # Utils - Type conversions
    "pandas_dtype_to_sql_type",
    "infer_simple_dtype",
    # Utils - ID and timestamp
    "generate_unique_id",
    "utc_timestamp",
    # Utils - Catalog access
    "get_catalog",
    "get_dataset_info",
    "list_catalog_assets",
    # Utils - Schema
    "build_schema_from_dataframe",
    "truncate_columns_display",
    # Utils - Paths
    "DATASETS_DIR",
    "CATALOG_PATH",
    "SQL_DIR",
    "DERIVED_DATASETS_DIR",
    # Utils - Dataset registry (shared infrastructure)
    "register_dataset",
    "get_registered_dataset",
    "list_registered_datasets",
    "clear_registry",
    "get_registered_dataset_info",
    "list_derived_datasets",
    # Utils - Unified dataset discovery
    "get_all_available_datasets",
    "list_all_dataset_refs",
    # Embedder
    "OpenAIEmbedder",
    # Retrieval / Dataset Discovery
    "catalog_search",
    "hybrid_search",
    "search",
    "CatalogSearcher",
    "SearchFilters",
    "SearchResult",
    "catalog_search_tool",
    "list_datasets_tool",
    "catalog_tools",
    # Data Loader
    "dataset_get",
    "dataset_get_tool",
    "data_loader_tools",
    "DatasetResult",
    "ColumnSchema",
    # SQL Query
    "sql_query",
    "sql_query_tool",
    "get_sql_schema_tool",
    "get_schema",
    "list_tables",
    "describe_table",
    "sql_tools",
    "QueryResult",
    "SQLWarehouse",
    # Join Merge
    "join_merge",
    "join_merge_tool",
    "join_merge_tools",
    "JoinResult",
    "JoinReport",
    # Transformations - Base types
    "BaseTransform",
    "TransformResult",
    "TransformAudit",
    "TransformWarning",
    "SchemaChange",
    # Transformations - Transform classes
    "SelectTransform",
    "DropTransform",
    "RenameTransform",
    "CastTransform",
    "ParseDatetimeTransform",
    "AddColumnTransform",
    # Transformations - Factory functions
    "select",
    "drop",
    "rename",
    "cast",
    "parse_datetime",
    "add_column",
    # Transformations - Registry
    "TRANSFORM_REGISTRY",
    "TRANSFORM_DOCS",
    "get_transform",
    "list_transforms",
    "get_transform_docs",
    # Transformations - transform_apply
    "transform_apply",
    "transform_apply_tool",
    "list_transforms_tool",
    "transform_tools",
    "TransformApplyResult",
    "TransformLineage",
]

