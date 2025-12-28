"""
sql_query - Execute validated SQL queries against the data warehouse.

Inputs: query, params?, dialect?, materialize?, max_rows?
Outputs: {dataset_ref, schema, stats(rows_scanned/runtime), query_id, data}

Uses:
- SQLAlchemy for connection management, schema introspection, and query validation
- Pandas for query execution and result handling
"""

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from .utils import (SQL_DIR, build_schema_from_dataframe, fuzzy_suggest,
                    generate_unique_id, get_similar_matches,
                    truncate_columns_display, utc_timestamp)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_MAX_ROWS = 1000
ABSOLUTE_MAX_ROWS = 10000

# Forbidden SQL patterns (case-insensitive)
FORBIDDEN_PATTERNS = [
    r'\bDROP\b', r'\bDELETE\b', r'\bTRUNCATE\b', r'\bINSERT\b',
    r'\bUPDATE\b', r'\bALTER\b', r'\bCREATE\b', r'\bGRANT\b',
    r'\bREVOKE\b', r'\bEXEC\b', r'\bEXECUTE\b', r'\bATTACH\b',
    r'\bDETACH\b', r';\s*\w',  # Multiple statements
]


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TableInfo:
    """Schema information for a table."""
    name: str
    columns: list[dict] = field(default_factory=list)
    row_count: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {"table": self.name, "columns": self.columns, "row_count": self.row_count}
    
    def column_names(self) -> list[str]:
        return [c["name"] for c in self.columns]


@dataclass
class QueryResult:
    """Result from sql_query execution."""
    query_id: str
    dataset_ref: str
    schema: list[dict]
    data: list[dict]
    stats: dict
    query_info: dict
    
    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "dataset_ref": self.dataset_ref,
            "schema": self.schema,
            "data": self.data,
            "stats": self.stats,
            "query_info": self.query_info,
        }


@dataclass
class QueryValidation:
    """Result of query validation."""
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    
    def add_error(self, msg: str):
        self.is_valid = False
        self.errors.append(msg)


# =============================================================================
# WAREHOUSE CLASS (SQLAlchemy + Pandas)
# =============================================================================

class SQLWarehouse:
    """SQL warehouse using SQLAlchemy for connections and Pandas for queries."""
    
    def __init__(self, sql_dir: Path = SQL_DIR):
        self.engine = create_engine("sqlite:///:memory:", echo=False)
        self.tables: dict[str, TableInfo] = {}
        
        # Load all SQL files
        for sql_file in sql_dir.glob("*.sql"):
            try:
                sql_content = sql_file.read_text()
                with self.engine.connect() as conn:
                    conn.execute(text("BEGIN"))
                    for stmt in sql_content.split(';'):
                        if stmt.strip():
                            try:
                                conn.execute(text(stmt))
                            except SQLAlchemyError:
                                pass
                    conn.execute(text("COMMIT"))
            except Exception as e:
                print(f"Warning: Failed to load {sql_file.name}: {e}")
        
        # Cache schema using SQLAlchemy inspect
        inspector = inspect(self.engine)
        for table_name in inspector.get_table_names():
            columns = [
                {"name": c["name"], "type": str(c["type"]), "nullable": c.get("nullable", True)}
                for c in inspector.get_columns(table_name)
            ]
            try:
                row_count = int(pd.read_sql(f"SELECT COUNT(*) as cnt FROM {table_name}", self.engine)["cnt"].iloc[0])
            except:
                row_count = None
            self.tables[table_name.lower()] = TableInfo(name=table_name, columns=columns, row_count=row_count)
    
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        return self.tables.get(table_name.lower())
    
    def get_all_tables(self) -> list[TableInfo]:
        return list(self.tables.values())
    
    def execute_query(self, query: str, params: Optional[dict] = None, max_rows: int = DEFAULT_MAX_ROWS) -> tuple[pd.DataFrame, float]:
        """Execute query using Pandas. Returns (DataFrame, execution_time)."""
        # Add LIMIT if not present
        if "LIMIT" not in query.upper() and query.strip().upper().startswith("SELECT"):
            query = f"{query.rstrip(';')} LIMIT {max_rows}"
        
        start_time = time.time()
        df = pd.read_sql(text(query), self.engine, params=params) if params else pd.read_sql(text(query), self.engine)
        return df, time.time() - start_time


# =============================================================================
# VALIDATION (Uses SQLAlchemy text() for safe parsing)
# =============================================================================

def validate_query(query: str, warehouse: SQLWarehouse, params: Optional[dict] = None) -> QueryValidation:
    """Validate SQL query for safety and correctness."""
    validation = QueryValidation()
    query_upper = query.strip().upper()
    
    # Check 1: Must start with SELECT or WITH
    if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
        validation.add_error("Only SELECT queries allowed. INSERT, UPDATE, DELETE, and DDL are blocked.")
        return validation
    
    # Check 2: Forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            validation.add_error("Query contains forbidden operation. Only read-only SELECT allowed.")
            return validation
    
    # Check 3: Validate table names exist
    table_pattern = r'\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)|JOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    referenced_tables = {(m[0] or m[1]).lower() for m in re.findall(table_pattern, query, re.IGNORECASE)}
    available_tables = list(warehouse.tables.keys())
    
    for table in referenced_tables:
        if table not in available_tables:
            validation.add_error(f"Table '{table}' does not exist.")
            validation.suggestions.append(fuzzy_suggest(table, available_tables))
    
    # Check 4: Validate parameters
    placeholders = set(re.findall(r':([a-zA-Z_][a-zA-Z0-9_]*)', query))
    if placeholders and not params:
        validation.add_error(f"Query has parameters ({', '.join(placeholders)}) but none provided.")
    elif params:
        missing = placeholders - set(params.keys())
        if missing:
            validation.add_error(f"Missing parameter values: {', '.join(missing)}")
    
    if not validation.is_valid:
        validation.suggestions.append("Use get_schema() to see available tables and columns.")
    
    return validation


# =============================================================================
# SCHEMA HELPERS
# =============================================================================

def get_table_sample(warehouse: SQLWarehouse, table_name: str, n: int = 3) -> list[dict]:
    """Get sample rows from a table."""
    table = warehouse.get_table_info(table_name)
    if not table:
        return []
    
    try:
        df, _ = warehouse.execute_query(f"SELECT * FROM {table.name} LIMIT {n}")
        return df.to_dict(orient="records")
    except:
        return []


# =============================================================================
# MAIN QUERY FUNCTION
# =============================================================================

_warehouse: Optional[SQLWarehouse] = None


def get_warehouse() -> SQLWarehouse:
    """Get or create global warehouse instance."""
    global _warehouse
    if _warehouse is None:
        _warehouse = SQLWarehouse()
    return _warehouse


def sql_query(
    query: str,
    params: Optional[dict] = None,
    dialect: Literal["sqlite"] = "sqlite",
    materialize: bool = False,
    max_rows: int = DEFAULT_MAX_ROWS,
    validate: bool = True,
) -> QueryResult:
    """
    Execute a SQL query against the data warehouse.
    
    Args:
        query: SQL SELECT query
        params: Named parameters (e.g., {"age": 30} for :age placeholder)
        dialect: SQL dialect (currently 'sqlite')
        materialize: Save results as table (not implemented)
        max_rows: Max rows to return (default 1000, max 10000)
        validate: Validate query before execution
    
    Returns:
        QueryResult with data, schema, and stats
    """
    warehouse = get_warehouse()
    max_rows = min(max(1, max_rows), ABSOLUTE_MAX_ROWS)
    
    query_id = generate_unique_id("q", query)
    
    # Validate
    if validate:
        validation = validate_query(query, warehouse, params)
        if not validation.is_valid:
            error_msg = " ".join(validation.errors)
            if validation.suggestions:
                error_msg += " Suggestions: " + " ".join(validation.suggestions)
            raise ValueError(error_msg)
    
    # Execute with Pandas
    try:
        df, execution_time = warehouse.execute_query(query, params, max_rows)
    except SQLAlchemyError as e:
        error_str = str(e)
        suggestion = ""
        
        if "no such table" in error_str:
            match = re.search(r"no such table: (\w+)", error_str)
            if match:
                suggestion = fuzzy_suggest(match.group(1), list(warehouse.tables.keys()))
        elif "no such column" in error_str:
            match = re.search(r"no such column: (\w+)", error_str)
            if match:
                # Dedupe columns from all tables
                all_cols = list(set(c["name"] for t in warehouse.tables.values() for c in t.columns))
                suggestion = fuzzy_suggest(match.group(1), all_cols)
        
        raise ValueError(f"SQL Error: {error_str} {suggestion}") from e
    
    # Build schema from DataFrame dtypes
    schema = build_schema_from_dataframe(df)
    
    # Convert to records
    data = df.to_dict(orient="records")
    
    return QueryResult(
        query_id=query_id,
        dataset_ref=f"query_result:{query_id}",
        schema=schema,
        data=data,
        stats={
            "rows_returned": len(data),
            "columns_returned": len(df.columns),
            "execution_time_ms": round(execution_time * 1000, 2),
            "max_rows_limit": max_rows,
            "truncated": len(data) >= max_rows,
        },
        query_info={
            "query": query,
            "params": params,
            "dialect": dialect,
            "executed_at": utc_timestamp(),
        },
    )


def get_schema() -> dict:
    """Get full warehouse schema as dict."""
    warehouse = get_warehouse()
    tables = []
    for t in warehouse.get_all_tables():
        col_names = [c["name"] for c in t.columns]
        cols_display = truncate_columns_display(col_names)
        tables.append({"name": t.name, "rows": t.row_count, "columns": cols_display})
    return {"tables": tables}


def list_tables() -> list[str]:
    """List all available table names."""
    return [t.name for t in get_warehouse().get_all_tables()]


def describe_table(table_name: str) -> dict:
    """Get detailed info for a table."""
    warehouse = get_warehouse()
    table = warehouse.get_table_info(table_name)
    if not table:
        available = list(warehouse.tables.keys())
        raise ValueError(f"Table '{table_name}' not found. {fuzzy_suggest(table_name, available)}")
    return table.to_dict()


# =============================================================================
# LANGCHAIN TOOLS
# =============================================================================

class SQLQueryInput(BaseModel):
    """Input for sql_query tool."""
    query: str = Field(description="SQL SELECT query. Use :param_name for parameters. Example: 'SELECT * FROM loan_default WHERE Age > :min_age LIMIT 100'")
    params: Optional[dict] = Field(default=None, description="Parameters for query (e.g., {'min_age': 30})")
    max_rows: int = Field(default=100, description="Max rows to return (default 100, max 10000)")


class GetSchemaInput(BaseModel):
    """Input for get_schema tool."""
    table_name: Optional[str] = Field(default=None, description="Specific table to describe, or None for all tables")


@tool(args_schema=SQLQueryInput)
def sql_query_tool(query: str, params: Optional[dict] = None, max_rows: int = 100) -> str:
    """
    Execute a SQL SELECT query against the data warehouse.
    
    IMPORTANT: Call get_sql_schema_tool() FIRST to see available tables and columns.
    
    Features:
    - Only read-only SELECT queries allowed
    - Parameterized queries with :param_name syntax
    - Fuzzy matching suggests correct names on typos
    - Automatic row limits
    
    Returns JSON with: columns, data, rows_returned, execution_time_ms, truncated
    """
    try:
        result = sql_query(query=query, params=params, max_rows=max_rows)
        
        # Return clean, parseable JSON
        import json
        return json.dumps({
            "columns": [c["name"] for c in result.schema],
            "data": result.data,
            "rows_returned": result.stats["rows_returned"],
            "execution_time_ms": result.stats["execution_time_ms"],
            "truncated": result.stats["truncated"],
        }, default=str)
    except ValueError as e:
        return f'{{"error": "{e}"}}'
    except Exception as e:
        return f'{{"error": "{type(e).__name__}: {e}"}}'


@tool(args_schema=GetSchemaInput)
def get_sql_schema_tool(table_name: Optional[str] = None) -> str:
    """
    Get schema of available SQL tables. CALL THIS FIRST before writing queries.
    
    Returns JSON with table names, columns, types, and row counts.
    """
    import json
    try:
        warehouse = get_warehouse()
        
        if table_name:
            table = warehouse.get_table_info(table_name)
            if not table:
                available = list(warehouse.tables.keys())
                return json.dumps({"error": f"Table '{table_name}' not found. {fuzzy_suggest(table_name, available)}"})
            
            # Get sample data
            try:
                sample_df, _ = warehouse.execute_query(f"SELECT * FROM {table.name} LIMIT 3")
                sample = sample_df.to_dict(orient="records")
            except:
                sample = []
            
            return json.dumps({
                "table": table.name,
                "rows": table.row_count,
                "columns": [{"name": c["name"], "type": c["type"]} for c in table.columns],
                "sample": sample,
            }, default=str)
        
        # All tables - truncate columns if > 10
        tables_info = []
        for t in warehouse.get_all_tables():
            col_names = [c["name"] for c in t.columns]
            cols_display = truncate_columns_display(col_names)
            tables_info.append({"name": t.name, "rows": t.row_count, "columns": cols_display})
        
        return json.dumps({"tables": tables_info}, default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


sql_tools = [sql_query_tool, get_sql_schema_tool]

__all__ = [
    "sql_query", "sql_query_tool", "get_sql_schema_tool",
    "get_schema", "list_tables", "describe_table",
    "sql_tools", "QueryResult", "SQLWarehouse",
]
