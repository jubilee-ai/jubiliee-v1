"""
catalog_search - Search internal data catalog to discover relevant datasets/tables/features.

Inputs: query, filters (domain/source/format/columns), top_k
Outputs: ranked list of {asset_id, name, description, schema_summary, format, source, tags, score}

Uses:
- bm25s for fast lexical search: https://github.com/xhluca/bm25s
- OpenAI text-embedding-3-small for semantic search: https://platform.openai.com/docs/api-reference/embeddings
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bm25s

try:
    from .embedder import OpenAIEmbedder
    from .utils import (CATALOG_PATH, cosine_similarity,
                        get_all_available_datasets, get_catalog, tokenize_text)
except ImportError:
    from embedder import OpenAIEmbedder
    from utils import (CATALOG_PATH, cosine_similarity,
                       get_all_available_datasets, get_catalog, tokenize_text)


@dataclass
class SearchFilters:
    """Filters for catalog search."""
    domain: Optional[str] = None          # e.g., "credit_risk", "insurance", "financial_health"
    source: Optional[str] = None          # e.g., "HuggingFace", local file
    format: Optional[str] = None          # e.g., "CSV", "Python (HuggingFace loader)"
    columns: Optional[list[str]] = None   # columns that must be present
    min_rows: Optional[int] = None        # minimum number of rows


@dataclass
class SearchResult:
    """A single search result from catalog search."""
    asset_id: str
    name: str
    description: str
    schema_summary: list[str]
    format: str
    source: Optional[str]
    tags: list[str]
    score: float
    use_case: str
    rows: Optional[int] = None
    # Score breakdown for debugging
    bm25_score: Optional[float] = None
    column_score: Optional[float] = None
    semantic_score: Optional[float] = None

    def to_dict(self) -> dict:
        result = {
            "asset_id": self.asset_id,
            "name": self.name,
            "description": self.description,
            "schema_summary": self.schema_summary,
            "format": self.format,
            "source": self.source,
            "tags": self.tags,
            "score": self.score,
            "use_case": self.use_case,
            "rows": self.rows,
        }
        # Include score breakdown if available
        if self.bm25_score is not None:
            result["score_breakdown"] = {
                "bm25": self.bm25_score,
                "column": self.column_score,
                "semantic": self.semantic_score,
            }
        return result


class CatalogSearcher:
    """
    Hybrid search engine for the data catalog.
    
    Supports:
    - BM25 lexical search on descriptions and use cases (via bm25s)
    - Semantic search using OpenAI text-embedding-3-small
    - Column matching (exact and partial)
    - Metadata filters (domain, source, format)
    """
    
    def __init__(
        self, 
        catalog_path: Optional[str] = None,
        enable_semantic: bool = False,
    ):
        if catalog_path is None:
            catalog_path = CATALOG_PATH
        
        self.catalog_path = Path(catalog_path)
        self.catalog: dict = {}
        self.datasets: list[dict] = []
        self.domain_mapping: dict[str, list[str]] = {}
        
        # BM25S search engines
        self.bm25_retriever: Optional[bm25s.BM25] = None
        self.description_corpus: list[str] = []
        self.description_tokens: list[list[str]] = []
        
        # Semantic search
        self.enable_semantic = enable_semantic
        self.embedder: Optional[OpenAIEmbedder] = None
        self.dataset_embeddings: Optional[list[list[float]]] = None
        
        self._load_catalog()
        self._build_indices()
        
        if enable_semantic:
            self._build_semantic_index()
    
    def _load_catalog(self) -> None:
        """Load the catalog from JSON file."""
        # Use shared cache if using default path
        if self.catalog_path == CATALOG_PATH:
            self.catalog = get_catalog()
        else:
            import json
            with open(self.catalog_path, "r") as f:
                self.catalog = json.load(f)
        
        self.datasets = self.catalog.get("datasets", [])
        
        # Build reverse domain mapping
        categories = self.catalog.get("summary", {}).get("categories", {})
        self.domain_mapping = {}
        for domain, files in categories.items():
            for file in files:
                self.domain_mapping[file] = self.domain_mapping.get(file, [])
                self.domain_mapping[file].append(domain)
    
    def _build_indices(self) -> None:
        """Build search indices using bm25s."""
        self.description_corpus = []
        for ds in self.datasets:
            cols = ds.get("columns", [])
            expanded_cols = []
            for col in cols:
                expanded_cols.append(col.lower())
                expanded_cols.extend(re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', col))
            
            text = " ".join([
                ds.get('name', ''),
                ds.get('description', ''),
                ds.get('use_case', ''),
                " ".join(cols),
                " ".join(expanded_cols)
            ])
            self.description_corpus.append(text)
        
        self.description_tokens = bm25s.tokenize(
            self.description_corpus,
            lower=True,
            stopwords="en"
        )
        
        self.bm25_retriever = bm25s.BM25()
        self.bm25_retriever.index(self.description_tokens)
    
    def _build_semantic_index(self) -> None:
        """Build semantic embeddings for all datasets using OpenAI."""
        self.embedder = OpenAIEmbedder()
        
        # Create text representations for embedding
        texts = []
        for ds in self.datasets:
            # Create a rich text representation
            cols = ds.get("columns", [])
            text = f"{ds.get('name', '')}. {ds.get('description', '')} {ds.get('use_case', '')} Columns: {', '.join(cols)}"
            texts.append(text)
        
        # Get embeddings (uses cache internally)
        self.dataset_embeddings = self.embedder.embed_batch(texts)
    
    def _get_dataset_tags(self, dataset: dict) -> list[str]:
        """Extract tags from a dataset entry."""
        tags = []
        
        file = dataset.get("file", "")
        if file in self.domain_mapping:
            tags.extend(self.domain_mapping[file])
        
        fmt = dataset.get("format", "")
        if "HuggingFace" in fmt:
            tags.append("huggingface")
        if "CSV" in fmt:
            tags.append("csv")
        
        return tags
    
    def _matches_filters(self, dataset: dict, filters: SearchFilters) -> bool:
        """Check if a dataset matches the given filters."""
        file = dataset.get("file", "")
        
        if filters.domain:
            domains = self.domain_mapping.get(file, [])
            if filters.domain.lower() not in [d.lower() for d in domains]:
                return False
        
        if filters.source:
            source = dataset.get("source", "")
            fmt = dataset.get("format", "")
            if filters.source.lower() not in source.lower() and filters.source.lower() not in fmt.lower():
                return False
        
        if filters.format:
            fmt = dataset.get("format", "")
            if filters.format.lower() not in fmt.lower():
                return False
        
        if filters.columns:
            ds_columns = [c.lower() for c in dataset.get("columns", [])]
            for required_col in filters.columns:
                if not any(required_col.lower() in col for col in ds_columns):
                    return False
        
        if filters.min_rows:
            rows = dataset.get("rows")
            if rows is None or rows < filters.min_rows:
                return False
        
        return True
    
    def _compute_column_score(self, query: str, dataset: dict) -> float:
        """Compute a score for column matching."""
        columns = dataset.get("columns", [])
        if not columns:
            return 0.0
        
        query_terms = tokenize_text(query)
        score = 0.0
        
        for col in columns:
            col_lower = col.lower()
            col_terms = set(tokenize_text(col))
            
            for term in query_terms:
                if term == col_lower:
                    score += 3.0
                elif term in col_lower:
                    score += 2.0
                elif term in col_terms:
                    score += 1.5
        
        return score
    
    def _compute_semantic_score(self, query: str, doc_idx: int) -> float:
        """Compute semantic similarity score using OpenAI embeddings."""
        if self.embedder is None or self.dataset_embeddings is None:
            return 0.0
        
        query_embedding = self.embedder.embed(query)
        doc_embedding = self.dataset_embeddings[doc_idx]
        
        return cosine_similarity(query_embedding, doc_embedding)
    
    def search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        top_k: int = 10,
        use_semantic: bool = True,
        bm25_weight: float = 0.5,
        column_weight: float = 0.2,
        semantic_weight: float = 0.3,
        include_score_breakdown: bool = False,
    ) -> list[SearchResult]:
        """
        Hybrid search combining BM25, column matching, and semantic similarity.
        
        Args:
            query: Search query string
            filters: Optional filters to apply
            top_k: Number of results to return
            use_semantic: Whether to use semantic search (requires enable_semantic=True in init)
            bm25_weight: Weight for BM25 lexical score (default 0.5)
            column_weight: Weight for column matching score (default 0.2)
            semantic_weight: Weight for semantic similarity (default 0.3)
            include_score_breakdown: If True, include individual scores in results
        
        Returns:
            List of SearchResult objects, ranked by combined relevance score
        """
        if filters is None:
            filters = SearchFilters()
        
        # Determine if we can use semantic search
        can_use_semantic = use_semantic and self.enable_semantic and self.embedder is not None
        
        # Get BM25 scores
        query_tokens = bm25s.tokenize([query], lower=True, stopwords="en")
        n_docs = len(self.datasets)
        doc_ids, bm25_scores = self.bm25_retriever.retrieve(query_tokens, k=n_docs)
        doc_ids = doc_ids[0]
        bm25_scores = bm25_scores[0]
        bm25_score_map = {int(doc_id): float(score) for doc_id, score in zip(doc_ids, bm25_scores)}
        
        # Normalize BM25 scores to 0-1 range for fair blending
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        
        results: list[tuple[int, float, dict]] = []
        
        for idx, dataset in enumerate(self.datasets):
            if not self._matches_filters(dataset, filters):
                continue
            
            # Get raw scores
            raw_bm25 = bm25_score_map.get(idx, 0.0)
            raw_col = self._compute_column_score(query, dataset)
            raw_sem = self._compute_semantic_score(query, idx) if can_use_semantic else 0.0
            
            # Normalize scores for blending
            # BM25: normalize by max score
            norm_bm25 = raw_bm25 / max_bm25 if max_bm25 > 0 else 0.0
            # Column: cap at 10 for normalization (10 = very strong column match)
            norm_col = min(raw_col / 10.0, 1.0)
            # Semantic: already 0-1 (cosine similarity)
            norm_sem = raw_sem
            
            # Compute weights
            if can_use_semantic:
                total_weight = bm25_weight + column_weight + semantic_weight
                w_bm25 = bm25_weight / total_weight
                w_col = column_weight / total_weight
                w_sem = semantic_weight / total_weight
            else:
                total_weight = bm25_weight + column_weight
                w_bm25 = bm25_weight / total_weight
                w_col = column_weight / total_weight
                w_sem = 0.0
            
            # Combined score
            final_score = (w_bm25 * norm_bm25) + (w_col * norm_col) + (w_sem * norm_sem)
            
            score_breakdown = {
                "bm25": round(raw_bm25, 4),
                "column": round(raw_col, 4),
                "semantic": round(raw_sem, 4) if can_use_semantic else None,
            }
            
            results.append((idx, final_score, score_breakdown))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Build SearchResult objects
        search_results = []
        for idx, score, breakdown in results[:top_k]:
            ds = self.datasets[idx]
            result = SearchResult(
                asset_id=ds.get("file", ""),
                name=ds.get("name", ""),
                description=ds.get("description", ""),
                schema_summary=ds.get("columns", []),
                format=ds.get("format", ""),
                source=ds.get("source"),
                tags=self._get_dataset_tags(ds),
                score=round(score, 4),
                use_case=ds.get("use_case", ""),
                rows=ds.get("rows"),
                bm25_score=breakdown["bm25"] if include_score_breakdown else None,
                column_score=breakdown["column"] if include_score_breakdown else None,
                semantic_score=breakdown["semantic"] if include_score_breakdown else None,
            )
            search_results.append(result)
        
        return search_results


# Global searcher instance for convenience functions
_default_searcher: Optional[CatalogSearcher] = None


def catalog_search(
    query: str,
    filters: Optional[dict] = None,
    top_k: int = 10,
    catalog_path: Optional[str] = None,
    use_semantic: bool = False,
    include_score_breakdown: bool = False,
) -> list[dict]:
    """
    Main entry point for catalog search.
    
    Args:
        query: Search query string
        filters: Optional dict with keys: domain, source, format, columns, min_rows
        top_k: Number of results to return
        catalog_path: Optional path to catalog.json
        use_semantic: Enable hybrid search with OpenAI embeddings (requires OPENAI_API_KEY)
        include_score_breakdown: If True, include bm25/column/semantic scores in results
    
    Returns:
        List of result dicts with: asset_id, name, description, schema_summary, 
        format, source, tags, score, use_case, rows
    
    Example:
        >>> results = catalog_search("loan default prediction with credit score")
        >>> results = catalog_search("insurance", filters={"domain": "insurance"})
        >>> results = catalog_search("customer risk", use_semantic=True)
    """
    searcher = CatalogSearcher(catalog_path, enable_semantic=use_semantic)
    
    search_filters = SearchFilters()
    if filters:
        search_filters = SearchFilters(
            domain=filters.get("domain"),
            source=filters.get("source"),
            format=filters.get("format"),
            columns=filters.get("columns"),
            min_rows=filters.get("min_rows"),
        )
    
    results = searcher.search(
        query, 
        filters=search_filters, 
        top_k=top_k,
        use_semantic=use_semantic,
        include_score_breakdown=include_score_breakdown,
    )
    return [r.to_dict() for r in results]


def hybrid_search(
    query: str,
    filters: Optional[dict] = None,
    top_k: int = 10,
    bm25_weight: float = 0.5,
    column_weight: float = 0.2,
    semantic_weight: float = 0.3,
    catalog_path: Optional[str] = None,
) -> list[dict]:
    """
    Hybrid search combining BM25 lexical search with OpenAI semantic search.
    
    This is the recommended search method for best results.
    
    Args:
        query: Search query string
        filters: Optional dict with keys: domain, source, format, columns, min_rows
        top_k: Number of results to return
        bm25_weight: Weight for BM25 lexical score (default 0.5)
        column_weight: Weight for column matching (default 0.2)
        semantic_weight: Weight for semantic similarity (default 0.3)
        catalog_path: Optional path to catalog.json
    
    Returns:
        List of result dicts with score breakdown included
    
    Example:
        >>> results = hybrid_search("find datasets about predicting if someone will pay back a loan")
        >>> results = hybrid_search("customer demographics for risk modeling")
    """
    searcher = CatalogSearcher(catalog_path, enable_semantic=True)
    
    search_filters = SearchFilters()
    if filters:
        search_filters = SearchFilters(
            domain=filters.get("domain"),
            source=filters.get("source"),
            format=filters.get("format"),
            columns=filters.get("columns"),
            min_rows=filters.get("min_rows"),
        )
    
    results = searcher.search(
        query,
        filters=search_filters,
        top_k=top_k,
        use_semantic=True,
        bm25_weight=bm25_weight,
        column_weight=column_weight,
        semantic_weight=semantic_weight,
        include_score_breakdown=True,
    )
    return [r.to_dict() for r in results]


# Convenience function for quick access
def search(query: str, **kwargs) -> list[dict]:
    """Shorthand for catalog_search."""
    return catalog_search(query, **kwargs)


# =============================================================================
# LANGCHAIN TOOL DEFINITION
# =============================================================================

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class CatalogSearchInput(BaseModel):
    """Input schema for catalog search tool."""
    
    query: str = Field(
        description="Natural language search query describing the data you need. "
                    "Can include dataset names, column names, use cases, or concepts. "
                    "Examples: 'loan default prediction', 'customer demographics', "
                    "'data with age and income columns', 'insurance underwriting'"
    )
    domain: Optional[str] = Field(
        default=None,
        description="Filter by domain category. Options: 'credit_risk', 'insurance', 'financial_health'"
    )
    source: Optional[str] = Field(
        default=None,
        description="Filter by data source. Example: 'HuggingFace' for HuggingFace datasets"
    )
    format: Optional[str] = Field(
        default=None,
        description="Filter by file format. Options: 'CSV', 'Python'"
    )
    required_columns: Optional[list[str]] = Field(
        default=None,
        description="List of column names that must be present in the dataset. "
                    "Example: ['age', 'income', 'credit_score']"
    )
    min_rows: Optional[int] = Field(
        default=None,
        description="Minimum number of rows required in the dataset"
    )
    top_k: int = Field(
        default=5,
        description="Maximum number of results to return (1-10)"
    )


@tool(args_schema=CatalogSearchInput)
def catalog_search_tool(
    query: str,
    domain: Optional[str] = None,
    source: Optional[str] = None,
    format: Optional[str] = None,
    required_columns: Optional[list[str]] = None,
    min_rows: Optional[int] = None,
    top_k: int = 5,
) -> str:
    """
    Search the internal data catalog to discover relevant datasets, tables, and features.
    
    Use this tool when you need to:
    - Find datasets for a specific prediction task (e.g., loan default, insurance pricing)
    - Discover what data is available for a domain (credit risk, insurance, financial health)
    - Find datasets with specific columns or features
    - Identify datasets suitable for training ML models
    
    The search uses hybrid retrieval combining:
    - BM25 lexical matching (keyword-based)
    - Semantic similarity (understands concepts and synonyms)
    - Column name matching
    
    Returns a ranked list of matching datasets with their metadata.
    """
    # Build filters
    filters = {}
    if domain:
        filters["domain"] = domain
    if source:
        filters["source"] = source
    if format:
        filters["format"] = format
    if required_columns:
        filters["columns"] = required_columns
    if min_rows:
        filters["min_rows"] = min_rows
    
    # Clamp top_k
    top_k = max(1, min(10, top_k))
    
    # Check if we can use hybrid search
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    
    if has_openai:
        results = hybrid_search(query, filters=filters if filters else None, top_k=top_k)
    else:
        results = catalog_search(query, filters=filters if filters else None, top_k=top_k)
    
    if not results:
        return "No datasets found matching your criteria. Try broadening your search or removing filters."
    
    # Format output for agent consumption
    output_lines = [f"Found {len(results)} relevant dataset(s):\n"]
    
    for i, r in enumerate(results, 1):
        output_lines.append(f"**{i}. {r['name']}**")
        output_lines.append(f"   - File: `{r['asset_id']}`")
        output_lines.append(f"   - Format: {r['format']}")
        if r.get('rows'):
            output_lines.append(f"   - Rows: {r['rows']:,}")
        output_lines.append(f"   - Use Case: {r['use_case']}")
        if r.get('schema_summary'):
            cols = r['schema_summary'][:8]  # Limit to 8 columns for readability
            cols_str = ", ".join(cols)
            if len(r['schema_summary']) > 8:
                cols_str += f", ... (+{len(r['schema_summary']) - 8} more)"
            output_lines.append(f"   - Columns: {cols_str}")
        if r.get('tags'):
            output_lines.append(f"   - Tags: {', '.join(r['tags'])}")
        output_lines.append(f"   - Relevance Score: {r['score']:.3f}")
        if r.get('score_breakdown'):
            bd = r['score_breakdown']
            output_lines.append(f"   - Score Breakdown: BM25={bd['bm25']:.2f}, Column={bd['column']:.1f}, Semantic={bd['semantic']:.3f}")
        output_lines.append("")
    
    return "\n".join(output_lines)


@tool
def list_datasets_tool() -> str:
    """
    List all available datasets that can be used in data operations.
    
    Shows:
    - SQL tables loaded in the warehouse
    - Datasets from previous operations (joins, queries)
    - Basic info about each dataset (rows, columns)
    
    Use this to see what data is currently available before running queries or joins.
    """
    datasets = get_all_available_datasets()
    output_lines = ["## Available Datasets\n"]
    
    # Registered datasets (from previous operations)
    if datasets["registered"]:
        output_lines.append("### From Previous Operations")
        for ds in datasets["registered"]:
            rows = ds.get("rows")
            cols = ds.get("columns")
            rows_s = f"{rows:,}" if isinstance(rows, int) else str(rows or "?")
            cols_s = str(cols) if cols is not None else "?"
            output_lines.append(f"- `{ds['ref']}`: {rows_s} rows × {cols_s} columns")
        output_lines.append("")
    
    # SQL tables
    if datasets["sql_tables"]:
        output_lines.append("### SQL Tables")
        for ds in datasets["sql_tables"]:
            cols = ds.get("column_names", [])
            cols_display = ", ".join(cols[:5]) + (f" (+{len(cols) - 5} more)" if len(cols) > 5 else "")
            rows = ds.get("rows") or 0
            output_lines.append(f"- `{ds['ref']}`: {rows:,} rows — [{cols_display}]")
        output_lines.append("")
    
    # Catalog datasets info
    if datasets["catalog"]:
        output_lines.append("### Catalog Datasets (use catalog_search_tool to explore)")
        output_lines.append(f"- {len(datasets['catalog'])} datasets available in catalog")
        output_lines.append("")
    
    if len(output_lines) == 1:
        return "No datasets currently available. Use sql_query_tool or dataset_get_tool to load data first."
    
    return "\n".join(output_lines)


# Create a list of tools for easy import
catalog_tools = [catalog_search_tool, list_datasets_tool]