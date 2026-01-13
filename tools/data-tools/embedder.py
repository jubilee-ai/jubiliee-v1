"""
OpenAI embeddings with disk caching.

Provides a reusable embedding class that caches results to avoid redundant API calls.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

try:
    from .utils import get_openai_client
except ImportError:
    from utils import get_openai_client


class OpenAIEmbedder:
    """
    OpenAI embeddings with caching.
    Uses text-embedding-3-small model.
    
    Example:
        >>> embedder = OpenAIEmbedder()
        >>> embedding = embedder.embed("Hello world")
        >>> embeddings = embedder.embed_batch(["Hello", "World"])
    """
    
    MODEL = "text-embedding-3-small"
    DIMENSIONS = 1536  # Default dimensions for text-embedding-3-small
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the embedder.
        
        Args:
            cache_dir: Directory to store cached embeddings. 
                      Defaults to .embedding_cache in the data-tools directory.
        """
        self.cache_dir = cache_dir or Path(__file__).parent / ".embedding_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._cache: dict[str, list[float]] = {}
        self._load_cache()
    
    def _cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _cache_file(self) -> Path:
        return self.cache_dir / "openai_embeddings.json"
    
    def _load_cache(self) -> None:
        """Load cached embeddings from disk."""
        cache_file = self._cache_file()
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}
    
    def _save_cache(self) -> None:
        """Save embeddings cache to disk."""
        with open(self._cache_file(), "w") as f:
            json.dump(self._cache, f)
    
    def embed(self, text: str) -> list[float]:
        """
        Get embedding for a single text, using cache if available.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        key = self._cache_key(text)
        
        if key in self._cache:
            return self._cache[key]
        
        # Call OpenAI API
        client = get_openai_client()
        response = client.embeddings.create(
            model=self.MODEL,
            input=text,
        )
        embedding = response.data[0].embedding
        
        # Cache the result
        self._cache[key] = embedding
        self._save_cache()
        
        return embedding
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Get embeddings for multiple texts, using cache where available.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (one per input text)
        """
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Fetch uncached embeddings from API
        if uncached_texts:
            client = get_openai_client()
            response = client.embeddings.create(
                model=self.MODEL,
                input=uncached_texts,
            )
            
            # Process results and update cache
            for j, embedding_data in enumerate(response.data):
                idx = uncached_indices[j]
                text = uncached_texts[j]
                embedding = embedding_data.embedding
                
                results[idx] = embedding
                self._cache[self._cache_key(text)] = embedding
            
            self._save_cache()
        
        return results

