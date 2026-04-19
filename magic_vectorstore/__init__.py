"""magic-vectorstore: A modular vector storage and similarity search library.

Provides vector storage strategies for RAG applications.

Example:
    >>> from magic_vectorstore.core import VectorEntry
    >>> from magic_vectorstore.strategies import ChromaVectorStore
    >>>
    >>> store = ChromaVectorStore(collection_name="my_collection")
    >>> entries = [VectorEntry(id="1", embedding=[0.1, 0.2], text="Hello world")]
    >>> store.add(entries)
    >>> result = store.search([0.1, 0.2], top_k=1)
    >>> print(f"Found {len(result.entries)} results")
"""

from magic_vectorstore.core import (
    BaseVectorStore,
    QueryResult,
    VectorEntry,
    VectorStoreStats,
    VectorStoreType,
)
from magic_vectorstore.strategies import ChromaVectorStore, FAISSVectorStore

__all__ = [
    # Core
    "BaseVectorStore",
    "QueryResult",
    "VectorEntry",
    "VectorStoreStats",
    "VectorStoreType",
    # Strategies
    "ChromaVectorStore",
    "FAISSVectorStore",
]

__version__ = "0.1.0"
