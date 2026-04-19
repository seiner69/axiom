"""magic-vectorstore core interfaces and data classes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class VectorStoreType(Enum):
    """Vector store type."""

    CHROMA = "chroma"
    FAISS = "faiss"
    IN_MEMORY = "in_memory"


@dataclass
class VectorEntry:
    """A vector entry in the store.

    Attributes:
        id: Unique identifier for the entry.
        embedding: The vector data.
        text: The original text content (if applicable).
        metadata: Additional metadata.
    """

    id: str
    embedding: list[float]
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "embedding": self.embedding,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class QueryResult:
    """Result of a vector similarity search.

    Attributes:
        entries: List of matching VectorEntry objects.
        scores: Similarity scores for each entry.
        query: The original query text or vector.
    """

    entries: list[VectorEntry] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    query: str | list[float] = ""

    def to_dict(self) -> dict:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "scores": self.scores,
            "query": self.query,
        }


@dataclass
class VectorStoreStats:
    """Statistics about a vector store.

    Attributes:
        total_entries: Total number of entries in the store.
        dimension: Dimension of each vector.
        store_type: Type of the underlying store.
        metadata: Additional statistics.
    """

    total_entries: int = 0
    dimension: int = 0
    store_type: VectorStoreType = VectorStoreType.IN_MEMORY
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_entries": self.total_entries,
            "dimension": self.dimension,
            "store_type": self.store_type.value,
            "metadata": self.metadata,
        }


from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """Abstract base class for all vector stores.

    All vector stores must implement the core CRUD operations.
    """

    @abstractmethod
    def add(self, entries: list[VectorEntry]) -> None:
        """Add entries to the store.

        Args:
            entries: List of VectorEntry objects to add.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Search for similar vectors.

        Args:
            query_vector: The query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            QueryResult with matching entries and scores.
        """
        ...

    @abstractmethod
    def delete(self, entry_ids: list[str]) -> None:
        """Delete entries from the store.

        Args:
            entry_ids: List of entry IDs to delete.
        """
        ...

    @abstractmethod
    def persist(self, path: str) -> None:
        """Persist the store to disk.

        Args:
            path: Path to persist to.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the vector store."""
        ...

    @property
    def description(self) -> str:
        """Return a short description."""
        return ""

    @abstractmethod
    def stats(self) -> VectorStoreStats:
        """Get statistics about the store."""
        ...
