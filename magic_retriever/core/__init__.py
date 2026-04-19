"""magic-retriever core interfaces and data classes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalMode(Enum):
    """Retrieval mode enumeration."""

    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximal Marginal Relevance


@dataclass
class RetrievedChunk:
    """A chunk retrieved from the vector store.

    Attributes:
        chunk_id: ID of the chunk.
        content: Text content of the chunk.
        score: Relevance score.
        rank: Rank position (1-indexed).
        metadata: Additional metadata.
    """

    chunk_id: str
    content: str
    score: float = 0.0
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


@dataclass
class RetrievalResult:
    """Result of a retrieval operation.

    Attributes:
        chunks: List of retrieved chunks.
        query: The original query.
        mode: Retrieval mode used.
        metadata: Additional metadata about the retrieval.
    """

    chunks: list[RetrievedChunk] = field(default_factory=list)
    query: str = ""
    mode: RetrievalMode = RetrievalMode.SIMILARITY
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "query": self.query,
            "mode": self.mode.value,
            "metadata": self.metadata,
        }


from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    """Abstract base class for all retrievers.

    All retrievers must implement the `retrieve` method.
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The query text.
            top_k: Number of chunks to retrieve.

        Returns:
            RetrievalResult with relevant chunks.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the retriever."""
        ...

    @property
    def description(self) -> str:
        """Return a short description."""
        return ""
