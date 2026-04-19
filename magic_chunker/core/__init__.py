"""magic-chunker core interfaces and data classes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(Enum):
    """Chunk type enumeration."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    CODE = "code"


@dataclass
class Node:
    """Represents a parsed node from a document.

    Attributes:
        id: Unique identifier for the node.
        content: The text content of the node.
        node_type: Type of the node (text, table, image, code).
        metadata: Additional metadata about the node.
        parent_id: Optional parent node ID for hierarchical structures.
    """

    id: str
    content: str
    node_type: ChunkType = ChunkType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type.value,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
        }


@dataclass
class Chunk:
    """Represents a chunk resulting from chunking a node or document.

    Attributes:
        id: Unique identifier for the chunk.
        content: The text content of the chunk.
        chunk_type: Type of the chunk.
        metadata: Additional metadata about the chunk.
        node_ids: List of source node IDs that contributed to this chunk.
        index: Position index of this chunk within its parent.
    """

    id: str
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)
    node_ids: list[str] = field(default_factory=list)
    index: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "metadata": self.metadata,
            "node_ids": self.node_ids,
            "index": self.index,
        }


@dataclass
class ChunkingResult:
    """Result of a chunking operation.

    Attributes:
        chunks: List of produced chunks.
        metadata: Metadata about the chunking operation.
    """

    chunks: list[Chunk] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata,
        }


from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract base class for all chunkers.

    All chunkers must implement the `chunk` method that takes a list of Nodes
    and returns a ChunkingResult.
    """

    @abstractmethod
    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """Chunk the given nodes into smaller pieces.

        Args:
            nodes: List of Nodes to chunk.

        Returns:
            ChunkingResult containing the produced chunks.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the chunker strategy."""
        ...

    @property
    def description(self) -> str:
        """Return a short description of the chunker."""
        return ""
