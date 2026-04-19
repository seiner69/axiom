"""magic-chunker: A modular document chunking library.

The chunker module provides various strategies for splitting documents
into smaller, semantically coherent chunks for RAG applications.

Example:
    >>> from magic_chunker.core import Node, ChunkType
    >>> from magic_chunker.strategies import SemanticChunker
    >>>
    >>> nodes = [Node(id="1", content="Long document text...")]
    >>> chunker = SemanticChunker()
    >>> result = chunker.chunk(nodes)
    >>> print(f"Produced {len(result.chunks)} chunks")
"""

from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node
from magic_chunker.strategies import (
    FixedSizeChunker,
    HTMLChunker,
    MarkdownChunker,
    PDFChunker,
    SemanticChunker,
    SentenceChunker,
)

__all__ = [
    # Core
    "BaseChunker",
    "Chunk",
    "ChunkType",
    "ChunkingResult",
    "Node",
    # Strategies
    "FixedSizeChunker",
    "SentenceChunker",
    "SemanticChunker",
    "MarkdownChunker",
    "PDFChunker",
    "HTMLChunker",
]

__version__ = "0.1.0"
