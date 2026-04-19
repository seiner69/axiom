"""magic-embedder: A modular text and image embedding library.

Provides text and image embedding strategies for RAG applications.

Example:
    >>> from magic_embedder.strategies import SentenceTransformerEmbedder
    >>>
    >>> embedder = SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
    >>> result = embedder.embed(["Hello world", "This is a test"])
    >>> print(f"Dimension: {result.dimension}, Count: {len(result.embeddings)}")
"""

from magic_embedder.core import (
    BaseEmbedder,
    BaseImageEmbedder,
    EmbeddingModelType,
    EmbeddingResult,
    ImageEmbeddingResult,
    TextEmbeddingResult,
)
from magic_embedder.strategies import (
    OpenAITextEmbedder,
    SentenceTransformerEmbedder,
)

__all__ = [
    # Core
    "BaseEmbedder",
    "BaseImageEmbedder",
    "EmbeddingModelType",
    "EmbeddingResult",
    "TextEmbeddingResult",
    "ImageEmbeddingResult",
    # Text strategies
    "OpenAITextEmbedder",
    "SentenceTransformerEmbedder",
]

# Image strategies - optional
try:
    from magic_embedder.strategies import CLIPImageEmbedder

    __all__.append("CLIPImageEmbedder")
except ImportError:
    pass

__version__ = "0.1.0"
