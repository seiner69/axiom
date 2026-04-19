"""All embedding strategies."""

from magic_embedder.strategies.text import OpenAITextEmbedder, SentenceTransformerEmbedder

__all__ = [
    # Text embedders
    "OpenAITextEmbedder",
    "SentenceTransformerEmbedder",
]

# Image embedders - optional, may fail if dependencies not installed
try:
    from magic_embedder.strategies.image import CLIPImageEmbedder

    __all__.append("CLIPImageEmbedder")
except ImportError:
    CLIPImageEmbedder = None
