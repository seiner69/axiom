"""OpenAI text embedding strategy."""

import os
from typing import Any

from magic_embedder.core import BaseEmbedder, EmbeddingModelType, TextEmbeddingResult

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("openai package is required for OpenAITextEmbedder. Install with: pip install openai")


# Model dimension map (approximate)
MODEL_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAITextEmbedder(BaseEmbedder):
    """OpenAI text embedding embedder.

    Supports OpenAI's embedding models (text-embedding-3-small,
    text-embedding-3-large, text-embedding-ada-002).

    Attributes:
        model: OpenAI embedding model name.
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        dimensions: Output dimension (for embedding-3 models, truncates if smaller).
        batch_size: Number of texts to embed in a single request.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 100,
    ):
        self.model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._dimensions = dimensions
        self.batch_size = batch_size

        if not self._api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass api_key.")

        self._client = OpenAI(api_key=self._api_key)

    @property
    def name(self) -> str:
        return f"openai_{self.model}"

    @property
    def description(self) -> str:
        return f"OpenAI text embedder ({self.model})"

    @property
    def dimension(self) -> int:
        if self._dimensions:
            return self._dimensions
        return MODEL_DIMENSIONS.get(self.model, 1536)

    @property
    def model_type(self) -> EmbeddingModelType:
        return EmbeddingModelType.TEXT

    def embed(self, texts: list[str]) -> TextEmbeddingResult:
        """Embed texts using OpenAI API.

        Args:
            texts: List of texts to embed.

        Returns:
            TextEmbeddingResult with embedding vectors.
        """
        all_embeddings: list[list[float]] = []

        # Batch processing
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model,
                input=batch,
                dimensions=self._dimensions,
            )

            for embedding in response.data:
                all_embeddings.append(embedding.embedding)

        return TextEmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.model,
            dimension=self.dimension,
            texts=texts,
            metadata={
                "batch_size": self.batch_size,
                "total_texts": len(texts),
            },
        )
