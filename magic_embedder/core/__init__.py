"""magic-embedder core interfaces and data classes."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EmbeddingModelType(Enum):
    """Embedding model type."""

    TEXT = "text"
    IMAGE = "image"
    MULTIMODAL = "multimodal"


@dataclass
class EmbeddingResult:
    """Result of an embedding operation.

    Attributes:
        embeddings: List of embedding vectors.
        model_name: Name of the embedding model used.
        dimension: Dimension of each embedding vector.
        metadata: Additional metadata about the embedding process.
    """

    embeddings: list[list[float]]
    model_name: str
    dimension: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "embeddings": self.embeddings,
            "model_name": self.model_name,
            "dimension": self.dimension,
            "metadata": self.metadata,
        }


@dataclass
class TextEmbeddingResult(EmbeddingResult):
    """Result of a text embedding operation.

    Attributes:
        texts: The source texts that were embedded.
    """

    texts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["texts"] = self.texts
        return d


@dataclass
class ImageEmbeddingResult(EmbeddingResult):
    """Result of an image embedding operation.

    Attributes:
        image_paths: Paths to the images that were embedded.
    """

    image_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["image_paths"] = self.image_paths
        return d


from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for all embedders.

    All embedders must implement the `embed` method.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> TextEmbeddingResult:
        """Embed texts into vectors.

        Args:
            texts: List of texts to embed.

        Returns:
            TextEmbeddingResult containing the embedding vectors.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the embedder."""
        ...

    @property
    def description(self) -> str:
        """Return a short description of the embedder."""
        return ""

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 0

    @property
    def model_type(self) -> EmbeddingModelType:
        """Return the model type."""
        return EmbeddingModelType.TEXT


class BaseImageEmbedder(ABC):
    """Abstract base class for image embedders."""

    @abstractmethod
    def embed_images(self, image_paths: list[str]) -> ImageEmbeddingResult:
        """Embed images into vectors.

        Args:
            image_paths: List of image paths to embed.

        Returns:
            ImageEmbeddingResult containing the embedding vectors.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the image embedder."""
        ...

    @property
    def description(self) -> str:
        """Return a short description of the image embedder."""
        return ""

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 0
