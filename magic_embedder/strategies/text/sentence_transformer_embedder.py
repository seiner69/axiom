"""SentenceTransformer text embedding strategy."""

from magic_embedder.core import BaseEmbedder, EmbeddingModelType, TextEmbeddingResult

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers package is required for SentenceTransformerEmbedder. "
        "Install with: pip install sentence-transformers"
    )


# Common model dimension map
MODEL_DIMENSIONS: dict[str, int] = {
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
    "multilingual-e5-small": 384,
    "bge-m3": 1024,
}


class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer embedding embedder.

    Supports any model from the SentenceTransformers library.

    Attributes:
        model_name: Name of the SentenceTransformer model.
        device: Device to use ('cpu', 'cuda', 'mps').
        normalize: Whether to normalize embeddings to unit length.
        prompt: Optional prompt to prepend to all texts.
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        normalize: bool = True,
        prompt: str | None = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self._device = device
        self.normalize = normalize
        self.prompt = prompt
        self.batch_size = batch_size

        self._model = SentenceTransformer(model_name, device=device)

    @property
    def name(self) -> str:
        return f"sentence_transformer_{self.model_name}"

    @property
    def description(self) -> str:
        return f"SentenceTransformer embedder ({self.model_name})"

    @property
    def dimension(self) -> int:
        if self.model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self.model_name]
        # Get from model
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_type(self) -> EmbeddingModelType:
        return EmbeddingModelType.TEXT

    def embed(self, texts: list[str]) -> TextEmbeddingResult:
        """Embed texts using SentenceTransformer.

        Args:
            texts: List of texts to embed.

        Returns:
            TextEmbeddingResult with embedding vectors.
        """
        # Prepend prompt if set
        if self.prompt:
            texts = [f"{self.prompt}{text}" for text in texts]

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

        # Convert numpy arrays to lists
        embedding_list = embeddings.tolist()

        return TextEmbeddingResult(
            embeddings=embedding_list,
            model_name=self.model_name,
            dimension=self.dimension,
            texts=texts,
            metadata={
                "batch_size": self.batch_size,
                "normalize": self.normalize,
                "prompt": self.prompt,
                "total_texts": len(texts),
            },
        )
