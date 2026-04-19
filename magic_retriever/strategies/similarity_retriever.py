"""Similarity-based retrieval strategy."""

from magic_retriever.core import BaseRetriever, RetrievedChunk, RetrievalMode, RetrievalResult


class SimilarityRetriever(BaseRetriever):
    """Simple similarity-based retriever.

    Embeds the query and retrieves the most similar chunks from the vector store.

    Attributes:
        embedder: Text embedder for encoding queries.
        vector_store: Vector store to search.
        top_k: Default number of results to retrieve.
    """

    def __init__(
        self,
        embedder,
        vector_store,
        top_k: int = 5,
    ):
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "similarity"

    @property
    def description(self) -> str:
        return "Similarity-based retrieval"

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant chunks for a query.

        Args:
            query: The query text.
            top_k: Number of chunks to retrieve.

        Returns:
            RetrievalResult with relevant chunks.
        """
        k = top_k if top_k is not None else self._top_k

        # Embed query
        embed_result = self._embedder.embed([query])
        query_vector = embed_result.embeddings[0]

        # Search vector store
        search_result = self._vector_store.search(query_vector, top_k=k)

        # Convert to RetrievedChunks
        chunks = []
        for i, (entry, score) in enumerate(zip(search_result.entries, search_result.scores)):
            chunk = RetrievedChunk(
                chunk_id=entry.id,
                content=entry.text or "",
                score=score,
                rank=i + 1,
                metadata=entry.metadata,
            )
            chunks.append(chunk)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            mode=RetrievalMode.SIMILARITY,
            metadata={
                "retriever": self.name,
                "embedder": embed_result.model_name,
                "total_retrieved": len(chunks),
            },
        )
