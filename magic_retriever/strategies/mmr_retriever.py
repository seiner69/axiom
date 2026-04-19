"""Maximal Marginal Relevance (MMR) retrieval strategy."""

from magic_retriever.core import BaseRetriever, RetrievedChunk, RetrievalMode, RetrievalResult


class MMRRetriever(BaseRetriever):
    """Maximal Marginal Relevance retriever.

    Retrieves chunks that are both relevant to the query and diverse from each other.
    Balances relevance (similarity to query) with diversity (dissimilarity between selected chunks).

    Attributes:
        embedder: Text embedder for encoding queries.
        vector_store: Vector store to search.
        top_k: Number of chunks to retrieve.
        fetch_k: Number of chunks to initially fetch (before MMR selection).
        lambda_mult: Balance between relevance and diversity (0=max diversity, 1=max relevance).
    """

    def __init__(
        self,
        embedder,
        vector_store,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        if fetch_k < top_k:
            raise ValueError("fetch_k must be >= top_k")

        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k
        self._fetch_k = fetch_k
        self._lambda_mult = lambda_mult

    @property
    def name(self) -> str:
        return "mmr"

    @property
    def description(self) -> str:
        return f"MMR retrieval (lambda={self._lambda_mult})"

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(va * vb for va, vb in zip(a, b))
        norm_a = sum(va**2 for va in a) ** 0.5
        norm_b = sum(vb**2 for vb in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant and diverse chunks for a query.

        Args:
            query: The query text.
            top_k: Number of chunks to retrieve.

        Returns:
            RetrievalResult with relevant and diverse chunks.
        """
        k = top_k if top_k is not None else self._top_k

        # Embed query
        embed_result = self._embedder.embed([query])
        query_vector = embed_result.embeddings[0]

        # Fetch more candidates than we need
        search_result = self._vector_store.search(query_vector, top_k=self._fetch_k)

        if not search_result.entries:
            return RetrievalResult(
                chunks=[],
                query=query,
                mode=RetrievalMode.MMR,
                metadata={"retriever": self.name, "total_retrieved": 0},
            )

        # Build candidate list with query similarity
        candidates = []
        for i, (entry, score) in enumerate(zip(search_result.entries, search_result.scores)):
            candidates.append({
                "entry": entry,
                "query_sim": score,
                "index": i,
            })

        # Sort by query similarity descending for initial selection
        candidates.sort(key=lambda x: x["query_sim"], reverse=True)

        selected: list[dict] = []
        remaining = candidates.copy()

        while len(selected) < k and remaining:
            # Calculate MMR score for each remaining candidate
            best_score = -float("inf")
            best_idx = 0

            for cand_idx, cand in enumerate(remaining):
                # Relevance: query similarity
                rel = cand["query_sim"]

                # Diversity: minimum similarity to already selected
                # If embeddings are empty (e.g., ChromaDB not returning them), skip diversity
                div = 0.0
                if selected and cand["entry"].embedding:
                    sims = []
                    for s in selected:
                        if s["entry"].embedding:
                            sim = self._cosine_similarity(cand["entry"].embedding, s["entry"].embedding)
                            sims.append(sim)
                    if sims:
                        div = min(sims)

                # MMR score
                mmr_score = self._lambda_mult * rel - (1 - self._lambda_mult) * div

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = cand_idx

            chosen = remaining.pop(best_idx)
            selected.append(chosen)

        # Build result
        chunks = []
        for rank, sel in enumerate(selected):
            entry = sel["entry"]
            chunk = RetrievedChunk(
                chunk_id=entry.id,
                content=entry.text or "",
                score=sel["query_sim"],
                rank=rank + 1,
                metadata=entry.metadata,
            )
            chunks.append(chunk)

        return RetrievalResult(
            chunks=chunks,
            query=query,
            mode=RetrievalMode.MMR,
            metadata={
                "retriever": self.name,
                "embedder": embed_result.model_name,
                "lambda_mult": self._lambda_mult,
                "fetch_k": self._fetch_k,
                "total_retrieved": len(chunks),
            },
        )
