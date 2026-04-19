"""Chroma vector store implementation."""

from pathlib import Path
from typing import Any

from magic_vectorstore.core import BaseVectorStore, QueryResult, VectorEntry, VectorStoreStats, VectorStoreType

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError(
        "chromadb package is required for ChromaVectorStore. "
        "Install with: pip install chromadb"
    )


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation.

    Wrapper around ChromaDB for vector storage and similarity search.

    Attributes:
        collection_name: Name of the Chroma collection.
        persist_directory: Directory to persist the database (None for in-memory).
        distance_metric: Distance metric ('cosine', 'euclidean', 'manhattan').
    """

    def __init__(
        self,
        collection_name: str = "default",
        persist_directory: str | None = None,
        distance_metric: str = "cosine",
    ):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        # Create client
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )

        # Track entries in memory for text retrieval
        self._texts: dict[str, str] = {}
        self._metadatas: dict[str, dict] = {}

    @property
    def name(self) -> str:
        return f"chroma_{self.collection_name}"

    @property
    def description(self) -> str:
        return f"Chroma vector store ({self.collection_name}, {self.distance_metric})"

    def add(self, entries: list[VectorEntry]) -> None:
        """Add entries to the store.

        Args:
            entries: List of VectorEntry objects to add.
        """
        if not entries:
            return

        ids = [e.id for e in entries]
        embeddings = [e.embedding for e in entries]
        texts = [e.text or "" for e in entries]
        metadatas = [e.metadata if e.metadata else None for e in entries]

        # Store text and metadata for retrieval
        for entry in entries:
            self._texts[entry.id] = entry.text or ""
            self._metadatas[entry.id] = entry.metadata

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> QueryResult:
        """Search for similar vectors.

        Args:
            query_vector: The query vector.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filter.

        Returns:
            QueryResult with matching entries and scores.
        """
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_metadata,
            include=["metadatas", "documents", "distances", "embeddings"],
        )

        entries: list[VectorEntry] = []
        scores: list[float] = []

        if results["ids"] and len(results["ids"]) > 0:
            for i, entry_id in enumerate(results["ids"][0]):
                # Chroma returns distances, convert to similarity scores
                distance = results["distances"][0][i]

                if self.distance_metric == "cosine":
                    # Cosine distance -> similarity: 1 - distance
                    score = 1.0 - distance
                elif self.distance_metric == "euclidean":
                    # Convert euclidean distance to similarity (approximate)
                    max_dist = 2.0
                    score = max(0.0, 1.0 - distance / max_dist)
                else:
                    score = 1.0 / (1.0 + distance)

                # Get stored embedding (Chroma returns it when include=["embeddings"])
                # embeddings[0] is the list of embeddings for the first query vector
                _embeddings = results.get("embeddings")
                stored_embedding: list[float] = []
                if _embeddings and len(_embeddings) > 0 and len(_embeddings[0]) > i:
                    emb = _embeddings[0][i]
                    if emb is not None:
                        stored_embedding = list(emb)

                # Get text from in-memory store (more reliable than querying Chroma back)
                text = self._texts.get(entry_id, "")

                entry = VectorEntry(
                    id=entry_id,
                    embedding=stored_embedding,
                    text=text,
                    metadata=self._metadatas.get(entry_id, {}),
                )
                entries.append(entry)
                scores.append(score)

        return QueryResult(
            entries=entries,
            scores=scores,
            query=query_vector,
        )

    def delete(self, entry_ids: list[str]) -> None:
        """Delete entries from the store.

        Args:
            entry_ids: List of entry IDs to delete.
        """
        self._collection.delete(ids=entry_ids)
        for entry_id in entry_ids:
            self._texts.pop(entry_id, None)
            self._metadatas.pop(entry_id, None)

    def persist(self, path: str) -> None:
        """Persist the store to disk.

        Note: Chroma with PersistentClient auto-persists. This method
        is included for interface compatibility.

        Args:
            path: Path to persist to (if not set in constructor).
        """
        if self.persist_directory is None and path:
            # Recreate with persist directory
            self.persist_directory = path
            self._client = chromadb.PersistentClient(path=path)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance_metric},
            )

    def stats(self) -> VectorStoreStats:
        """Get statistics about the store."""
        return VectorStoreStats(
            total_entries=self._collection.count(),
            dimension=0,  # Chroma doesn't expose dimension easily
            store_type=VectorStoreType.CHROMA,
            metadata={
                "collection_name": self.collection_name,
                "distance_metric": self.distance_metric,
                "persist_directory": self.persist_directory,
            },
        )
