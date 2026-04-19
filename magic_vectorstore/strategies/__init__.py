"""All vector store strategies."""

from magic_vectorstore.strategies.chroma import ChromaVectorStore
from magic_vectorstore.strategies.faiss import FAISSVectorStore

__all__ = ["ChromaVectorStore", "FAISSVectorStore"]
