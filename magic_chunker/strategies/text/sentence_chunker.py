"""Sentence-based text chunking strategy."""

import re
from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node


class SentenceChunker(BaseChunker):
    """Chunks text by sentences, grouping them into chunks of target size.

    Attributes:
        chunk_size: Target number of sentences per chunk.
        overlap: Number of overlapping sentences between adjacent chunks.
        min_sentences: Minimum number of sentences in a chunk (last chunk may be smaller).
    """

    SENTENCE_ENDINGS = re.compile(r"(?<=[。！？；?!.])[\s\n]+")
    CHINESE_SENTENCE_ENDINGS = re.compile(r"(?<=[。！？])[\s\n]+")

    def __init__(
        self,
        chunk_size: int = 5,
        overlap: int = 1,
        min_sentences: int = 1,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if min_sentences <= 0:
            raise ValueError("min_sentences must be positive")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_sentences = min_sentences

    @property
    def name(self) -> str:
        return "sentence"

    @property
    def description(self) -> str:
        return f"Sentence chunker (size={self.chunk_size}, overlap={self.overlap})"

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Try Chinese endings first, then fall back to general
        sentences = self.CHINESE_SENTENCE_ENDINGS.split(text)
        if len(sentences) > 1:
            return [s.strip() for s in sentences if s.strip()]
        return [s.strip() for s in self.SENTENCE_ENDINGS.split(text) if s.strip()]

    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """Chunk nodes by sentences.

        Args:
            nodes: List of Nodes to chunk.

        Returns:
            ChunkingResult with sentence-based chunks.
        """
        all_chunks: list[Chunk] = []
        chunk_counter = 0

        for node in nodes:
            if node.node_type != ChunkType.TEXT:
                chunk = Chunk(
                    id=f"chunk_{chunk_counter}",
                    content=node.content,
                    chunk_type=node.node_type,
                    metadata=node.metadata.copy(),
                    node_ids=[node.id],
                    index=chunk_counter,
                )
                all_chunks.append(chunk)
                chunk_counter += 1
                continue

            sentences = self._split_sentences(node.content)
            if not sentences:
                continue

            # Group sentences into chunks
            start = 0
            while start < len(sentences):
                end = min(start + self.chunk_size, len(sentences))
                chunk_sentences = sentences[start:end]
                chunk_text = "".join(chunk_sentences)

                # Only create chunk if we meet minimum sentences or it's the last chunk
                is_last = end >= len(sentences)
                if len(chunk_sentences) >= self.min_sentences or is_last:
                    chunk = Chunk(
                        id=f"chunk_{chunk_counter}",
                        content=chunk_text,
                        chunk_type=ChunkType.TEXT,
                        metadata=node.metadata.copy(),
                        node_ids=[node.id],
                        index=chunk_counter,
                    )
                    all_chunks.append(chunk)
                    chunk_counter += 1

                if is_last:
                    break

                start = end - self.overlap

        return ChunkingResult(
            chunks=all_chunks,
            metadata={
                "chunker": self.name,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "min_sentences": self.min_sentences,
                "total_nodes": len(nodes),
                "total_chunks": len(all_chunks),
            },
        )
