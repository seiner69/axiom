"""Fixed-size text chunking strategy."""

from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size pieces with optional overlap.

    Attributes:
        chunk_size: Target size of each chunk in characters.
        overlap: Number of overlapping characters between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def name(self) -> str:
        return "fixed_size"

    @property
    def description(self) -> str:
        return f"Fixed-size chunker (size={self.chunk_size}, overlap={self.overlap})"

    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """Chunk nodes into fixed-size pieces.

        Args:
            nodes: List of Nodes to chunk.

        Returns:
            ChunkingResult with fixed-size chunks.
        """
        all_chunks: list[Chunk] = []
        chunk_counter = 0

        for node in nodes:
            if node.node_type != ChunkType.TEXT:
                # Non-text nodes are kept as single chunks
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

            text = node.content
            start = 0
            text_len = len(text)

            while start < text_len:
                end = start + self.chunk_size
                chunk_text = text[start:end]

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

                if end >= text_len:
                    break

                start = end - self.overlap

        return ChunkingResult(
            chunks=all_chunks,
            metadata={
                "chunker": self.name,
                "chunk_size": self.chunk_size,
                "overlap": self.overlap,
                "total_nodes": len(nodes),
                "total_chunks": len(all_chunks),
            },
        )
