"""Semantic text chunking strategy."""

import re
from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node


# Paragraph boundaries: double newlines or specific separators
PARAGRAPH_PATTERN = re.compile(r"\n\s*\n|<!--.*?-->|\n\n", re.DOTALL)
# Section boundaries: markdown headers or major punctuation
SECTION_PATTERN = re.compile(r"^#{1,6}\s+.+$|^\d+\.\s+.+$|^[A-Z][^.!?]*$", re.MULTILINE)


class SemanticChunker(BaseChunker):
    """Chunks text by semantic boundaries (paragraphs and sections).

    Attempts to keep related content together while splitting on natural
    semantic boundaries like paragraphs, sections, and headers.

    Attributes:
        min_chunk_size: Minimum characters in a chunk before forcing merge.
        max_chunk_size: Maximum characters in a chunk before forcing split.
        split_on_headers: Whether to split on markdown headers.
        merge_small_chunks: Whether to merge chunks smaller than min_chunk_size.
    """

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        split_on_headers: bool = True,
        merge_small_chunks: bool = True,
    ):
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        if min_chunk_size > max_chunk_size:
            raise ValueError("min_chunk_size must be <= max_chunk_size")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.split_on_headers = split_on_headers
        self.merge_small_chunks = merge_small_chunks

    @property
    def name(self) -> str:
        return "semantic"

    @property
    def description(self) -> str:
        return (
            f"Semantic chunker (min={self.min_chunk_size}, "
            f"max={self.max_chunk_size}, headers={self.split_on_headers})"
        )

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        paragraphs = PARAGRAPH_PATTERN.split(text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_by_headers(self, text: str) -> list[str]:
        """Split text by markdown headers."""
        lines = text.split("\n")
        sections: list[str] = []
        current_section: list[str] = []

        for line in lines:
            if re.match(r"^#{1,6}\s+", line):
                # Save current section if non-empty
                if current_section:
                    sections.append("\n".join(current_section))
                    current_section = []
            current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append("\n".join(current_section))

        return [s.strip() for s in sections if s.strip()]

    def _smart_split(self, text: str) -> list[str]:
        """Split text using semantic awareness."""
        if self.split_on_headers:
            sections = self._split_by_headers(text)
            if len(sections) > 1:
                result: list[str] = []
                for section in sections:
                    result.extend(self._split_paragraphs(section))
                return result

        return self._split_paragraphs(text)

    def _merge_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Merge small chunks with neighbors."""
        if not chunks:
            return chunks

        merged: list[Chunk] = []
        buffer = ""

        for chunk in chunks:
            if not buffer:
                buffer = chunk.content
            elif len(buffer) + len(chunk.content) <= self.max_chunk_size:
                buffer += "\n\n" + chunk.content
            else:
                merged.append(Chunk(
                    id=f"merged_{len(merged)}",
                    content=buffer,
                    chunk_type=chunk.chunk_type,
                    metadata=chunk.metadata.copy(),
                    node_ids=chunk.node_ids.copy(),
                    index=len(merged),
                ))
                buffer = chunk.content

        if buffer:
            merged.append(Chunk(
                id=f"merged_{len(merged)}",
                content=buffer,
                chunk_type=chunks[-1].chunk_type,
                metadata=chunks[-1].metadata.copy(),
                node_ids=chunks[-1].node_ids.copy(),
                index=len(merged),
            ))

        return merged

    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """Chunk nodes by semantic boundaries.

        Args:
            nodes: List of Nodes to chunk.

        Returns:
            ChunkingResult with semantically grouped chunks.
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

            segments = self._smart_split(node.content)
            for segment in segments:
                if not segment.strip():
                    continue

                chunk = Chunk(
                    id=f"chunk_{chunk_counter}",
                    content=segment,
                    chunk_type=ChunkType.TEXT,
                    metadata=node.metadata.copy(),
                    node_ids=[node.id],
                    index=chunk_counter,
                )
                all_chunks.append(chunk)
                chunk_counter += 1

        # Merge small chunks if enabled
        if self.merge_small_chunks:
            all_chunks = self._merge_chunks(all_chunks)
            # Re-index chunks
            for i, chunk in enumerate(all_chunks):
                chunk.index = i

        return ChunkingResult(
            chunks=all_chunks,
            metadata={
                "chunker": self.name,
                "min_chunk_size": self.min_chunk_size,
                "max_chunk_size": self.max_chunk_size,
                "split_on_headers": self.split_on_headers,
                "merge_small_chunks": self.merge_small_chunks,
                "total_nodes": len(nodes),
                "total_chunks": len(all_chunks),
            },
        )
