"""Markdown-specific chunking strategy."""

import re
from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node


# Markdown structural patterns
HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
TABLE_PATTERN = re.compile(r"\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+")
LIST_PATTERN = re.compile(r"^[\s]*[-*+]\s+|^[\s]*\d+\.\s+", re.MULTILINE)


class MarkdownChunker(BaseChunker):
    """Chunks markdown files by structural elements.

    Preserves the hierarchy of markdown documents by chunking at:
    - Header boundaries (h1-h6)
    - Code blocks (kept intact)
    - Tables (kept intact)
    - List items

    Attributes:
        max_header_level: Maximum header level to chunk on (e.g., 2 means chunk on h1, h2).
        preserve_code_blocks: Whether to keep code blocks as single chunks.
        preserve_tables: Whether to keep tables as single chunks.
        text_chunker: Chunker to use for text content.
    """

    def __init__(
        self,
        max_header_level: int = 2,
        preserve_code_blocks: bool = True,
        preserve_tables: bool = True,
        text_chunker: BaseChunker | None = None,
    ):
        if max_header_level < 1 or max_header_level > 6:
            raise ValueError("max_header_level must be between 1 and 6")

        self.max_header_level = max_header_level
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        self.text_chunker = text_chunker

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def description(self) -> str:
        return f"Markdown chunker (max_header_level={self.max_header_level})"

    def _extract_code_blocks(self, text: str) -> list[tuple[str, int, int]]:
        """Extract code blocks with their positions."""
        blocks = []
        for match in CODE_BLOCK_PATTERN.finditer(text):
            blocks.append((match.group(), match.start(), match.end()))
        return blocks

    def _extract_tables(self, text: str) -> list[tuple[str, int, int]]:
        """Extract tables with their positions."""
        blocks = []
        for match in TABLE_PATTERN.finditer(text):
            blocks.append((match.group(), match.start(), match.end()))
        return blocks

    def _chunk_by_headers(self, text: str) -> list[str]:
        """Split text by headers up to max_header_level."""
        lines = text.split("\n")
        chunks: list[str] = []
        current_chunk_lines: list[str] = []
        current_level = 0
        skip_levels: set[int] = set()

        for line in lines:
            header_match = HEADER_PATTERN.match(line)
            if header_match:
                level = len(header_match.group(1))
                if level <= self.max_header_level:
                    # Save current chunk
                    if current_chunk_lines:
                        chunks.append("\n".join(current_chunk_lines))
                        current_chunk_lines = []
                    current_level = level
                    skip_levels.discard(level)
                else:
                    # Skip content under deeper headers
                    skip_levels.add(level)
                    continue
            else:
                if current_level in skip_levels:
                    continue
                current_chunk_lines.append(line)

        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))

        return [c.strip() for c in chunks if c.strip()]

    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """Chunk nodes by markdown structure.

        Args:
            nodes: List of Nodes to chunk.

        Returns:
            ChunkingResult with markdown-aware chunks.
        """
        all_chunks: list[Chunk] = []
        chunk_counter = 0
        code_counter = 0
        table_counter = 0

        for node in nodes:
            text = node.content

            # Extract and remove code blocks
            code_blocks = []
            if self.preserve_code_blocks:
                code_blocks = self._extract_code_blocks(text)
                for block, start, end in reversed(code_blocks):
                    text = text[:start] + text[end:]

            # Extract and remove tables
            tables = []
            if self.preserve_tables:
                tables = self._extract_tables(text)
                for table, start, end in reversed(tables):
                    text = text[:start] + text[end:]

            # Add code blocks as separate chunks
            if self.preserve_code_blocks:
                for block, _, _ in code_blocks:
                    chunk = Chunk(
                        id=f"chunk_{chunk_counter}",
                        content=block,
                        chunk_type=ChunkType.CODE,
                        metadata=node.metadata.copy(),
                        node_ids=[node.id],
                        index=chunk_counter,
                    )
                    all_chunks.append(chunk)
                    chunk_counter += 1

            # Add tables as separate chunks
            if self.preserve_tables:
                for table, _, _ in tables:
                    chunk = Chunk(
                        id=f"chunk_{chunk_counter}",
                        content=table,
                        chunk_type=ChunkType.TABLE,
                        metadata=node.metadata.copy(),
                        node_ids=[node.id],
                        index=chunk_counter,
                    )
                    all_chunks.append(chunk)
                    chunk_counter += 1

            # Chunk remaining text by headers
            header_chunks = self._chunk_by_headers(text)
            for chunk_text in header_chunks:
                if not chunk_text.strip():
                    continue

                # Further chunk if text_chunker is provided
                if self.text_chunker:
                    sub_nodes = [Node(
                        id=f"sub_{node.id}",
                        content=chunk_text,
                        node_type=ChunkType.TEXT,
                        metadata=node.metadata.copy(),
                    )]
                    sub_result = self.text_chunker.chunk(sub_nodes)
                    for sub_chunk in sub_result.chunks:
                        sub_chunk.id = f"chunk_{chunk_counter}"
                        sub_chunk.index = chunk_counter
                        all_chunks.append(sub_chunk)
                        chunk_counter += 1
                else:
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

        return ChunkingResult(
            chunks=all_chunks,
            metadata={
                "chunker": self.name,
                "max_header_level": self.max_header_level,
                "preserve_code_blocks": self.preserve_code_blocks,
                "preserve_tables": self.preserve_tables,
                "text_chunker": self.text_chunker.name if self.text_chunker else None,
                "total_nodes": len(nodes),
                "total_chunks": len(all_chunks),
            },
        )
