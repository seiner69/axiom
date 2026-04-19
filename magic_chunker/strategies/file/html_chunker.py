"""HTML-specific chunking strategy."""

import re
from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node


# HTML tag patterns
BLOCK_TAGS = re.compile(
    r"<(?:article|aside|div|h[1-6]|header|footer|main|nav|section|blockquote|li|tr|p)[^>]*>",
    re.IGNORECASE,
)
HEADING_TAGS = re.compile(r"<(?:h[1-6])[^>]*>(.*?)</h[1-6]>", re.IGNORECASE | re.DOTALL)
SCRIPT_STYLE = re.compile(r"<(?:script|style)[^>]*>.*?</(?:script|style)>", re.IGNORECASE | re.DOTALL)


class HTMLChunker(BaseChunker):
    """Chunks HTML content by DOM structure.

    Preserves the structure of HTML documents by chunking at:
    - Heading tags (h1-h6)
    - Block-level elements (div, section, article, etc.)
    - Paragraphs

    Attributes:
        max_heading_level: Maximum heading level to chunk on.
        remove_scripts: Whether to remove script and style tags.
        text_chunker: Chunker to use for text content.
    """

    def __init__(
        self,
        max_heading_level: int = 2,
        remove_scripts: bool = True,
        text_chunker: BaseChunker | None = None,
    ):
        if max_heading_level < 1 or max_heading_level > 6:
            raise ValueError("max_heading_level must be between 1 and 6")

        self.max_heading_level = max_heading_level
        self.remove_scripts = remove_scripts
        self.text_chunker = text_chunker

    @property
    def name(self) -> str:
        return "html"

    @property
    def description(self) -> str:
        return f"HTML chunker (max_heading_level={self.max_heading_level})"

    def _clean_html(self, text: str) -> str:
        """Remove script and style tags."""
        if self.remove_scripts:
            return SCRIPT_STYLE.sub("", text)
        return text

    def _extract_headings(self, text: str) -> list[tuple[str, int]]:
        """Extract headings with their levels."""
        headings = []
        for match in HEADING_TAGS.finditer(text):
            tag = match.group(0)
            level = int(re.search(r"h(\d)", tag, re.IGNORECASE).group(1))
            content = match.group(1).strip()
            headings.append((content, level))
        return headings

    def _split_by_blocks(self, text: str) -> list[str]:
        """Split text by block-level HTML elements."""
        parts = BLOCK_TAGS.split(text)
        return [p.strip() for p in parts if p.strip()]

    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """Chunk nodes by HTML structure.

        Args:
            nodes: List of Nodes to chunk.

        Returns:
            ChunkingResult with HTML-aware chunks.
        """
        all_chunks: list[Chunk] = []
        chunk_counter = 0

        for node in nodes:
            text = self._clean_html(node.content)

            # Extract headings for structure info
            headings = self._extract_headings(text)
            has_structure = any(level <= self.max_heading_level for _, level in headings)

            if has_structure:
                # Split by blocks and process
                blocks = self._split_by_blocks(text)
                for block in blocks:
                    if not block.strip():
                        continue

                    if self.text_chunker:
                        sub_nodes = [Node(
                            id=f"html_{node.id}",
                            content=block,
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
                            content=block,
                            chunk_type=ChunkType.TEXT,
                            metadata=node.metadata.copy(),
                            node_ids=[node.id],
                            index=chunk_counter,
                        )
                        all_chunks.append(chunk)
                        chunk_counter += 1
            else:
                # No clear structure - use text_chunker or single chunk
                if self.text_chunker:
                    sub_result = self.text_chunker.chunk([node])
                    for sub_chunk in sub_result.chunks:
                        sub_chunk.id = f"chunk_{chunk_counter}"
                        sub_chunk.index = chunk_counter
                        all_chunks.append(sub_chunk)
                        chunk_counter += 1
                else:
                    chunk = Chunk(
                        id=f"chunk_{chunk_counter}",
                        content=text,
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
                "max_heading_level": self.max_heading_level,
                "remove_scripts": self.remove_scripts,
                "text_chunker": self.text_chunker.name if self.text_chunker else None,
                "total_nodes": len(nodes),
                "total_chunks": len(all_chunks),
            },
        )
