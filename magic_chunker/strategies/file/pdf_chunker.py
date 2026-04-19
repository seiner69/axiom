"""PDF-specific chunking strategy."""

from magic_chunker.core import BaseChunker, Chunk, ChunkType, ChunkingResult, Node


class PDFChunker(BaseChunker):
    """Chunks PDF content by pages and sections.

    Assumes PDF content has been pre-processed to extract text with
    page information stored in node metadata.

    Metadata expected in nodes:
        - page_number: int
        - section_title: str (optional)
        - file_path: str (optional)

    Attributes:
        chunk_by_page: Whether to respect page boundaries.
        max_page_gap: Maximum gap between pages to still consider same section.
        text_chunker: Chunker to use for page content.
    """

    def __init__(
        self,
        chunk_by_page: bool = True,
        max_page_gap: int = 1,
        text_chunker: BaseChunker | None = None,
    ):
        if max_page_gap < 0:
            raise ValueError("max_page_gap must be non-negative")

        self.chunk_by_page = chunk_by_page
        self.max_page_gap = max_page_gap
        self.text_chunker = text_chunker

    @property
    def name(self) -> str:
        return "pdf"

    @property
    def description(self) -> str:
        return f"PDF chunker (by_page={self.chunk_by_page})"

    def chunk(self, nodes: list[Node]) -> ChunkingResult:
        """Chunk nodes by PDF structure.

        Args:
            nodes: List of Nodes to chunk.

        Returns:
            ChunkingResult with PDF-aware chunks.
        """
        all_chunks: list[Chunk] = []
        chunk_counter = 0

        # Group nodes by page if metadata available
        page_groups: dict[int, list[Node]] = {}
        for node in nodes:
            page = node.metadata.get("page_number", 0)
            if page not in page_groups:
                page_groups[page] = []
            page_groups[page].append(node)

        if not page_groups or not self.chunk_by_page:
            # No page info or chunking disabled - treat as single group
            if self.text_chunker:
                result = self.text_chunker.chunk(nodes)
                for sub_chunk in result.chunks:
                    sub_chunk.id = f"chunk_{chunk_counter}"
                    sub_chunk.index = chunk_counter
                    all_chunks.append(sub_chunk)
                    chunk_counter += 1
            else:
                for node in nodes:
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
        else:
            # Process by pages
            sorted_pages = sorted(page_groups.keys())
            current_group_nodes: list[Node] = []
            last_page = None

            for page in sorted_pages:
                if last_page is not None and page - last_page > self.max_page_gap:
                    # Page gap too large - process current group
                    if current_group_nodes:
                        chunk_text = "\n\n".join(n.content for n in current_group_nodes)
                        if self.text_chunker:
                            sub_nodes = [Node(
                                id=f"page_{last_page}",
                                content=chunk_text,
                                node_type=ChunkType.TEXT,
                                metadata=current_group_nodes[0].metadata.copy(),
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
                                metadata=current_group_nodes[0].metadata.copy(),
                                node_ids=[n.id for n in current_group_nodes],
                                index=chunk_counter,
                            )
                            all_chunks.append(chunk)
                            chunk_counter += 1
                        current_group_nodes = []

                current_group_nodes.extend(page_groups[page])
                last_page = page

            # Process remaining group
            if current_group_nodes:
                chunk_text = "\n\n".join(n.content for n in current_group_nodes)
                if self.text_chunker:
                    sub_nodes = [Node(
                        id=f"page_{last_page}",
                        content=chunk_text,
                        node_type=ChunkType.TEXT,
                        metadata=current_group_nodes[0].metadata.copy(),
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
                        metadata=current_group_nodes[0].metadata.copy(),
                        node_ids=[n.id for n in current_group_nodes],
                        index=chunk_counter,
                    )
                    all_chunks.append(chunk)

        return ChunkingResult(
            chunks=all_chunks,
            metadata={
                "chunker": self.name,
                "chunk_by_page": self.chunk_by_page,
                "max_page_gap": self.max_page_gap,
                "text_chunker": self.text_chunker.name if self.text_chunker else None,
                "total_nodes": len(nodes),
                "total_chunks": len(all_chunks),
            },
        )
