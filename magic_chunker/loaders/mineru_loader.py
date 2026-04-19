"""MinerU output loader.

Loads and parses MinerU's output files into magic-chunker Node objects.

Supports two output formats:
- content_list.json: Structured JSON with type, bbox, page_idx, table_body
    → Recommended: preserves complete structure, table/header/text clearly separated
- .md file: Markdown formatted output
    → Use when markdown structure is needed (e.g., for MarkdownChunker)

Recommended pipeline for MinerU data:
    # Best: JSON loader + SemanticChunker
    loader = MinerUContentListLoader("content_list.json")
    nodes = loader.load()
    chunker = SemanticChunker(min_chunk_size=200, max_chunk_size=1000)
    result = chunker.chunk(nodes)

    # Alternative: MD loader + SemanticChunker (no MarkdownChunker)
    loader = MinerUMarkdownLoader("report.md")
    nodes = loader.load()
    chunker = SemanticChunker(min_chunk_size=200, max_chunk_size=1000)
    result = chunker.chunk(nodes)
"""

import json
import re
from pathlib import Path
from typing import Any

from magic_chunker.core import ChunkType, Node


# --- Shared helpers ---


def _is_meaningful_text(text: str) -> bool:
    """Return True if text is meaningful (not just whitespace/punctuation)."""
    return bool(text and text.strip())


# --- Loader for content_list.json ---


class MinerUContentListLoader:
    """Loader for MinerU's content_list.json format.

    Parses the structured JSON output from MinerU and converts each element
    into a Node with appropriate type and metadata.

    Input structure (content_list.json):
        {
            "type": "text" | "header" | "page_number" | "table" | "image" | "list",
            "text": "...",
            "bbox": [x0, y0, x1, y1],
            "page_idx": 0,
            "text_level": 1 | 2 | ...,   # optional, for text/header elements
        }

    Example:
        >>> loader = MinerUContentListLoader("path/to/content_list.json")
        >>> nodes = loader.load()
        >>> for node in nodes:
        ...     print(node.id, node.content[:50])
    """

    TYPE_TO_CHUNK_TYPE = {
        "text": ChunkType.TEXT,
        "header": ChunkType.TEXT,
        "table": ChunkType.TABLE,
        "image": ChunkType.IMAGE,
        "list": ChunkType.TEXT,
        "page_number": None,  # skip page numbers
    }

    def __init__(self, json_path: str | Path):
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"MinerU content_list.json not found: {self.json_path}")

    def _element_to_node(self, item: dict[str, Any], index: int) -> Node | None:
        """Convert a single content_list element to a Node."""
        elem_type = item.get("type")

        # Determine content - different types use different fields
        if elem_type == "table":
            # Tables store content in table_body, not text
            text = item.get("table_body", "")
            if not text:
                # Fall back to img_path if table_body is empty
                text = item.get("img_path", "")
        else:
            text = item.get("text", "")

        # Skip page numbers
        if elem_type == "page_number":
            return None

        # Determine chunk type
        chunk_type = self.TYPE_TO_CHUNK_TYPE.get(elem_type, ChunkType.TEXT)

        # Build metadata
        metadata: dict[str, Any] = {
            "source": self.json_path.stem,
            "elem_type": elem_type,
        }

        if "page_idx" in item:
            metadata["page_idx"] = item["page_idx"]

        bbox = item.get("bbox")
        if bbox:
            # Store bbox as string since ChromaDB doesn't accept list values in metadata
            metadata["bbox"] = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

        text_level = item.get("text_level")
        if text_level is not None:
            metadata["text_level"] = text_level

        # Additional type-specific metadata
        if elem_type == "header":
            metadata["is_header"] = True
        elif elem_type == "table":
            metadata["is_table"] = True
            if item.get("img_path"):
                metadata["img_path"] = item["img_path"]
            caption = item.get("table_caption")
            if caption:
                # table_caption can be a list, convert to string for ChromaDB compatibility
                metadata["table_caption"] = caption if isinstance(caption, str) else str(caption)
        elif elem_type == "image":
            img_path = item.get("img_path", "")
            if img_path:
                metadata["img_path"] = img_path
                # Use img_path as node content so it can be embedded or referenced
                if not text:
                    text = img_path
            if item.get("image_caption"):
                caption = item.get("image_caption")
                metadata["image_caption"] = caption if isinstance(caption, str) else str(caption)

        # Generate ID
        page = item.get("page_idx", 0)
        node_id = f"mineru_{page}_{index}"

        # Skip empty or meaningless text
        if not _is_meaningful_text(text):
            return None

        return Node(
            id=node_id,
            content=text,
            node_type=chunk_type,
            metadata=metadata,
        )

    def load(self) -> list[Node]:
        """Load all elements from content_list.json as Nodes.

        Returns:
            List of Node objects, ordered as they appear in the document.
        """
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected content_list.json to contain a list, got {type(data)}")

        nodes: list[Node] = []
        for i, item in enumerate(data):
            node = self._element_to_node(item, i)
            if node is not None:
                nodes.append(node)

        return nodes

    @property
    def base_dir(self) -> Path:
        """Return the base directory of the JSON file (for resolving relative paths)."""
        return self.json_path.parent

    def resolve_image_path(self, img_path: str | Path) -> Path:
        """Resolve a relative image path to an absolute path.

        Args:
            img_path: Relative path from content_list.json location (e.g., 'images/xxx.jpg').

        Returns:
            Absolute path to the image file.
        """
        path = Path(img_path)
        if path.is_absolute():
            return path
        return (self.base_dir / path).resolve()

    def get_image_nodes(self) -> list[Node]:
        """Return only image-type nodes from the loaded content.

        Returns:
            List of Node objects with node_type == IMAGE.
        """
        return [n for n in self.load() if n.node_type == ChunkType.IMAGE]

    def get_image_paths(self) -> list[tuple[str, Path]]:
        """Return all image paths with their node IDs.

        Returns:
            List of (node_id, absolute_path) tuples for each image node.
        """
        nodes = self.get_image_nodes()
        result = []
        for node in nodes:
            img_path = node.metadata.get("img_path", node.content)
            full_path = self.resolve_image_path(img_path)
            result.append((node.id, full_path))
        return result


# --- Loader for .md output ---


# Markdown header pattern: # ## ### etc.
MD_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# Table pattern: <table>...</table>
MD_TABLE_PATTERN = re.compile(r"<table>[\s\S]*?</table>", re.MULTILINE)
# Horizontal rule: --- or *** or ___
MD_HR_PATTERN = re.compile(r"^[-*_]{3,}$", re.MULTILINE)


class MinerUMarkdownLoader:
    """Loader for MinerU's markdown (.md) output.

    Parses the markdown output and converts it to Nodes, preserving
    section structure via headers and table boundaries.

    Example:
        >>> loader = MinerUMarkdownLoader("path/to/report.pdf.md")
        >>> nodes = loader.load()
        >>> for node in nodes:
        ...     print(node.id, node.content[:50])
    """

    def __init__(
        self,
        md_path: str | Path,
        include_tables: bool = True,
        include_headers: bool = True,
    ):
        self.md_path = Path(md_path)
        if not self.md_path.exists():
            raise FileNotFoundError(f"MinerU markdown file not found: {self.md_path}")
        self.include_tables = include_tables
        self.include_headers = include_headers

    def _extract_tables(self, text: str) -> list[tuple[str, int, int]]:
        """Extract HTML tables with positions in the text."""
        tables = []
        for match in MD_TABLE_PATTERN.finditer(text):
            tables.append((match.group(), match.start(), match.end()))
        return tables

    def _build_node_id(self, page_hint: str | None, index: int) -> str:
        """Build a deterministic node ID."""
        prefix = page_hint or self.md_path.stem
        return f"{prefix}_{index}"

    def load(self) -> list[Node]:
        """Load all content from markdown file as Nodes.

        Strategy:
        - Split by header lines (##, ###, etc.) to build sections
        - Each section is emitted as a single Node with full section content
          (including header marker), preserving markdown structure for downstream chunkers
        - Tables are emitted as separate TABLE nodes
        - If include_headers=True, header lines themselves are also emitted as TEXT nodes

        Returns:
            List of Node objects in document order.
        """
        with open(self.md_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract tables and remove them from main text
        tables = self._extract_tables(text) if self.include_tables else []

        nodes: list[Node] = []
        node_counter = 0

        # Remove tables from text for section processing
        text_no_tables = text
        for table, start, end in reversed(tables):
            text_no_tables = text_no_tables[:start] + text_no_tables[end:]

        lines = text_no_tables.split("\n")

        # Current section state
        current_section_lines: list[str] = []
        current_section_start_line = 0
        current_section_level = 0
        current_header_text = ""

        def flush_section(
            lines_to_flush: list[str],
            start_line: int,
            section_level: int,
            header_title: str,
        ) -> None:
            """Emit the accumulated section as a single Node with full markdown content."""
            nonlocal node_counter
            section_text = "\n".join(lines_to_flush).strip()
            if not section_text:
                return

            node = Node(
                id=self._build_node_id(None, node_counter),
                content=section_text,
                node_type=ChunkType.TEXT,
                metadata={
                    "source": self.md_path.stem,
                    "first_line": start_line,
                    "text_level": section_level,
                    "section_header": header_title,
                },
            )
            nodes.append(node)
            node_counter += 1

        for line_num, line in enumerate(lines):
            header_match = MD_HEADER_PATTERN.match(line.strip())

            if header_match:
                # Flush current section before starting new header
                if current_section_lines:
                    flush_section(
                        current_section_lines,
                        current_section_start_line,
                        current_section_level,
                        current_header_text,
                    )
                    current_section_lines = []

                header_level = len(header_match.group(1))
                header_text = header_match.group(2).strip()

                # Emit header as its own node if requested
                if self.include_headers:
                    node = Node(
                        id=self._build_node_id(None, node_counter),
                        content=header_text,
                        node_type=ChunkType.TEXT,
                        metadata={
                            "source": self.md_path.stem,
                            "elem_type": "header",
                            "text_level": header_level,
                            "line_num": line_num,
                        },
                    )
                    nodes.append(node)
                    node_counter += 1

                # Start new section AFTER this header (content follows the header)
                current_section_lines = []
                current_section_start_line = line_num + 1
                current_section_level = header_level
                current_header_text = header_text
            else:
                current_section_lines.append(line)

        # Flush last section
        if current_section_lines:
            flush_section(
                current_section_lines,
                current_section_start_line,
                current_section_level,
                current_header_text,
            )

        # Add tables as separate nodes
        for table_html, _, _ in tables:
            node = Node(
                id=self._build_node_id(None, node_counter),
                content=table_html,
                node_type=ChunkType.TABLE,
                metadata={
                    "source": self.md_path.stem,
                    "elem_type": "table",
                },
            )
            nodes.append(node)
            node_counter += 1

        return nodes
