#!/usr/bin/env python3
"""Entry point for magic-chunker.

Usage:
    python -m magic_chunker.run --input <path> --strategy <strategy> [options]
"""

import argparse
import json
import sys
from pathlib import Path

from magic_chunker.core import ChunkType, ChunkingResult, Node
from magic_chunker.strategies import (
    FixedSizeChunker,
    HTMLChunker,
    MarkdownChunker,
    PDFChunker,
    SemanticChunker,
    SentenceChunker,
)

CHUNKER_MAP = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceChunker,
    "semantic": SemanticChunker,
    "markdown": MarkdownChunker,
    "pdf": PDFChunker,
    "html": HTMLChunker,
}


def load_input(path: Path) -> list[Node]:
    """Load input file and convert to nodes.

    Supports JSON format with list of dicts containing 'content' and optional 'metadata'.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = []
    for i, item in enumerate(data):
        content = item.get("content", "")
        metadata = item.get("metadata", {})
        node_type_str = item.get("node_type", "text")

        try:
            node_type = ChunkType(node_type_str)
        except ValueError:
            node_type = ChunkType.TEXT

        node = Node(
            id=item.get("id", f"node_{i}"),
            content=content,
            node_type=node_type,
            metadata=metadata,
        )
        nodes.append(node)

    return nodes


def save_output(result: ChunkingResult, path: Path) -> None:
    """Save chunking result to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="magic-chunker: Document chunking tool")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file (default: stdout)")
    parser.add_argument(
        "--strategy", "-s",
        choices=["fixed", "sentence", "semantic", "markdown", "pdf", "html"],
        default="semantic",
        help="Chunking strategy (default: semantic)",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size (for fixed/sentence)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap size (for fixed/sentence)")

    args = parser.parse_args()

    # Load input
    nodes = load_input(args.input)

    # Create chunker
    chunker_cls = CHUNKER_MAP[args.strategy]

    if args.strategy in ("fixed", "sentence"):
        chunker = chunker_cls(chunk_size=args.chunk_size, overlap=args.overlap)
    elif args.strategy in ("markdown", "pdf", "html"):
        chunker = chunker_cls()
    else:
        chunker = chunker_cls()

    # Chunk
    result = chunker.chunk(nodes)

    # Output
    if args.output:
        save_output(result, args.output)
    else:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
