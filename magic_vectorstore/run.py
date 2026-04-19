#!/usr/bin/env python3
"""Entry point for magic-vectorstore.

Usage:
    python -m magic_vectorstore.run --input <path> --strategy <strategy> --action <action> [options]
"""

import argparse
import json
import sys
from pathlib import Path

from magic_vectorstore.core import VectorEntry
from magic_vectorstore.strategies import ChromaVectorStore, FAISSVectorStore

STORE_MAP = {
    "chroma": ChromaVectorStore,
    "faiss": FAISSVectorStore,
}


def load_entries(path: Path) -> list[VectorEntry]:
    """Load entries from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for item in data.get("entries", []):
        entry = VectorEntry(
            id=item["id"],
            embedding=item["embedding"],
            text=item.get("text"),
            metadata=item.get("metadata", {}),
        )
        entries.append(entry)

    return entries


def main():
    parser = argparse.ArgumentParser(description="magic-vectorstore: Vector storage tool")
    parser.add_argument("--input", "-i", type=Path, help="Input JSON file with entries")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file for results")
    parser.add_argument(
        "--strategy", "-s",
        choices=["chroma", "faiss"],
        default="chroma",
        help="Vector store strategy (default: chroma)",
    )
    parser.add_argument(
        "--action", "-a",
        choices=["add", "search", "stats"],
        default="add",
        help="Action to perform",
    )
    parser.add_argument("--persist", "-p", type=str, help="Persist path")
    parser.add_argument("--query-vector", type=str, help="Query vector (JSON list)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")

    args = parser.parse_args()

    # Create store
    store_cls = STORE_MAP[args.strategy]

    if args.strategy == "faiss":
        store = store_cls(dimension=384)
    else:
        store = store_cls(collection_name="default")

    if args.action == "add" and args.input:
        entries = load_entries(args.input)
        store.add(entries)
        print(f"Added {len(entries)} entries")

        if args.persist:
            store.persist(args.persist)
            print(f"Persisted to {args.persist}")

    elif args.action == "search":
        if not args.query_vector:
            print("Error: --query-vector is required for search", file=sys.stderr)
            sys.exit(1)

        query = json.loads(args.query_vector)
        result = store.search(query, top_k=args.top_k)

        output = result.to_dict()
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(output, ensure_ascii=False, indent=2))

    elif args.action == "stats":
        stats = store.stats()
        print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
