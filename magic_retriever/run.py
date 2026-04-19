#!/usr/bin/env python3
"""Entry point for magic-retriever.

Usage:
    python -m magic_retriever.run --query <query> --strategy <strategy> [options]
"""

import argparse
import sys
from pathlib import Path

from magic_retriever.strategies import MMRRetriever, SimilarityRetriever


def main():
    parser = argparse.ArgumentParser(description="magic-retriever: Retrieval tool")
    parser.add_argument("--query", "-q", type=str, required=True, help="Query text")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument(
        "--strategy", "-s",
        choices=["similarity", "mmr"],
        default="similarity",
        help="Retrieval strategy (default: similarity)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--lambda-mult", type=float, default=0.5, help="MMR lambda (default: 0.5)")

    args = parser.parse_args()

    # Note: In real usage, embedder and vector_store would be set up here
    print(f"Query: {args.query}")
    print(f"Strategy: {args.strategy}")
    print(f"Top-k: {args.top_k}")
    print("Note: Run this via the RAG pipeline with proper embedder/store setup")


if __name__ == "__main__":
    main()
