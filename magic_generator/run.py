#!/usr/bin/env python3
"""Entry point for magic-generator.

Usage:
    python -m magic_generator.run --query <query> --strategy <strategy> [options]
"""

import argparse
import sys
from pathlib import Path

from magic_generator.core import GenerationPrompt
from magic_generator.strategies import AnthropicGenerator, OpenAIGenerator

GENERATOR_MAP = {
    "openai": OpenAIGenerator,
    "anthropic": AnthropicGenerator,
}


def main():
    parser = argparse.ArgumentParser(description="magic-generator: LLM generation tool")
    parser.add_argument("--query", "-q", type=str, required=True, help="Query text")
    parser.add_argument("--context", "-c", type=str, nargs="+", help="Context strings")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file")
    parser.add_argument(
        "--strategy", "-s",
        choices=["openai", "anthropic"],
        default="openai",
        help="Generation strategy (default: openai)",
    )
    parser.add_argument("--model", "-m", type=str, help="Model name (strategy-dependent)")

    args = parser.parse_args()

    # Note: In real usage, generator would be initialized with API key
    print(f"Query: {args.query}")
    print(f"Strategy: {args.strategy}")
    if args.context:
        print(f"Context: {len(args.context)} chunks")
    print("Note: Run this via the RAG pipeline with proper API key setup")


if __name__ == "__main__":
    main()
