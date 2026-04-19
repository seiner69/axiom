#!/usr/bin/env python3
"""Entry point for magic-embedder.

Usage:
    python -m magic_embedder.run --input <path> --strategy <strategy> [options]
"""

import argparse
import json
import sys
from pathlib import Path

from magic_embedder.core import TextEmbeddingResult
from magic_embedder.strategies import (
    CLIPImageEmbedder,
    OpenAITextEmbedder,
    SentenceTransformerEmbedder,
)

TEXT_EMBEDDER_MAP = {
    "openai": OpenAITextEmbedder,
    "sentence_transformer": SentenceTransformerEmbedder,
}

IMAGE_EMBEDDER_MAP = {
    "clip": CLIPImageEmbedder,
}


def main():
    parser = argparse.ArgumentParser(description="magic-embedder: Text/image embedding tool")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file (default: stdout)")
    parser.add_argument(
        "--strategy", "-s",
        choices=["openai", "sentence_transformer", "clip"],
        default="sentence_transformer",
        help="Embedding strategy (default: sentence_transformer)",
    )
    parser.add_argument("--model", "-m", type=str, help="Model name (strategy-dependent)")
    parser.add_argument(
        "--type", "-t",
        choices=["text", "image"],
        default="text",
        help="Input type (default: text)",
    )

    args = parser.parse_args()

    # Load input
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = data.get("texts", [])
    image_paths = data.get("image_paths", [])

    # Create embedder
    if args.type == "text":
        if not texts:
            print("Error: No texts found in input", file=sys.stderr)
            sys.exit(1)

        embedder_cls = TEXT_EMBEDDER_MAP[args.strategy]
        if args.strategy == "openai":
            embedder = embedder_cls(model=args.model or "text-embedding-3-small")
        else:
            embedder = embedder_cls(model_name=args.model or "all-MiniLM-L6-v2")

        result = embedder.embed(texts)
        output = result.to_dict()

    else:
        if not image_paths:
            print("Error: No image_paths found in input", file=sys.stderr)
            sys.exit(1)

        embedder_cls = IMAGE_EMBEDDER_MAP[args.strategy]
        embedder = embedder_cls(model_name=args.model or "ViT-B/16")
        result = embedder.embed_images(image_paths)
        output = result.to_dict()

    # Output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
