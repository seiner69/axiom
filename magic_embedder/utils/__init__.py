"""Utility functions for magic-embedder."""

import numpy as np


def normalize_embeddings(embeddings: list[list[float]]) -> list[list[float]]:
    """Normalize embeddings to unit length.

    Args:
        embeddings: List of embedding vectors.

    Returns:
        Normalized embedding vectors.
    """
    result = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm > 0:
            result.append([v / norm for v in emb])
        else:
            result.append(emb)
    return result


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score (between -1 and 1).
    """
    dot = sum(va * vb for va, vb in zip(a, b))
    norm_a = sum(va ** 2 for va in a) ** 0.5
    norm_b = sum(vb ** 2 for vb in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def batch_items(items: list, batch_size: int) -> list[list]:
    """Split items into batches.

    Args:
        items: List of items to batch.
        batch_size: Size of each batch.

    Returns:
        List of batches.
    """
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
