"""Utility functions for magic-vectorstore."""

from typing import Any


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity score (between -1 and 1).
    """
    dot = sum(va * vb for va, vb in zip(a, b))
    norm_a = sum(va**2 for va in a) ** 0.5
    norm_b = sum(vb**2 for vb in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_vector(vector: list[float]) -> list[float]:
    """Normalize a vector to unit length.

    Args:
        vector: Vector to normalize.

    Returns:
        Normalized vector.
    """
    norm = sum(v**2 for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]
