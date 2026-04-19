"""Utility functions for magic-chunker."""

import hashlib


def generate_id(content: str, prefix: str = "node") -> str:
    """Generate a unique ID from content.

    Args:
        content: Content to hash.
        prefix: Prefix for the ID.

    Returns:
        A unique ID string.
    """
    hash_val = hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"{prefix}_{hash_val}"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length.

    Args:
        text: Text to truncate.
        max_length: Maximum length.
        suffix: Suffix to append if truncated.

    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix
