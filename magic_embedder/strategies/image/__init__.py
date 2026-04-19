"""Image embedding strategies.

Note: Some strategies require optional dependencies that may not be installed.
Attempting to use them will raise an ImportError with installation instructions.
"""

__all__ = ["CLIPImageEmbedder"]

_CLIPImageEmbedder = None


def __getattr__(name: str):
    """Lazy import to avoid hard dependency on optional packages."""
    global _CLIPImageEmbedder
    if name == "CLIPImageEmbedder":
        if _CLIPImageEmbedder is None:
            try:
                from .clip_image_embedder import CLIPImageEmbedder

                _CLIPImageEmbedder = CLIPImageEmbedder
            except ImportError:
                raise ImportError(
                    "CLIPImageEmbedder requires the 'clip' package. "
                    "Install with: pip install git+https://github.com/openai/CLIP.git"
                )
        return _CLIPImageEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
