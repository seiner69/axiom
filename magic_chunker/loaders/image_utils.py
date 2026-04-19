"""Image embedding utilities for MinerU data.

Provides helpers to embed images extracted by MinerUContentListLoader
using a image embedder (e.g., CLIPImageEmbedder).

Example:
    >>> from magic_chunker.loaders import MinerUContentListLoader
    >>> from magic_chunker.loaders.image_utils import embed_mineru_images
    >>> from magic_embedder.strategies import CLIPImageEmbedder
    >>>
    >>> loader = MinerUContentListLoader("content_list.json")
    >>> embedder = CLIPImageEmbedder(model_name="ViT-B/16")
    >>> entries = embed_mineru_images(loader, embedder)
    >>> print(f"Embedded {len(entries)} images")
"""


def embed_mineru_images(loader, image_embedder, skip_missing: bool = True):
    """Embed images from a MinerUContentListLoader.

    Args:
        loader: A MinerUContentListLoader instance (must have `get_image_paths()` method).
        image_embedder: An image embedder with `embed_images(paths) -> ImageEmbeddingResult`.
            Expected to have `embed_images` method and `name` / `dimension` properties.
        skip_missing: If True, skip image paths that don't exist on disk.

    Returns:
        List of VectorEntry objects, one per image. Each entry has:
        - id: node ID from the loader
        - embedding: CLIP embedding vector
        - text: image caption if available, else img_path
        - metadata: includes img_path, page_idx, bbox

    Raises:
        FileNotFoundError: If an image file is not found and skip_missing=False.
    """
    # Import at runtime to avoid circular dependency
    from magic_vectorstore.core import VectorEntry

    image_paths_data = loader.get_image_paths()
    if not image_paths_data:
        return []

    node_ids = []
    abs_paths = []

    for node_id, full_path in image_paths_data:
        if not full_path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"Image not found: {full_path}")
        node_ids.append(node_id)
        abs_paths.append(str(full_path))

    if not abs_paths:
        return []

    # Embed images
    embed_result = image_embedder.embed_images(abs_paths)

    # Build vector entries
    # Get caption from metadata of image nodes
    image_nodes = loader.get_image_nodes()
    id_to_caption = {}
    for n in image_nodes:
        cap = n.metadata.get("image_caption", "")
        id_to_caption[n.id] = cap if cap else n.metadata.get("img_path", "")

    entries = []
    for i, node_id in enumerate(node_ids):
        caption = id_to_caption.get(node_id, "")
        entries.append(
            VectorEntry(
                id=node_id,
                embedding=embed_result.embeddings[i],
                text=caption,
                metadata={
                    "img_path": abs_paths[i],
                    "source": loader.json_path.stem,
                    "chunk_type": "image",
                },
            )
        )

    return entries
