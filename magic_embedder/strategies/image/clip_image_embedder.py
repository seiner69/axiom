"""CLIP image embedding strategy using transformers."""

from magic_embedder.core import BaseImageEmbedder, ImageEmbeddingResult

try:
    import torch
    from PIL import Image
except ImportError:
    raise ImportError(
        "torch and PIL are required for CLIPImageEmbedder. "
        "Install with: pip install torch Pillow"
    )

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    raise ImportError(
        "transformers is required for CLIPImageEmbedder. "
        "Install with: pip install transformers"
    )


# Model dimension map
MODEL_DIMENSIONS = {
    "openai/clip-vit-base-patch32": 512,
    "openai/clip-vit-large-patch14": 768,
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": 512,
}


class CLIPImageEmbedder(BaseImageEmbedder):
    """CLIP image embedder using HuggingFace transformers.

    Uses OpenAI CLIP (via transformers) to generate image embeddings.

    Attributes:
        model_name: HuggingFace model name for CLIP.
        device: Device to use ('cpu', 'cuda').
        batch_size: Batch size for encoding.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        self._model = CLIPModel.from_pretrained(model_name).to(device)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model.eval()

    @property
    def name(self) -> str:
        return f"clip_{self.model_name.replace('/', '_')}"

    @property
    def description(self) -> str:
        return f"CLIP image embedder ({self.model_name})"

    @property
    def dimension(self) -> int:
        if self.model_name in MODEL_DIMENSIONS:
            return MODEL_DIMENSIONS[self.model_name]
        # Try to get from model config
        try:
            return self._model.config.projection_dim
        except Exception:
            return 512

    def embed_images(self, image_paths: list[str]) -> ImageEmbeddingResult:
        """Embed images using CLIP.

        Args:
            image_paths: List of image file paths to embed.

        Returns:
            ImageEmbeddingResult with embedding vectors.
        """
        import os

        all_embeddings: list[list[float]] = []

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []

            for path in batch_paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Image not found: {path}")
                img = Image.open(path).convert("RGB")
                images.append(img)

            # Process batch
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)
                # Normalize
                outputs = outputs / outputs.norm(dim=-1, keepdim=True)

            for embedding in outputs:
                all_embeddings.append(embedding.float().cpu().numpy().tolist())

        return ImageEmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.name,
            dimension=self.dimension,
            image_paths=image_paths,
            metadata={
                "batch_size": self.batch_size,
                "device": self._device,
                "total_images": len(image_paths),
            },
        )
