"""OpenCLIP embedding model wrapper for location photos."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol, Union
import numpy as np
from PIL import Image

from campus_hunt_cv.config import ModelConfig
from campus_hunt_cv.exceptions import ModelInferenceError
from campus_hunt_cv.image_validation import validate_and_decode_image

logger = logging.getLogger(__name__)


class EmbedderProtocol(Protocol):
    """Protocol interface for embedding models to enable zero-dependency unit tests."""

    @property
    def model_version(self) -> str:
        ...

    def embed_image(self, image: Image.Image) -> np.ndarray:
        ...

    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        ...

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        ...


def resolve_device(device_str: str) -> str:
    """Resolve compute device string ('auto', 'cuda', 'mps', 'cpu') to an available PyTorch device."""
    import torch

    device_normalized = device_str.lower().strip()
    if device_normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device_normalized == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    if device_normalized == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return "cpu"

    return device_normalized


class OpenCLIPLocationEmbedder:
    """Production OpenCLIP model wrapper for generating normalized image embeddings.

    - Loads the model once during initialization or lazy loading.
    - Sets model to eval mode and uses inference mode during forward passes.
    - Ensures identical preprocessing for reference images and submitted photos.
    - Generates strictly unit L2-normalized float32 embeddings.
    """

    def __init__(self, config: Optional[ModelConfig] = None, lazy_load: bool = False):
        self.config = config or ModelConfig()
        self._device: Optional[str] = None
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None

        if not lazy_load:
            self.load_model()

    @property
    def model_version(self) -> str:
        """Identifying string for the current model architecture and weights."""
        return self.config.model_version

    @property
    def device(self) -> str:
        """Resolved compute device string."""
        if self._device is None:
            self._device = resolve_device(self.config.device)
        return self._device

    def load_model(self) -> None:
        """Load OpenCLIP model and official preprocessing transform.

        Sets evaluation mode immediately. Does not re-initialize if already loaded.
        """
        if self._model is not None and self._preprocess is not None:
            return

        import torch
        import open_clip

        self._device = resolve_device(self.config.device)

        if self._device == "cpu" and self.config.num_threads is not None:
            torch.set_num_threads(self.config.num_threads)
            logger.info("Configured PyTorch CPU thread count: %d", self.config.num_threads)

        logger.info(
            "Loading OpenCLIP model '%s' (pretrained='%s') on device '%s'...",
            self.config.model_name,
            self.config.pretrained,
            self._device,
        )

        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name=self.config.model_name,
                pretrained=self.config.pretrained,
                device=self._device,
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            logger.info("OpenCLIP model successfully loaded in evaluation mode.")
        except Exception as e:
            raise ModelInferenceError(
                f"Failed to load OpenCLIP model '{self.config.model_name}' "
                f"with weights '{self.config.pretrained}': {str(e)}",
                reason_code="ERROR_MODEL_LOAD_FAILED",
            ) from e

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Apply official preprocessing and compute a normalized embedding vector for a single PIL image.

        Args:
            image: Pre-validated PIL image in RGB mode.

        Returns:
            Unit L2-normalized 1D float32 numpy array of shape (D,).
        """
        self.load_model()
        import torch

        try:
            with torch.inference_mode():
                tensor = self._preprocess(image).unsqueeze(0).to(self._device)
                features = self._model.encode_image(tensor)
                # Normalize embedding to unit L2 norm
                norm = features.norm(dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=1e-12)
                normalized_features = features / norm
                embedding = normalized_features.squeeze(0).cpu().numpy().astype(np.float32)
                return embedding
        except Exception as e:
            raise ModelInferenceError(
                f"Model forward pass failed during image embedding: {str(e)}",
                reason_code="ERROR_EMBEDDING_INFERENCE_FAILED",
            ) from e

    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Validate, decode, and compute normalized embedding for raw image bytes."""
        image = validate_and_decode_image(image_bytes)
        return self.embed_image(image)

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Compute normalized embeddings for a batch of PIL images.

        Args:
            images: List of PIL images in RGB mode.

        Returns:
            Unit L2-normalized 2D float32 numpy array of shape (N, D).
        """
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        self.load_model()
        import torch

        try:
            tensors = [self._preprocess(img) for img in images]
            batch_tensor = torch.stack(tensors).to(self._device)

            with torch.inference_mode():
                features = self._model.encode_image(batch_tensor)
                norm = features.norm(dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=1e-12)
                normalized_features = features / norm
                embeddings = normalized_features.cpu().numpy().astype(np.float32)
                return embeddings
        except Exception as e:
            raise ModelInferenceError(
                f"Model forward pass failed during batch embedding: {str(e)}",
                reason_code="ERROR_BATCH_EMBEDDING_FAILED",
            ) from e

