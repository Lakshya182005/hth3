"""Pytest test fixtures and mocks."""

import io
from typing import Dict, List, Optional
import numpy as np
from PIL import Image
import pytest

from campus_hunt_cv.config import CampusHuntConfig, ScoringConfig
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.model import EmbedderProtocol
from campus_hunt_cv.verifier import LocationVerifier


def create_test_image_bytes(
    width: int = 100,
    height: int = 100,
    color: tuple = (120, 150, 180),
    format: str = "JPEG",
    mode: str = "RGB",
) -> bytes:
    """Helper to create valid in-memory image bytes."""
    img = Image.new(mode, (width, height), color=color)
    bio = io.BytesIO()
    # JPEG does not support RGBA; save as PNG if RGBA
    save_format = "PNG" if mode == "RGBA" else format
    img.save(bio, format=save_format)
    return bio.getvalue()


class MockEmbedder(EmbedderProtocol):
    """Deterministic mock embedder that requires zero model downloads or GPU resources."""

    def __init__(
        self,
        dimension: int = 512,
        embedding_map: Optional[Dict[str, np.ndarray]] = None,
        default_vector: Optional[np.ndarray] = None,
        model_version_str: str = "ViT-B-32/mock_test_weights",
    ):
        self.dimension = dimension
        self.embedding_map = embedding_map or {}
        self.model_version_str = model_version_str

        if default_vector is not None:
            self.default_vector = default_vector / np.linalg.norm(default_vector)
        else:
            vec = np.ones((dimension,), dtype=np.float32)
            self.default_vector = vec / np.linalg.norm(vec)

    @property
    def model_version(self) -> str:
        return self.model_version_str

    def embed_image(self, image: Image.Image) -> np.ndarray:
        # Generate pseudo-deterministic embedding from image size and average pixel
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        mean_val = float(np.mean(arr))
        # Create deterministic unit vector
        rng = np.random.RandomState(int(mean_val) % 1000)
        vec = rng.randn(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_image_bytes(self, image_bytes: bytes) -> np.ndarray:
        bio = io.BytesIO(image_bytes)
        img = Image.open(bio)
        return self.embed_image(img)

    def embed_batch(self, images: List[Image.Image]) -> np.ndarray:
        return np.vstack([self.embed_image(img) for img in images])


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder(dimension=64)


@pytest.fixture
def mock_reference_store() -> ReferenceEmbeddingsStore:
    """Create in-memory ReferenceEmbeddingsStore with 3 distinct locations."""
    dim = 64
    rng = np.random.RandomState(42)

    # Base orthogonal or well-separated vectors for 3 locations
    v_clock = rng.randn(dim).astype(np.float32)
    v_clock /= np.linalg.norm(v_clock)

    v_library = rng.randn(dim).astype(np.float32)
    v_library /= np.linalg.norm(v_library)

    v_science = rng.randn(dim).astype(np.float32)
    v_science /= np.linalg.norm(v_science)

    # 3 reference photos each with slight perturbations
    def make_cluster(center: np.ndarray, count: int = 3, noise: float = 0.05) -> np.ndarray:
        photos = []
        for i in range(count):
            p = center + rng.randn(dim).astype(np.float32) * noise
            p /= np.linalg.norm(p)
            photos.append(p)
        return np.vstack(photos)

    embeddings = {
        "clock_tower": make_cluster(v_clock, count=3),
        "library_atrium": make_cluster(v_library, count=3),
        "science_quad": make_cluster(v_science, count=2),
    }

    metadata = {
        "schema_version": "1.0",
        "created_at": "2026-10-30T00:00:00Z",
        "model_name": "ViT-B-32",
        "pretrained": "mock_test_weights",
        "feature_dimension": dim,
        "normalized": True,
        "total_images": 8,
        "total_locations": 3,
        "locations": {
            "clock_tower": {"count": 3, "files": ["c1.jpg", "c2.jpg", "c3.jpg"]},
            "library_atrium": {"count": 3, "files": ["l1.jpg", "l2.jpg", "l3.jpg"]},
            "science_quad": {"count": 2, "files": ["s1.jpg", "s2.jpg"]},
        },
    }

    return ReferenceEmbeddingsStore(embeddings_by_location=embeddings, metadata=metadata)


@pytest.fixture
def calibrated_config() -> CampusHuntConfig:
    """Config with calibrated thresholds set."""
    cfg = CampusHuntConfig()
    cfg.scoring = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.08,
        mismatch_threshold=0.30,
        top_k=3,
    )
    return cfg


@pytest.fixture
def uncalibrated_config() -> CampusHuntConfig:
    """Config with default uncalibrated state."""
    cfg = CampusHuntConfig()
    cfg.scoring = ScoringConfig(
        calibrated=False,
        match_threshold=None,
        margin_threshold=None,
        top_k=3,
    )
    return cfg

