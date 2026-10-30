"""Integration tests requiring OpenCLIP model loading and forward pass.

SOFTWARE SMOKE TESTS ONLY:
These integration tests verify code execution plumbing and tensor shapes.
They use trivial in-memory color squares and must NEVER be treated as calibration data,
evaluation data, accuracy benchmarks, or campus performance metrics.
Do not add synthetic images to data/ directories.

These tests are marked with `@pytest.mark.integration` and are skipped during
fast offline unit testing. Run with `pytest -m integration` to execute.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from campus_hunt_cv.config import CampusHuntConfig, ModelConfig, ScoringConfig
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.model import OpenCLIPLocationEmbedder
from campus_hunt_cv.verifier import LocationVerifier, verify_location
from tests.conftest import create_test_image_bytes


@pytest.mark.integration
def test_openclip_embedder_real_forward_pass():
    """Verify that OpenCLIPLocationEmbedder initializes on CPU and computes 512D unit vector."""
    config = ModelConfig(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    embedder = OpenCLIPLocationEmbedder(config=config)

    test_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    emb = embedder.embed_image(test_img)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (512,)
    assert emb.dtype == np.float32

    # Verify unit L2 norm
    l2_norm = np.linalg.norm(emb)
    assert np.isclose(l2_norm, 1.0, atol=1e-4)


@pytest.mark.integration
def test_full_pipeline_end_to_end_synthetic(tmp_path: Path):
    """End-to-end integration test with real model embeddings on synthetic reference data."""
    config = ModelConfig(
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
        device="cpu",
    )
    embedder = OpenCLIPLocationEmbedder(config=config)

    # Generate two distinct synthetic reference images
    img_a = Image.new("RGB", (224, 224), color=(255, 0, 0))
    img_b = Image.new("RGB", (224, 224), color=(0, 0, 255))

    emb_a = embedder.embed_image(img_a).reshape(1, -1)
    emb_b = embedder.embed_image(img_b).reshape(1, -1)

    npz_path = tmp_path / "embeddings.npz"
    meta_path = tmp_path / "metadata.json"

    store = ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"red_room": emb_a, "blue_room": emb_b},
        model_name=config.model_name,
        pretrained=config.pretrained,
    )

    hunt_cfg = CampusHuntConfig()
    hunt_cfg.scoring = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.05,
    )

    verifier = LocationVerifier(config=hunt_cfg, embedder=embedder, store=store)

    # Test photo matching red_room
    red_photo_bytes = create_test_image_bytes(width=224, height=224, color=(250, 5, 5))
    res = verifier.verify(red_photo_bytes, target_location_id="red_room")

    assert res["result"] == "strong_match"
    assert res["best_matching_location_id"] == "red_room"
    assert res["best_competing_location_id"] == "blue_room"
    assert res["margin"] > 0

