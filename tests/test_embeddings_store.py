"""Unit tests for ReferenceEmbeddingsStore safe storage and retrieval."""

import json
from pathlib import Path
import numpy as np
import pytest

from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.exceptions import ReferenceDataError


def test_save_and_load_roundtrip(tmp_path: Path):
    """Store should save to NPZ/JSON and reload with identical values and allow_pickle=False."""
    npz_path = tmp_path / "embeddings.npz"
    meta_path = tmp_path / "metadata.json"

    # Create synthetic embeddings
    loc_a = np.array([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)
    loc_b = np.array([[0.0, 1.0]], dtype=np.float32)
    embeddings = {"loc_a": loc_a, "loc_b": loc_b}

    store = ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location=embeddings,
        model_name="ViT-B-32",
        pretrained="mock_checkpoint",
        per_location_files={"loc_a": ["a1.jpg", "a2.jpg"], "loc_b": ["b1.jpg"]},
    )

    assert npz_path.exists()
    assert meta_path.exists()

    # Verify reload
    reloaded = ReferenceEmbeddingsStore.load(npz_path=npz_path, metadata_path=meta_path)
    assert reloaded.locations == ["loc_a", "loc_b"]
    assert reloaded.get_location_count("loc_a") == 2
    assert reloaded.get_location_count("loc_b") == 1
    assert reloaded.has_location("loc_a") is True
    assert reloaded.has_location("loc_nonexistent") is False

    np.testing.assert_allclose(reloaded.get_location_embeddings("loc_a"), loc_a)
    np.testing.assert_allclose(reloaded.get_location_embeddings("loc_b"), loc_b)


def test_load_missing_npz_raises(tmp_path: Path):
    """Missing NPZ file should raise actionable ReferenceDataError."""
    npz_path = tmp_path / "missing.npz"
    meta_path = tmp_path / "meta.json"
    meta_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ReferenceDataError) as exc_info:
        ReferenceEmbeddingsStore.load(npz_path=npz_path, metadata_path=meta_path)
    assert exc_info.value.reason_code == "ERROR_REFERENCE_EMBEDDINGS_NOT_FOUND"


def test_load_missing_metadata_raises(tmp_path: Path):
    """Missing metadata JSON file should raise actionable ReferenceDataError."""
    npz_path = tmp_path / "emb.npz"
    npz_path.touch()
    meta_path = tmp_path / "missing_meta.json"

    with pytest.raises(ReferenceDataError) as exc_info:
        ReferenceEmbeddingsStore.load(npz_path=npz_path, metadata_path=meta_path)
    assert exc_info.value.reason_code == "ERROR_REFERENCE_METADATA_NOT_FOUND"


def test_load_invalid_npz_schema(tmp_path: Path):
    """NPZ file missing required keys should raise ReferenceDataError."""
    npz_path = tmp_path / "bad.npz"
    meta_path = tmp_path / "meta.json"
    meta_path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")

    np.savez(npz_path, wrong_key=np.array([1, 2, 3]))

    with pytest.raises(ReferenceDataError) as exc_info:
        ReferenceEmbeddingsStore.load(npz_path=npz_path, metadata_path=meta_path)
    assert exc_info.value.reason_code == "ERROR_NPZ_INVALID_SCHEMA"

