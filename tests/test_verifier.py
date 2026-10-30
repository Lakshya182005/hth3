"""Unit tests for the LocationVerifier module and verify_location public entrypoint."""

import numpy as np
import pytest

from campus_hunt_cv.config import CampusHuntConfig, ScoringConfig
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.verifier import LocationVerifier, verify_location
from tests.conftest import MockEmbedder, create_test_image_bytes


def test_verification_contract_schema(mock_reference_store: ReferenceEmbeddingsStore):
    """The return dictionary must strictly match the specified schema."""
    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=CampusHuntConfig(),
        embedder=embedder,
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes(width=80, height=80)
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    # Required keys according to contract
    required_keys = {
        "result",
        "target_score",
        "best_matching_location_id",
        "best_matching_score",
        "best_competing_location_id",
        "best_competing_score",
        "margin",
        "reason_code",
        "model_version",
    }
    assert set(result.keys()) == required_keys
    assert result["result"] in ("strong_match", "ambiguous", "mismatch")
    assert isinstance(result["reason_code"], str)
    assert isinstance(result["model_version"], str)


def test_missing_reference_data_returns_ambiguous():
    """If no reference embeddings exist, return ambiguous with clear error code."""
    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=CampusHuntConfig(),
        embedder=embedder,
        store=None,  # No store loaded
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    assert result["result"] == "ambiguous"
    assert result["reason_code"] == "ERROR_NO_REFERENCE_DATA_LOADED"
    assert result["target_score"] is None
    assert result["best_matching_location_id"] is None


def test_invalid_image_bytes_handled_cleanly(mock_reference_store: ReferenceEmbeddingsStore):
    """Corrupt or non-image bytes must return an error reason_code without raising unhandled exceptions."""
    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=CampusHuntConfig(),
        embedder=embedder,
        store=mock_reference_store,
    )

    result = verifier.verify(image_bytes=b"corrupted_payload", target_location_id="clock_tower")

    assert result["result"] == "ambiguous"
    assert result["reason_code"].startswith("ERROR_IMAGE_")
    assert result["target_score"] is None


def test_uncalibrated_defaults_to_ambiguous(
    uncalibrated_config: CampusHuntConfig,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """When thresholds are not calibrated, decision must conservatively default to ambiguous."""
    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=uncalibrated_config,
        embedder=embedder,
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    assert result["result"] == "ambiguous"
    assert result["reason_code"] == "UNCALIBRATED_THRESHOLDS"
    # Even though ambiguous, target_score and best match should still be computed and reported
    assert result["target_score"] is not None
    assert result["best_matching_location_id"] is not None


def test_calibrated_strong_match_scenario(
    calibrated_config: CampusHuntConfig,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """When submitted image embedding closely matches the target location cluster, return strong_match."""
    # Retrieve clock_tower reference embedding and create a near-identical query
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]
    query_vector = clock_ref + np.random.RandomState(1).randn(64).astype(np.float32) * 0.01
    query_vector /= np.linalg.norm(query_vector)

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return query_vector

    verifier = LocationVerifier(
        config=calibrated_config,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    assert result["result"] == "strong_match"
    assert result["reason_code"] == "STRONG_MATCH"
    assert result["best_matching_location_id"] == "clock_tower"
    assert result["target_score"] is not None
    assert result["target_score"] >= calibrated_config.scoring.match_threshold
    assert result["margin"] is not None
    assert result["margin"] >= calibrated_config.scoring.margin_threshold


def test_calibrated_mismatch_scenario(
    calibrated_config: CampusHuntConfig,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """When participant claims clock_tower but photo clearly matches library_atrium, return mismatch."""
    library_ref = mock_reference_store.get_location_embeddings("library_atrium")[0]
    query_vector = library_ref.copy()

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return query_vector

    verifier = LocationVerifier(
        config=calibrated_config,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    # Team's server-authoritative location is clock_tower, but submitted photo is library_atrium
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    assert result["result"] == "mismatch"
    assert result["reason_code"] == "MISMATCH_COMPETING_LOCATION_MATCHED"
    assert result["best_matching_location_id"] == "library_atrium"
    assert result["best_competing_location_id"] == "library_atrium"
    assert result["margin"] < 0  # Negative margin because target lost


def test_unknown_target_location_in_calibrated_mode(
    calibrated_config: CampusHuntConfig,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """If target location does not exist in reference data, return ambiguous."""
    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=calibrated_config,
        embedder=embedder,
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="nonexistent_building")

    assert result["result"] == "ambiguous"
    assert result["reason_code"] == "UNKNOWN_TARGET_LOCATION"
    assert result["target_score"] is None


def test_standalone_verify_location_function(
    calibrated_config: CampusHuntConfig,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """Public helper function verify_location should work with explicit verifier injection."""
    embedder = MockEmbedder(dimension=64)
    custom_verifier = LocationVerifier(
        config=calibrated_config,
        embedder=embedder,
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verify_location(
        image_bytes=image_bytes,
        target_location_id="clock_tower",
        verifier=custom_verifier,
    )

    assert "result" in result
    assert result["model_version"] == embedder.model_version

