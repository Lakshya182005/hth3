"""Regression tests verifying Fixes 1 through 6."""

import json
from pathlib import Path
import numpy as np
from pydantic import ValidationError
import pytest

from campus_hunt_cv.cli.calibrate import check_calibration_data_adequacy, evaluate_thresholds, run_calibration
from campus_hunt_cv.config import CampusHuntConfig, ScoringConfig
from campus_hunt_cv.decision import DecisionEngine
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.exceptions import ReferenceDataError
from campus_hunt_cv.similarity import SimilarityScorer
from campus_hunt_cv.verifier import LocationVerifier
from tests.conftest import MockEmbedder, create_test_image_bytes


# ============================================================================
# FIX 1: Calibration Config Load & Nested Schema Validation
# ============================================================================

def test_fix1_calibration_config_roundtrip(tmp_path: Path, mock_reference_store: ReferenceEmbeddingsStore):
    """Calibrated config written in nested format must load correctly and activate calibrated scoring."""
    config_path = tmp_path / "calibrated_thresholds.json"

    # Exact structure written by campus-hunt-calibrate
    config_dict = {
        "model": {
            "model_name": "ViT-B-32",
            "pretrained": "mock_test_weights",
            "device": "cpu",
        },
        "image_limits": {
            "max_image_bytes": 15728640,
            "max_image_pixels": 50000000,
            "min_image_dimension": 32,
            "allowed_formats": ["JPEG", "JPG", "PNG", "WEBP"],
        },
        "scoring": {
            "aggregation_method": "top_k_mean",
            "top_k": 3,
            "calibrated": True,
            "match_threshold": 0.65,
            "margin_threshold": 0.08,
            "mismatch_threshold": None,
            "min_reference_images_per_location": 1,
        },
        "storage": {
            "reference_data_dir": "data/reference",
            "calibration_data_dir": "data/calibration",
            "test_data_dir": "data/test",
            "embeddings_path": "data/reference_embeddings.npz",
            "metadata_path": "data/reference_embeddings.meta.json",
        },
        "calibration_metadata": {
            "model_version": "ViT-B-32/mock_test_weights",
            "calibrated_at": "2026-10-30T19:00:00Z",
            "calibration_sample_counts": {"clock_tower": 5, "unknown": 10},
            "calibration_metrics": {
                "correct_accept_rate": 0.95,
                "wrong_false_accept_rate": 0.01,
                "unknown_rejection_rate": 0.98,
                "correct_trials": 20,
                "correct_accepts": 19,
                "wrong_trials": 40,
                "wrong_false_accepts": 0,
                "unknown_trials": 10,
                "unknown_rejections": 10,
            },
            "test_evaluation": {
                "status": "unavailable",
                "reason": "No test captures found in data/test/",
            },
        },
    }

    config_path.write_text(json.dumps(config_dict, indent=2), encoding="utf-8")

    # Load via documented method
    loaded_config = CampusHuntConfig.from_json_file(config_path)

    # 1. Confirm fields have expected values
    assert loaded_config.scoring.calibrated is True
    assert loaded_config.scoring.match_threshold == 0.65
    assert loaded_config.scoring.margin_threshold == 0.08
    assert loaded_config.scoring.mismatch_threshold is None
    assert loaded_config.model.model_name == "ViT-B-32"
    assert loaded_config.calibration_metadata is not None
    assert loaded_config.calibration_metadata.calibration_metrics.correct_accept_rate == 0.95

    # 2. Confirm verifier uses these values rather than defaulting to uncalibrated behavior
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    verifier = LocationVerifier(
        config=loaded_config,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    # Must be strong_match, NOT ambiguous/UNCALIBRATED_THRESHOLDS
    assert result["result"] == "strong_match"
    assert result["reason_code"] == "STRONG_MATCH"


def test_fix1_misplaced_flat_fields_rejected_by_extra_forbid(tmp_path: Path):
    """A config with misplaced flat calibration fields must fail validation rather than silently ignoring them."""
    bad_config_path = tmp_path / "bad_flat_config.json"
    # Flat format that previously caused silent uncalibrated defaults
    flat_data = {
        "calibrated": True,
        "match_threshold": 0.70,
        "margin_threshold": 0.10,
        "model_version": "ViT-B-32/laion2b_s34b_b79k",
    }
    bad_config_path.write_text(json.dumps(flat_data), encoding="utf-8")

    with pytest.raises(ValidationError):
        CampusHuntConfig.from_json_file(bad_config_path)


# ============================================================================
# FIX 2: Separate Calibration from Final Evaluation & Inadequate Data
# ============================================================================

def test_fix2_calibration_adequacy_check(mock_reference_store: ReferenceEmbeddingsStore):
    """Calibration must fail if real campus captures are missing or inadequate."""
    # Reference store has: clock_tower, library_atrium, science_quad
    incomplete_counts = {
        "clock_tower": 2,
        # missing library_atrium and science_quad
        "unknown": 5,
    }

    is_adequate, missing = check_calibration_data_adequacy(
        store=mock_reference_store,
        sample_counts=incomplete_counts,
        min_samples_per_location=1,
    )

    assert is_adequate is False
    assert any("library_atrium" in m for m in missing)
    assert any("science_quad" in m for m in missing)


def test_fix2_missing_unknown_fails_adequacy(mock_reference_store: ReferenceEmbeddingsStore):
    """Calibration must fail if no unknown/distractor images are provided."""
    counts = {
        "clock_tower": 3,
        "library_atrium": 3,
        "science_quad": 2,
        # unknown is missing (count = 0)
    }

    is_adequate, missing = check_calibration_data_adequacy(
        store=mock_reference_store,
        sample_counts=counts,
        min_samples_per_location=1,
    )

    assert is_adequate is False
    assert any("unknown" in m for m in missing)


def test_fix2_empty_calibration_dir_refuses_to_write_config(tmp_path: Path):
    """Running calibration on an empty directory must return False and not write a config."""
    empty_calib = tmp_path / "calib"
    empty_calib.mkdir()
    empty_test = tmp_path / "test"
    empty_test.mkdir()
    out_cfg = tmp_path / "out.json"

    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    store = ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"clock_tower": np.ones((2, 64), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock",
    )

    success = run_calibration(
        calibration_dir=empty_calib,
        test_dir=empty_test,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
    )

    assert success is False
    assert not out_cfg.exists()


# ============================================================================
# FIX 3: No Unmeasured Mismatch Threshold & Progression Rule
# ============================================================================

def test_fix3_no_unmeasured_mismatch_threshold_inserted():
    """Calibrated config must have mismatch_threshold=None, never an unmeasured 0.30."""
    cfg = ScoringConfig(
        calibrated=True,
        match_threshold=0.65,
        margin_threshold=0.08,
        mismatch_threshold=None,
    )
    assert cfg.mismatch_threshold is None

    engine = DecisionEngine(config=cfg)
    # Target score is low (0.35), but competitor margin is not decisive
    # Because mismatch_threshold is None, this must remain conservative ambiguous, not an unmeasured mismatch
    res, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.35,
        best_matching_location_id="library_atrium",
        best_matching_score=0.40,
        best_competing_location_id="library_atrium",
        best_competing_score=0.40,
        margin=-0.05,  # competitor won by only 0.05 < 0.08 margin_threshold
        target_ref_count=3,
    )
    assert res == "ambiguous"
    assert reason == "AMBIGUOUS_BORDERLINE_COMPETITOR"


# ============================================================================
# FIX 4: Real Reference Image Counts & Minimum Policy
# ============================================================================

def test_fix4_decision_uses_real_reference_counts():
    """Locations with fewer than min_reference_images_per_location must return INSUFFICIENT_REFERENCE_DATA."""
    cfg = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.05,
        min_reference_images_per_location=3,  # Requires at least 3 reference images
    )
    engine = DecisionEngine(config=cfg)

    # Real count is 2 (< 3)
    res, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.90,
        best_matching_location_id="clock_tower",
        best_matching_score=0.90,
        best_competing_location_id="library_atrium",
        best_competing_score=0.40,
        margin=0.50,
        target_ref_count=2,  # Actual count from store is 2
    )
    assert res == "ambiguous"
    assert reason == "INSUFFICIENT_REFERENCE_DATA"


def test_fix4_evaluate_thresholds_honors_insufficient_reference_counts(mock_reference_store):
    """evaluate_thresholds must not accept locations with insufficient reference photos."""
    # mock_reference_store has: clock_tower=3, library_atrium=3, science_quad=2
    # If we require min_reference_images_per_location = 3:
    scoring_config = ScoringConfig(min_reference_images_per_location=3)
    scorer = SimilarityScorer()

    samples = [
        # Photo for science_quad (which has only 2 reference photos in store)
        {
            "actual": "science_quad",
            "file": "s1.jpg",
            "scores": {"clock_tower": 0.20, "library_atrium": 0.20, "science_quad": 0.95},
        }
    ]

    metrics = evaluate_thresholds(
        samples=samples,
        store=mock_reference_store,
        scorer=scorer,
        base_scoring_config=scoring_config,
        match_threshold=0.60,
        margin_threshold=0.05,
    )

    # science_quad has target_ref_count = 2 < 3 -> decision is ambiguous (INSUFFICIENT_REFERENCE_DATA)
    # It must NOT be accepted as a strong match!
    assert metrics["correct_accepts"] == 0
    assert metrics["correct_accept_rate"] == 0.0


# ============================================================================
# FIX 6: Model / Embedding Version Compatibility Checking
# ============================================================================

def test_fix6_store_load_mismatched_model_version_raises(tmp_path: Path):
    """Loading a store with expected_model_version differing from stored metadata must raise ReferenceDataError."""
    npz_path = tmp_path / "embeddings.npz"
    meta_path = tmp_path / "metadata.json"

    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"clock_tower": np.ones((2, 64), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="checkpoint_v1",
    )

    with pytest.raises(ReferenceDataError) as exc_info:
        ReferenceEmbeddingsStore.load(
            npz_path=npz_path,
            metadata_path=meta_path,
            expected_model_version="ViT-L-14/checkpoint_v2",  # Different model
        )
    assert exc_info.value.reason_code == "ERROR_MODEL_VERSION_MISMATCH"


def test_fix6_verifier_fails_safe_on_model_version_mismatch(mock_reference_store: ReferenceEmbeddingsStore):
    """Verifier must return ambiguous with ERROR_MODEL_VERSION_MISMATCH if store and embedder models differ."""
    # mock_reference_store was built with model_version "ViT-B-32/mock_test_weights"
    # Provide an embedder with a different model_version
    mismatched_embedder = MockEmbedder(
        dimension=64,
        model_version_str="ViT-L-14/other_weights",
    )

    verifier = LocationVerifier(
        config=CampusHuntConfig(),
        embedder=mismatched_embedder,
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    res = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    assert res["result"] == "ambiguous"
    assert res["reason_code"] == "ERROR_MODEL_VERSION_MISMATCH"
    assert res["target_score"] is None

