"""Regression tests verifying Fixes 1 through 6, Fix A, and Fix B."""

import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from pydantic import ValidationError
import pytest

from campus_hunt_cv.cli.calibrate import (
    check_calibration_data_adequacy,
    check_split_data_adequacy,
    evaluate_thresholds,
    main as calibrate_main,
    run_calibration,
)
from campus_hunt_cv.config import (
    CalibrationMetadata,
    CalibrationMetrics,
    CampusHuntConfig,
    DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
    ModelConfig,
    ScoringConfig,
    TestEvaluationMetrics,
    normalize_environment,
)

# Tell pytest not to treat TestEvaluationMetrics as a pytest test suite class
TestEvaluationMetrics.__test__ = False
from campus_hunt_cv.decision import DecisionEngine
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.exceptions import ConfigurationError, ReferenceDataError
from campus_hunt_cv.similarity import SimilarityScorer
from campus_hunt_cv.verifier import LocationVerifier, check_config_validation_status
from tests.conftest import MockEmbedder, create_test_image_bytes


def populate_split(split_dir: Path, loc_counts: dict):
    """Helper to create minimal valid image files in location subdirectories."""
    for loc, count in loc_counts.items():
        loc_dir = split_dir / loc
        loc_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img_bytes = create_test_image_bytes(width=50, height=50)
            (loc_dir / f"img_{i}.jpg").write_bytes(img_bytes)


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
            "min_calibration_samples_per_location": 1,
            "min_calibration_unknown_samples": 1,
            "min_test_samples_per_location": 1,
            "min_test_unknown_samples": 1,
            "target_max_far": 0.02,
            "target_min_unknown_rejection": 0.90,
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
                "status": "completed",
                "reason": None,
                "sample_counts": {"clock_tower": 5, "unknown": 10},
                "missing_categories": None,
                "metrics": {
                    "correct_accept_rate": 0.94,
                    "wrong_false_accept_rate": 0.01,
                    "unknown_rejection_rate": 0.97,
                    "correct_trials": 20,
                    "correct_accepts": 19,
                    "wrong_trials": 40,
                    "wrong_false_accepts": 0,
                    "unknown_trials": 10,
                    "unknown_rejections": 10,
                },
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
# FIX 2 & FIX A: Calibration / Test Split Adequacy & Categorization
# ============================================================================

def test_fix2_calibration_adequacy_check(mock_reference_store: ReferenceEmbeddingsStore):
    """Calibration must fail if real campus captures are missing or inadequate."""
    incomplete_counts = {
        "clock_tower": 2,
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
    }

    is_adequate, missing = check_calibration_data_adequacy(
        store=mock_reference_store,
        sample_counts=counts,
        min_samples_per_location=1,
    )

    assert is_adequate is False
    assert any("unknown" in m for m in missing)


def test_calibration_allow_missing_unknown_passes_adequacy(mock_reference_store: ReferenceEmbeddingsStore):
    """When allow_missing_unknown is True, adequacy passes even without unknown directory."""
    counts = {
        "clock_tower": 5,
        "library_atrium": 5,
        "science_quad": 5,
    }

    # Strict mode fails
    is_adequate_strict, missing_strict = check_calibration_data_adequacy(
        store=mock_reference_store,
        sample_counts=counts,
        min_samples_per_location=5,
        min_unknown_samples=10,
        allow_missing_unknown=False,
    )
    assert is_adequate_strict is False
    assert any("unknown" in m for m in missing_strict)

    # Provisional mode passes
    is_adequate_prov, missing_prov = check_calibration_data_adequacy(
        store=mock_reference_store,
        sample_counts=counts,
        min_samples_per_location=5,
        min_unknown_samples=10,
        allow_missing_unknown=True,
    )
    assert is_adequate_prov is True
    assert missing_prov == []


def test_provisional_calibration_end_to_end_blocks_production_auto_accept(tmp_path: Path):
    """Calibrating with allow_missing_unknown generates provisional config with null unknown metrics; verifier blocks auto-accept in production."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    calib_dir.mkdir()
    test_dir.mkdir()

    # Create mock calibration and test images for 2 locations (no unknown)
    # loc_a has red channel ~ 100, loc_b has red channel ~ 200
    for loc in ["loc_a", "loc_b"]:
        (calib_dir / loc).mkdir()
        (test_dir / loc).mkdir()
        r_val = 100 if loc == "loc_a" else 200
        for i in range(5):
            img_bytes = create_test_image_bytes(width=50, height=50, color=(r_val, 100, 100))
            (calib_dir / loc / f"img_{i}.jpg").write_bytes(img_bytes)
            (test_dir / loc / f"img_{i}.jpg").write_bytes(img_bytes)

    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"
    out_cfg = tmp_path / "calibrated_thresholds.json"

    # Reference embeddings for loc_a and loc_b
    dim = 64
    rng = np.random.RandomState(42)
    v_a = rng.randn(dim).astype(np.float32)
    v_a /= np.linalg.norm(v_a)
    v_b = rng.randn(dim).astype(np.float32)
    v_b /= np.linalg.norm(v_b)

    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.stack([v_a, v_a]), "loc_b": np.stack([v_b, v_b])},
        model_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    )

    class ControlledEmbedder(MockEmbedder):
        def __init__(self, dimension):
            super().__init__(dimension=dimension, model_version_str="ViT-B-32/laion2b_s34b_b79k")

        def embed_image(self, image):
            arr = np.array(image.convert("RGB"))
            if arr[0, 0, 0] > 150:
                return v_b.copy()
            return v_a.copy()

    with patch("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", return_value=ControlledEmbedder(dimension=dim)):
        success = run_calibration(
            calibration_dir=calib_dir,
            test_dir=test_dir,
            embeddings_npz=npz_path,
            metadata_json=meta_path,
            output_config=out_cfg,
            allow_missing_unknown=True,
            candidate_match_thresholds=[0.50, 0.60],
            candidate_margins=[0.05],
        )

    assert success is True
    assert out_cfg.exists()

    config = CampusHuntConfig.from_json_file(out_cfg)
    assert config.scoring.calibrated is True
    assert config.calibration_metadata.status == "provisional"
    assert config.calibration_metadata.is_provisional is True
    assert config.calibration_metadata.calibration_metrics.unknown_rejection_rate is None
    assert config.calibration_metadata.calibration_metrics.unknown_trials is None
    assert config.calibration_metadata.calibration_metrics.unknown_rejections is None

    # Verify runtime behavior: In production, auto-accept MUST be blocked
    store = ReferenceEmbeddingsStore.load(npz_path=npz_path, metadata_path=meta_path)
    verifier = LocationVerifier(
        config=config,
        embedder=ControlledEmbedder(dimension=dim),
        store=store,
    )

    test_img = create_test_image_bytes(width=50, height=50)
    result = verifier.verify(image_bytes=test_img, target_location_id="loc_a")

    # In production, even if raw match succeeds, gate forces ambiguous
    assert result["result"] == "ambiguous"
    assert result["reason_code"] == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"
    # Scores are preserved for admin review
    assert result["target_score"] is not None
    assert result["best_matching_location_id"] == "loc_a"



def test_fix2_empty_calibration_dir_refuses_to_write_config(tmp_path: Path):
    """Running calibration on an empty directory must return False and not write a config."""
    empty_calib = tmp_path / "calib"
    empty_calib.mkdir()
    empty_test = tmp_path / "test"
    empty_test.mkdir()
    out_cfg = tmp_path / "out.json"

    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    ReferenceEmbeddingsStore.save(
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
# FIX A: Require a Complete Hold-Out Test Split
# ============================================================================

def test_fixA_test_split_adequacy_empty(mock_reference_store: ReferenceEmbeddingsStore):
    """When test split is empty, status must be 'unavailable' and metrics must be None."""
    sample_counts = {}
    is_complete, missing = check_split_data_adequacy(
        store=mock_reference_store,
        sample_counts=sample_counts,
        min_samples_per_location=1,
        min_unknown_samples=1,
    )
    assert is_complete is False
    assert any("clock_tower" in m for m in missing)
    assert any("library_atrium" in m for m in missing)
    assert any("science_quad" in m for m in missing)
    assert any("unknown" in m for m in missing)


def test_fixA_test_split_adequacy_partial(mock_reference_store: ReferenceEmbeddingsStore):
    """When test split contains some captures but is missing categories, status must be 'incomplete'."""
    partial_counts = {
        "clock_tower": 2,
        # missing library_atrium, science_quad, unknown
    }
    is_complete, missing = check_split_data_adequacy(
        store=mock_reference_store,
        sample_counts=partial_counts,
        min_samples_per_location=1,
        min_unknown_samples=1,
    )
    assert is_complete is False
    assert not any("clock_tower" in m for m in missing)
    assert any("library_atrium" in m for m in missing)
    assert any("science_quad" in m for m in missing)
    assert any("unknown" in m for m in missing)


def test_fixA_test_split_adequacy_complete(mock_reference_store: ReferenceEmbeddingsStore):
    """When test split contains all locations and unknown meeting minimum counts, status is 'completed'."""
    complete_counts = {
        "clock_tower": 2,
        "library_atrium": 1,
        "science_quad": 3,
        "unknown": 2,
    }
    is_complete, missing = check_split_data_adequacy(
        store=mock_reference_store,
        sample_counts=complete_counts,
        min_samples_per_location=1,
        min_unknown_samples=1,
    )
    assert is_complete is True
    assert len(missing) == 0


def test_fixA_configurable_data_adequacy_safeguards(mock_reference_store: ReferenceEmbeddingsStore):
    """Minimum counts are configurable data-quality safeguards."""
    counts = {
        "clock_tower": 2,
        "library_atrium": 2,
        "science_quad": 2,
        "unknown": 2,
    }
    # With min_samples=1, counts=2 passes
    assert check_split_data_adequacy(mock_reference_store, counts, min_samples_per_location=1, min_unknown_samples=1)[0] is True
    # With higher requirement (e.g. min_samples=3), counts=2 is detected as inadequate
    is_adequate, missing = check_split_data_adequacy(mock_reference_store, counts, min_samples_per_location=3, min_unknown_samples=1)
    assert is_adequate is False
    assert len(missing) == 3


def test_fixA_test_outcomes_integration(tmp_path: Path, monkeypatch):
    """Integration test checking that unavailable, incomplete, and completed test splits are recorded correctly."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "calib_out.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    # Reference store with 2 locations
    dim = 64
    rng = np.random.RandomState(42)
    v_a = rng.randn(dim).astype(np.float32)
    v_a /= np.linalg.norm(v_a)
    v_b = rng.randn(dim).astype(np.float32)
    v_b /= np.linalg.norm(v_b)

    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": v_a.reshape(1, -1), "loc_b": v_b.reshape(1, -1)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )

    # Populate adequate calibration set (meets new default safeguards: >=5 per location, >=10 unknown)
    populate_split(calib_dir, {"loc_a": 5, "loc_b": 5, "unknown": 10})

    # Mock embedder to avoid model downloads
    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    # 1. Test Split is Empty -> 'unavailable'
    test_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
        max_far=1.0,  # permissive to focus on test outcome
        min_unknown_rejection=0.0,
        base_config=base_cfg,
    )
    assert success is True
    loaded = CampusHuntConfig.from_json_file(out_cfg)
    assert loaded.calibration_metadata.test_evaluation.status == "unavailable"
    assert loaded.calibration_metadata.test_evaluation.metrics is None
    assert len(loaded.calibration_metadata.test_evaluation.missing_categories) > 0

    # 2. Test Split is Partial (has loc_a only, missing loc_b and unknown) -> 'incomplete'
    populate_split(test_dir, {"loc_a": 5})
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
        max_far=1.0,
        min_unknown_rejection=0.0,
        base_config=base_cfg,
    )
    assert success is True
    loaded = CampusHuntConfig.from_json_file(out_cfg)
    assert loaded.calibration_metadata.test_evaluation.status == "incomplete"
    assert loaded.calibration_metadata.test_evaluation.metrics is None
    missing_cats = loaded.calibration_metadata.test_evaluation.missing_categories
    assert any("loc_b" in m for m in missing_cats)
    assert any("unknown" in m for m in missing_cats)

    # 3. Test Split is Complete (has loc_a, loc_b, and unknown meeting >=5 and >=10) -> 'completed'
    populate_split(test_dir, {"loc_b": 5, "unknown": 10})
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
        max_far=1.0,
        min_unknown_rejection=0.0,
        base_config=base_cfg,
    )
    assert success is True
    loaded = CampusHuntConfig.from_json_file(out_cfg)
    assert loaded.calibration_metadata.test_evaluation.status == "completed"
    assert loaded.calibration_metadata.test_evaluation.metrics is not None
    assert loaded.calibration_metadata.test_evaluation.missing_categories is None


# ============================================================================
# FIX B: Fail Closed When No Threshold Meets the Required Limits
# ============================================================================

def test_fixB_calibration_fails_closed_when_constraints_unmet(tmp_path: Path, monkeypatch):
    """When no candidate threshold satisfies constraints, calibration must return False and NOT write config."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "must_not_be_created.json"
    diag_report = tmp_path / "diagnostic.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    populate_split(calib_dir, {"loc_a": 5, "unknown": 10})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    # Set impossible constraints: max_far = -1.0 (impossible to satisfy)
    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
        diagnostic_report=diag_report,
        max_far=-1.0,
        min_unknown_rejection=1.0,
        base_config=base_cfg,
    )

    # 1. Routine returns failure
    assert success is False
    # 2. No active calibrated config is created
    assert not out_cfg.exists()
    # 3. Diagnostic report is created, marked as non-deployable and calibrated=False
    assert diag_report.exists()
    diag_data = json.loads(diag_report.read_text(encoding="utf-8"))
    assert diag_data["deployable"] is False
    assert diag_data["calibrated"] is False
    assert diag_data["status"] == "calibration_failed_constraints_unmet"


def test_fixB_calibration_does_not_overwrite_existing_config_on_failure(tmp_path: Path, monkeypatch):
    """If an active config already exists, a failed calibration must NOT overwrite or modify it."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "existing_config.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    populate_split(calib_dir, {"loc_a": 5, "unknown": 10})

    # Pre-create an existing config with distinct signature
    original_config_content = json.dumps({"sentinel": "pre_existing_data", "timestamp": "12345"})
    out_cfg.write_text(original_config_content, encoding="utf-8")

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )

    # Attempt calibration with impossible FAR constraint
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
        max_far=-1.0,
        min_unknown_rejection=1.0,
        base_config=base_cfg,
    )

    assert success is False
    # File content must be byte-for-byte identical to original
    assert out_cfg.read_text(encoding="utf-8") == original_config_content


def test_fixB_cli_returns_nonzero_exit_code_on_failure(tmp_path: Path, monkeypatch):
    """The calibrate CLI main() must exit with non-zero exit code when constraints fail."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    populate_split(calib_dir, {"loc_a": 5, "unknown": 10})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    test_args = [
        "campus-hunt-calibrate",
        "--calibration-dir", str(calib_dir),
        "--test-dir", str(test_dir),
        "--embeddings-npz", str(npz_path),
        "--metadata-json", str(meta_path),
        "--output-config", str(tmp_path / "out.json"),
        "--max-far", "-1.0",  # Impossible constraint
    ]
    monkeypatch.setenv("CAMPUS_HUNT_PRETRAINED", "mock_test")
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        calibrate_main()
    assert exc_info.value.code != 0


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
    res, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.35,
        best_matching_location_id="library_atrium",
        best_matching_score=0.40,
        best_competing_location_id="library_atrium",
        best_competing_score=0.40,
        margin=-0.05,
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
        min_reference_images_per_location=3,
    )
    engine = DecisionEngine(config=cfg)

    res, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.90,
        best_matching_location_id="clock_tower",
        best_matching_score=0.90,
        best_competing_location_id="library_atrium",
        best_competing_score=0.40,
        margin=0.50,
        target_ref_count=2,  # Actual count from store is 2 < 3
    )
    assert res == "ambiguous"
    assert reason == "INSUFFICIENT_REFERENCE_DATA"


def test_fix4_evaluate_thresholds_honors_insufficient_reference_counts(mock_reference_store):
    """evaluate_thresholds must not accept locations with insufficient reference photos."""
    scoring_config = ScoringConfig(min_reference_images_per_location=3)
    scorer = SimilarityScorer()

    samples = [
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

    # science_quad has target_ref_count = 2 < 3 -> ambiguous
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
            expected_model_version="ViT-L-14/checkpoint_v2",
        )
    assert exc_info.value.reason_code == "ERROR_MODEL_VERSION_MISMATCH"


def test_fix6_verifier_fails_safe_on_model_version_mismatch(mock_reference_store: ReferenceEmbeddingsStore):
    """Verifier must return ambiguous with ERROR_MODEL_VERSION_MISMATCH if store and embedder models differ."""
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


# ============================================================================
# SAMPLE-COUNT SAFEGUARD UPDATES
# ============================================================================

def test_sample_safeguards_defaults_greater_than_one():
    """Verify default minimum sample counts are strictly greater than one."""
    cfg = ScoringConfig()
    assert cfg.min_calibration_samples_per_location == 5
    assert cfg.min_calibration_unknown_samples == 10
    assert cfg.min_test_samples_per_location == 5
    assert cfg.min_test_unknown_samples == 10

    assert cfg.min_calibration_samples_per_location > 1
    assert cfg.min_calibration_unknown_samples > 1
    assert cfg.min_test_samples_per_location > 1
    assert cfg.min_test_unknown_samples > 1

    # Verify constant exports
    assert DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION == 5
    assert DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES == 10
    assert DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION == 5
    assert DEFAULT_MIN_TEST_UNKNOWN_SAMPLES == 10


def test_exactly_one_calibration_image_per_category_fails_adequacy(mock_reference_store: ReferenceEmbeddingsStore):
    """Exactly one calibration image per location and one unknown image must fail adequacy."""
    single_counts = {loc: 1 for loc in mock_reference_store.locations}
    single_counts["unknown"] = 1

    # Call with default safeguard minimums (5 per location, 10 unknown)
    is_adequate, missing = check_calibration_data_adequacy(
        store=mock_reference_store,
        sample_counts=single_counts,
    )

    assert is_adequate is False
    # All hunt locations and unknown must be flagged as inadequate
    assert len(missing) == len(mock_reference_store.locations) + 1
    for loc in mock_reference_store.locations:
        assert any(f"{loc} (count=1 < 5)" in m for m in missing)
    assert any("unknown (count=1 < 10)" in m for m in missing)


def test_exactly_one_test_image_per_category_not_marked_as_completed(
    tmp_path: Path, monkeypatch, mock_reference_store: ReferenceEmbeddingsStore
):
    """Exactly one test image per location and one unknown image must not be marked as completed."""
    # 1. Pure check
    single_counts = {loc: 1 for loc in mock_reference_store.locations}
    single_counts["unknown"] = 1

    is_complete, missing = check_split_data_adequacy(
        store=mock_reference_store,
        sample_counts=single_counts,
        min_samples_per_location=5,
        min_unknown_samples=10,
    )
    assert is_complete is False
    assert len(missing) == len(mock_reference_store.locations) + 1

    # 2. Integration check with run_calibration
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "out.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    # Adequate calibration (5 per location, 10 unknown)
    populate_split(calib_dir, {"loc_a": 5, "unknown": 10})
    # Single image in test split (1 per location, 1 unknown)
    populate_split(test_dir, {"loc_a": 1, "unknown": 1})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
        max_far=1.0,
        min_unknown_rejection=0.0,
        base_config=base_cfg,
    )
    assert success is True
    loaded = CampusHuntConfig.from_json_file(out_cfg)

    # Must NOT be completed; must be incomplete with metrics=None
    assert loaded.calibration_metadata.test_evaluation.status == "incomplete"
    assert loaded.calibration_metadata.test_evaluation.status != "completed"
    assert loaded.calibration_metadata.test_evaluation.metrics is None
    assert loaded.calibration_metadata.test_evaluation.missing_categories is not None
    assert any("loc_a (count=1 < 5)" in m for m in loaded.calibration_metadata.test_evaluation.missing_categories)
    assert any("unknown (count=1 < 10)" in m for m in loaded.calibration_metadata.test_evaluation.missing_categories)


def test_meeting_configured_minimums_allows_evaluation_to_proceed(tmp_path: Path, monkeypatch):
    """Meeting the configured minimums allows calibration and independent test evaluation to complete."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "out.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    # Meet default minimums: 5 for loc_a, 10 for unknown
    populate_split(calib_dir, {"loc_a": 5, "unknown": 10})
    populate_split(test_dir, {"loc_a": 5, "unknown": 10})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=out_cfg,
        max_far=1.0,
        min_unknown_rejection=0.0,
        base_config=base_cfg,
    )
    assert success is True
    loaded = CampusHuntConfig.from_json_file(out_cfg)

    # Calibration is standard (calibrated, not provisional)
    assert loaded.calibration_metadata.status == "calibrated"
    assert loaded.calibration_metadata.is_provisional is False
    assert loaded.calibration_metadata.warning is None

    # Test evaluation completed cleanly
    assert loaded.calibration_metadata.test_evaluation.status == "completed"
    assert loaded.calibration_metadata.test_evaluation.is_provisional is False
    assert loaded.calibration_metadata.test_evaluation.metrics is not None
    assert loaded.calibration_metadata.test_evaluation.missing_categories is None


def test_manually_lowering_minimum_produces_warning_and_provisional_status(tmp_path: Path, monkeypatch, caplog):
    """Manually lowering sample minimums produces a prominent warning and marks statuses as provisional."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "provisional_out.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    # Populate only 1 image per location and 1 unknown image
    populate_split(calib_dir, {"loc_a": 1, "unknown": 1})
    populate_split(test_dir, {"loc_a": 1, "unknown": 1})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )

    with caplog.at_level("WARNING"):
        success = run_calibration(
            calibration_dir=calib_dir,
            test_dir=test_dir,
            embeddings_npz=npz_path,
            metadata_json=meta_path,
            output_config=out_cfg,
            max_far=1.0,
            min_unknown_rejection=0.0,
            min_calib_samples_per_location=1,  # Manually lowered
            min_calib_unknown_samples=1,       # Manually lowered
            min_test_samples_per_location=1,   # Manually lowered
            min_test_unknown_samples=1,        # Manually lowered
            base_config=base_cfg,
        )

    assert success is True
    # Verify prominent warnings were logged
    assert any("PROVISIONAL CALIBRATION WARNING" in record.message for record in caplog.records)
    assert any("PROVISIONAL TEST SPLIT WARNING" in record.message for record in caplog.records)

    loaded = CampusHuntConfig.from_json_file(out_cfg)

    # Calibration status is marked as provisional rather than verified
    assert loaded.calibration_metadata.status == "provisional"
    assert loaded.calibration_metadata.is_provisional is True
    assert loaded.calibration_metadata.warning is not None
    assert "below project standards" in loaded.calibration_metadata.warning

    # Minimums used are recorded
    assert loaded.calibration_metadata.minimums_used == {
        "min_calibration_samples_per_location": 1,
        "min_calibration_unknown_samples": 1,
        "min_test_samples_per_location": 1,
        "min_test_unknown_samples": 1,
    }

    # Test evaluation status is provisional
    assert loaded.calibration_metadata.test_evaluation.status == "provisional"
    assert loaded.calibration_metadata.test_evaluation.is_provisional is True
    assert loaded.calibration_metadata.test_evaluation.minimums_used == {
        "min_test_samples_per_location": 1,
        "min_test_unknown_samples": 1,
    }
    assert loaded.calibration_metadata.test_evaluation.metrics is not None


def test_provisional_warning_exact_defaults_missing_unknown(tmp_path: Path, monkeypatch, caplog):
    """Exact default minimums (5 and 10) with --allow-missing-unknown marks provisional without claiming lowered safeguards."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "provisional_defaults_out.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    # Populate exact default 5 images per location, NO unknown folder
    populate_split(calib_dir, {"loc_a": 5})
    populate_split(test_dir, {"loc_a": 5})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )

    with caplog.at_level("WARNING"):
        success = run_calibration(
            calibration_dir=calib_dir,
            test_dir=test_dir,
            embeddings_npz=npz_path,
            metadata_json=meta_path,
            output_config=out_cfg,
            max_far=1.0,
            min_unknown_rejection=0.0,
            allow_missing_unknown=True,
            # No manually lowered minimums passed: defaults 5 and 10 remain effective
            base_config=base_cfg,
        )

    assert success is True
    # Verify provisional warning was logged
    assert any("PROVISIONAL CALIBRATION WARNING" in record.message for record in caplog.records)
    # But safeguards lowered warning must NOT be logged
    assert not any("safeguards were manually lowered" in record.message for record in caplog.records)
    assert not any("below project standards" in record.message for record in caplog.records)

    loaded = CampusHuntConfig.from_json_file(out_cfg)

    # Status is provisional
    assert loaded.calibration_metadata.status == "provisional"
    assert loaded.calibration_metadata.is_provisional is True
    assert loaded.calibration_metadata.warning is not None
    # Must NOT say below project standards or safeguards lowered
    assert "below project standards" not in loaded.calibration_metadata.warning
    assert "safeguards were manually lowered" not in loaded.calibration_metadata.warning

    # Must specify unavailable unknown evaluation in provisional reasons
    assert loaded.calibration_metadata.provisional_reasons is not None
    assert any("Unknown/distractor evaluation is unavailable" in r for r in loaded.calibration_metadata.provisional_reasons)
    assert not any("Sample safeguards were manually set below project standards" in r for r in loaded.calibration_metadata.provisional_reasons)

    # Test evaluation is provisional with clear reason
    assert loaded.calibration_metadata.test_evaluation.status == "provisional"
    assert loaded.calibration_metadata.test_evaluation.is_provisional is True
    assert loaded.calibration_metadata.test_evaluation.provisional_reasons is not None
    assert any("Unknown/distractor evaluation is unavailable" in r for r in loaded.calibration_metadata.test_evaluation.provisional_reasons)
    assert not any("Sample safeguards were manually set below project standards" in r for r in loaded.calibration_metadata.test_evaluation.provisional_reasons)

    # Unknown metrics are null
    assert loaded.calibration_metadata.calibration_metrics.unknown_rejection_rate is None
    assert loaded.calibration_metadata.test_evaluation.metrics.unknown_rejection_rate is None

    # Gate still fails safe to ambiguous in production
    can_auto_accept, reason = check_config_validation_status(loaded)
    assert can_auto_accept is False
    assert reason == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"


def test_provisional_warning_genuinely_lowered_counts(tmp_path: Path, monkeypatch, caplog):
    """Genuinely lowered minimums report lowered sample safeguards, cite project defaults, and record provisional reasons."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    out_cfg = tmp_path / "provisional_lowered_out.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    # Populate lowered counts: 2 for loc_a, 3 for unknown
    populate_split(calib_dir, {"loc_a": 2, "unknown": 3})
    populate_split(test_dir, {"loc_a": 2, "unknown": 3})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )

    with caplog.at_level("WARNING"):
        success = run_calibration(
            calibration_dir=calib_dir,
            test_dir=test_dir,
            embeddings_npz=npz_path,
            metadata_json=meta_path,
            output_config=out_cfg,
            max_far=1.0,
            min_unknown_rejection=0.0,
            min_calib_samples_per_location=2,
            min_calib_unknown_samples=3,
            min_test_samples_per_location=2,
            min_test_unknown_samples=3,
            allow_missing_unknown=False,
            base_config=base_cfg,
        )

    assert success is True
    # Verify lowered safeguards warning was logged
    assert any("PROVISIONAL CALIBRATION WARNING" in record.message for record in caplog.records)
    assert any("below project standards" in record.message for record in caplog.records)

    loaded = CampusHuntConfig.from_json_file(out_cfg)

    # Status is provisional
    assert loaded.calibration_metadata.status == "provisional"
    assert loaded.calibration_metadata.is_provisional is True
    assert loaded.calibration_metadata.warning is not None
    assert "below project standards" in loaded.calibration_metadata.warning

    # Provisional reasons record lowered safeguards explicitly
    assert loaded.calibration_metadata.provisional_reasons is not None
    assert any("Sample safeguards were manually set below project standards" in r for r in loaded.calibration_metadata.provisional_reasons)
    assert not any("Unknown/distractor evaluation is unavailable" in r for r in loaded.calibration_metadata.provisional_reasons)

    assert loaded.calibration_metadata.test_evaluation.status == "provisional"
    assert loaded.calibration_metadata.test_evaluation.provisional_reasons is not None
    assert any("Sample safeguards were manually set below project standards" in r for r in loaded.calibration_metadata.test_evaluation.provisional_reasons)
    assert not any("Unknown/distractor evaluation is unavailable" in r for r in loaded.calibration_metadata.test_evaluation.provisional_reasons)

    # Unknown metrics ARE present because unknown was evaluated
    assert loaded.calibration_metadata.calibration_metrics.unknown_rejection_rate is not None
    assert loaded.calibration_metadata.test_evaluation.metrics.unknown_rejection_rate is not None


def test_inadequate_calibration_does_not_create_or_overwrite_active_config(tmp_path: Path, monkeypatch):
    """Inadequate calibration data must not create a config or overwrite an existing active config."""
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    existing_cfg_path = tmp_path / "active_calibrated.json"
    npz_path = tmp_path / "emb.npz"
    meta_path = tmp_path / "meta.json"

    # Write pre-existing valid active config
    sentinel_text = json.dumps({"active_status": "production_ready", "version": 1})
    existing_cfg_path.write_text(sentinel_text, encoding="utf-8")

    dim = 64
    ReferenceEmbeddingsStore.save(
        npz_path=npz_path,
        metadata_path=meta_path,
        embeddings_by_location={"loc_a": np.ones((1, dim), dtype=np.float32)},
        model_name="ViT-B-32",
        pretrained="mock_test",
    )
    # Inadequate calibration data (only 1 image under default 5 and 10 minimums)
    populate_split(calib_dir, {"loc_a": 1, "unknown": 1})

    mock_emb = MockEmbedder(dimension=dim)
    monkeypatch.setattr("campus_hunt_cv.cli.calibrate.OpenCLIPLocationEmbedder", lambda config: mock_emb)

    base_cfg = CampusHuntConfig(
        model=ModelConfig(model_name="ViT-B-32", pretrained="mock_test", device="cpu"),
    )
    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=existing_cfg_path,
        base_config=base_cfg,
    )

    # 1. Calibration routine returns False
    assert success is False
    # 2. Pre-existing file is completely unchanged
    assert existing_cfg_path.read_text(encoding="utf-8") == sentinel_text

    # 3. If file did not exist, no new file is created
    non_existent_path = tmp_path / "should_not_exist.json"
    success2 = run_calibration(
        calibration_dir=calib_dir,
        test_dir=test_dir,
        embeddings_npz=npz_path,
        metadata_json=meta_path,
        output_config=non_existent_path,
        base_config=base_cfg,
    )
    assert success2 is False
    assert not non_existent_path.exists()


# ============================================================================
# RUNTIME HANDLING OF PROVISIONAL & UNVALIDATED CONFIGS
# ============================================================================

def test_provisional_calibration_cannot_auto_accept_by_default(mock_reference_store: ReferenceEmbeddingsStore):
    """Provisional calibration must fail safe to ambiguous with PROVISIONAL_CALIBRATION_REQUIRES_REVIEW."""
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    cfg = CampusHuntConfig()
    cfg.scoring = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.08,
        allow_unvalidated_auto_accept=False,
    )
    cfg.calibration_metadata = CalibrationMetadata(
        status="provisional",
        is_provisional=True,
        warning="Lowered safeguards",
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-10-30T12:00:00Z",
        calibration_sample_counts={"clock_tower": 2, "unknown": 2},
        calibration_metrics=CalibrationMetrics(
            correct_accept_rate=0.95,
            wrong_false_accept_rate=0.01,
            unknown_rejection_rate=0.98,
        ),
        test_evaluation=TestEvaluationMetrics(
            status="completed",
            is_provisional=False,
            metrics=CalibrationMetrics(
                correct_accept_rate=0.94,
                wrong_false_accept_rate=0.01,
                unknown_rejection_rate=0.97,
            ),
        ),
    )

    verifier = LocationVerifier(
        config=cfg,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    # Score fields are computed and preserved
    assert result["target_score"] is not None
    assert result["target_score"] >= 0.60
    assert result["margin"] is not None
    assert result["margin"] >= 0.08
    assert result["best_matching_location_id"] == "clock_tower"

    # Verdict must be ambiguous with clear review code (never strong_match)
    assert result["result"] == "ambiguous"
    assert result["reason_code"] == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"


def test_unavailable_incomplete_or_provisional_holdout_cannot_auto_accept(mock_reference_store: ReferenceEmbeddingsStore):
    """Unavailable, incomplete, provisional hold-out, or missing metadata must fail safe to HOLDOUT_TEST_NOT_COMPLETED."""
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    image_bytes = create_test_image_bytes()

    # Sub-case A: test_evaluation.status == 'unavailable'
    cfg_unavail = CampusHuntConfig()
    cfg_unavail.scoring = ScoringConfig(calibrated=True, match_threshold=0.60, margin_threshold=0.08)
    cfg_unavail.calibration_metadata = CalibrationMetadata(
        status="calibrated",
        is_provisional=False,
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-10-30T12:00:00Z",
        calibration_sample_counts={"clock_tower": 5, "unknown": 10},
        calibration_metrics=CalibrationMetrics(
            correct_accept_rate=0.95, wrong_false_accept_rate=0.01, unknown_rejection_rate=0.98
        ),
        test_evaluation=TestEvaluationMetrics(status="unavailable", reason="No test photos", metrics=None),
    )
    v_unavail = LocationVerifier(config=cfg_unavail, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store)
    res_a = v_unavail.verify(image_bytes, "clock_tower")
    assert res_a["result"] == "ambiguous"
    assert res_a["reason_code"] == "HOLDOUT_TEST_NOT_COMPLETED"
    assert res_a["target_score"] >= 0.60

    # Sub-case B: test_evaluation.status == 'incomplete'
    cfg_incomp = CampusHuntConfig()
    cfg_incomp.scoring = ScoringConfig(calibrated=True, match_threshold=0.60, margin_threshold=0.08)
    cfg_incomp.calibration_metadata = CalibrationMetadata(
        status="calibrated",
        is_provisional=False,
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-10-30T12:00:00Z",
        calibration_sample_counts={"clock_tower": 5, "unknown": 10},
        calibration_metrics=CalibrationMetrics(
            correct_accept_rate=0.95, wrong_false_accept_rate=0.01, unknown_rejection_rate=0.98
        ),
        test_evaluation=TestEvaluationMetrics(
            status="incomplete",
            reason="Missing categories",
            missing_categories=["unknown (count=0 < 10)"],
            metrics=None,
        ),
    )
    v_incomp = LocationVerifier(config=cfg_incomp, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store)
    res_b = v_incomp.verify(image_bytes, "clock_tower")
    assert res_b["result"] == "ambiguous"
    assert res_b["reason_code"] == "HOLDOUT_TEST_NOT_COMPLETED"
    assert res_b["target_score"] >= 0.60

    # Sub-case C: test_evaluation.status == 'provisional'
    cfg_prov_test = CampusHuntConfig()
    cfg_prov_test.scoring = ScoringConfig(calibrated=True, match_threshold=0.60, margin_threshold=0.08)
    cfg_prov_test.calibration_metadata = CalibrationMetadata(
        status="calibrated",
        is_provisional=False,
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-10-30T12:00:00Z",
        calibration_sample_counts={"clock_tower": 5, "unknown": 10},
        calibration_metrics=CalibrationMetrics(
            correct_accept_rate=0.95, wrong_false_accept_rate=0.01, unknown_rejection_rate=0.98
        ),
        test_evaluation=TestEvaluationMetrics(
            status="provisional",
            is_provisional=True,
            metrics=CalibrationMetrics(correct_accept_rate=0.94, wrong_false_accept_rate=0.01, unknown_rejection_rate=0.97),
        ),
    )
    v_prov_test = LocationVerifier(config=cfg_prov_test, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store)
    res_c = v_prov_test.verify(image_bytes, "clock_tower")
    assert res_c["result"] == "ambiguous"
    assert res_c["reason_code"] == "HOLDOUT_TEST_NOT_COMPLETED"
    assert res_c["target_score"] >= 0.60

    # Sub-case D: calibration_metadata is None (legacy or unvalidated)
    cfg_no_meta = CampusHuntConfig()
    cfg_no_meta.scoring = ScoringConfig(calibrated=True, match_threshold=0.60, margin_threshold=0.08)
    cfg_no_meta.calibration_metadata = None
    v_no_meta = LocationVerifier(config=cfg_no_meta, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store)
    res_d = v_no_meta.verify(image_bytes, "clock_tower")
    assert res_d["result"] == "ambiguous"
    assert res_d["reason_code"] == "HOLDOUT_TEST_NOT_COMPLETED"
    assert res_d["target_score"] >= 0.60


def test_fully_calibrated_completed_holdout_produces_strong_match(mock_reference_store: ReferenceEmbeddingsStore):
    """Fully calibrated config with completed hold-out test evaluation produces strong_match."""
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    cfg = CampusHuntConfig()
    cfg.scoring = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.08,
        allow_unvalidated_auto_accept=False,
    )
    cfg.calibration_metadata = CalibrationMetadata(
        status="calibrated",
        is_provisional=False,
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-10-30T12:00:00Z",
        calibration_sample_counts={"clock_tower": 5, "library_atrium": 5, "unknown": 10},
        calibration_metrics=CalibrationMetrics(
            correct_accept_rate=0.95,
            wrong_false_accept_rate=0.01,
            unknown_rejection_rate=0.98,
        ),
        test_evaluation=TestEvaluationMetrics(
            status="completed",
            is_provisional=False,
            metrics=CalibrationMetrics(
                correct_accept_rate=0.94,
                wrong_false_accept_rate=0.01,
                unknown_rejection_rate=0.97,
            ),
        ),
    )

    verifier = LocationVerifier(
        config=cfg,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    assert result["result"] == "strong_match"
    assert result["reason_code"] == "STRONG_MATCH"
    assert result["best_matching_location_id"] == "clock_tower"
    assert result["target_score"] >= 0.60
    assert result["margin"] >= 0.08


def test_unvalidated_auto_accept_blocked_when_env_unset(
    monkeypatch,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """Safeguard: unvalidated auto-accept is strictly blocked when CAMPUS_HUNT_ENVIRONMENT is unset."""
    monkeypatch.delenv("CAMPUS_HUNT_ENVIRONMENT", raising=False)
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    cfg = CampusHuntConfig()
    cfg.scoring = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.08,
        allow_unvalidated_auto_accept=True,  # Opt-in attempted
    )
    cfg.calibration_metadata = CalibrationMetadata(
        status="provisional",
        is_provisional=True,
        warning="Lowered safeguards",
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-10-30T12:00:00Z",
        calibration_sample_counts={"clock_tower": 1, "unknown": 1},
        calibration_metrics=CalibrationMetrics(
            correct_accept_rate=0.95, wrong_false_accept_rate=0.01, unknown_rejection_rate=0.98
        ),
        test_evaluation=TestEvaluationMetrics(status="unavailable", metrics=None),
    )

    # When environment is unset, runtime defaults strictly to production
    assert cfg.effective_environment == "production"
    assert cfg.is_production is True

    verifier = LocationVerifier(
        config=cfg,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    # Override MUST be blocked: fails closed to ambiguous
    assert result["result"] == "ambiguous"
    assert result["reason_code"] == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"
    assert result["target_score"] >= 0.60
    assert result["margin"] >= 0.08


def test_unvalidated_auto_accept_blocked_in_production_with_json_and_env_var(
    monkeypatch,
    tmp_path: Path,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """Safeguard: unvalidated auto-accept is blocked in production even when enabled in JSON and env var."""
    monkeypatch.setenv("CAMPUS_HUNT_ENVIRONMENT", "production")
    monkeypatch.setenv("CAMPUS_HUNT_ALLOW_UNVALIDATED_AUTO_ACCEPT", "1")

    json_file = tmp_path / "unsafe_prod_config.json"
    raw_cfg = {
        "environment": "production",
        "model": {
            "model_name": "ViT-B-32",
            "pretrained": "mock_test_weights",
            "device": "cpu",
            "num_threads": None,
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
            "match_threshold": 0.60,
            "margin_threshold": 0.08,
            "mismatch_threshold": None,
            "min_reference_images_per_location": 1,
            "min_calibration_samples_per_location": 5,
            "min_calibration_unknown_samples": 10,
            "min_test_samples_per_location": 5,
            "min_test_unknown_samples": 10,
            "target_max_far": 0.02,
            "target_min_unknown_rejection": 0.90,
            "allow_unvalidated_auto_accept": True,  # Enabled in JSON
        },
        "storage": {
            "reference_data_dir": "data/reference",
            "calibration_data_dir": "data/calibration",
            "test_data_dir": "data/test",
            "embeddings_path": "data/reference_embeddings.npz",
            "metadata_path": "data/reference_embeddings.meta.json",
        },
        "calibration_metadata": {
            "status": "provisional",
            "is_provisional": True,
            "warning": "Lowered safeguards",
            "model_version": "ViT-B-32/mock_test_weights",
            "calibrated_at": "2026-10-30T12:00:00Z",
            "minimums_used": None,
            "calibration_sample_counts": {"clock_tower": 2, "unknown": 2},
            "calibration_metrics": {
                "correct_accept_rate": 0.95,
                "wrong_false_accept_rate": 0.01,
                "unknown_rejection_rate": 0.98,
                "correct_trials": 10,
                "correct_accepts": 9,
                "wrong_trials": 10,
                "wrong_false_accepts": 0,
                "unknown_trials": 10,
                "unknown_rejections": 10,
            },
            "test_evaluation": {
                "status": "incomplete",
                "reason": "Missing categories",
                "sample_counts": {"clock_tower": 1},
                "missing_categories": ["unknown"],
                "minimums_used": None,
                "is_provisional": False,
                "metrics": None,
            },
        },
    }
    json_file.write_text(json.dumps(raw_cfg, indent=2), encoding="utf-8")

    loaded_cfg = CampusHuntConfig.from_json_file(json_file)
    assert loaded_cfg.effective_environment == "production"
    assert loaded_cfg.is_production is True
    assert loaded_cfg.scoring.allow_unvalidated_auto_accept is True

    # validate_for_production() raises ConfigurationError
    with pytest.raises(ConfigurationError) as exc_info:
        loaded_cfg.validate_for_production()
    assert "allow_unvalidated_auto_accept" in str(exc_info.value)

    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    verifier = LocationVerifier(
        config=loaded_cfg,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    # In production, verifier fails closed to ambiguous
    assert result["result"] == "ambiguous"
    assert result["reason_code"] == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"


def test_unvalidated_auto_accept_works_only_in_dev_or_test(
    monkeypatch,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """Safeguard: unvalidated auto-accept works ONLY when explicitly in development or test."""
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    def build_test_cfg(allow_opt_in: bool) -> CampusHuntConfig:
        cfg = CampusHuntConfig()
        cfg.scoring = ScoringConfig(
            calibrated=True,
            match_threshold=0.60,
            margin_threshold=0.08,
            allow_unvalidated_auto_accept=allow_opt_in,
        )
        cfg.calibration_metadata = CalibrationMetadata(
            status="provisional",
            is_provisional=True,
            warning="Lowered safeguards",
            model_version="ViT-B-32/mock_test_weights",
            calibrated_at="2026-10-30T12:00:00Z",
            calibration_sample_counts={"clock_tower": 1, "unknown": 1},
            calibration_metrics=CalibrationMetrics(
                correct_accept_rate=0.95, wrong_false_accept_rate=0.01, unknown_rejection_rate=0.98
            ),
            test_evaluation=TestEvaluationMetrics(status="unavailable", metrics=None),
        )
        return cfg

    image_bytes = create_test_image_bytes()

    # Case A: Development environment + allow_unvalidated_auto_accept=True -> PERMITTED
    monkeypatch.setenv("CAMPUS_HUNT_ENVIRONMENT", "development")
    cfg_dev = build_test_cfg(allow_opt_in=True)
    verifier_dev = LocationVerifier(config=cfg_dev, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store)
    res_dev = verifier_dev.verify(image_bytes=image_bytes, target_location_id="clock_tower")
    assert res_dev["result"] == "strong_match"
    assert res_dev["reason_code"] == "STRONG_MATCH"

    # Case B: Test environment + allow_unvalidated_auto_accept=True -> PERMITTED
    monkeypatch.setenv("CAMPUS_HUNT_ENVIRONMENT", "test")
    cfg_test = build_test_cfg(allow_opt_in=True)
    verifier_test = LocationVerifier(config=cfg_test, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store)
    res_test = verifier_test.verify(image_bytes=image_bytes, target_location_id="clock_tower")
    assert res_test["result"] == "strong_match"
    assert res_test["reason_code"] == "STRONG_MATCH"

    # Case C: Development environment BUT allow_unvalidated_auto_accept=False -> BLOCKED
    monkeypatch.setenv("CAMPUS_HUNT_ENVIRONMENT", "development")
    cfg_dev_no_override = build_test_cfg(allow_opt_in=False)
    verifier_dev_no_override = LocationVerifier(
        config=cfg_dev_no_override, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store
    )
    res_dev_no_override = verifier_dev_no_override.verify(image_bytes=image_bytes, target_location_id="clock_tower")
    assert res_dev_no_override["result"] == "ambiguous"
    assert res_dev_no_override["reason_code"] == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"

    # Case D: Invalid environment string (e.g. 'staging') -> Defaults to production -> BLOCKED
    monkeypatch.setenv("CAMPUS_HUNT_ENVIRONMENT", "staging_cluster")
    cfg_invalid_env = build_test_cfg(allow_opt_in=True)
    assert cfg_invalid_env.effective_environment == "production"
    verifier_invalid = LocationVerifier(
        config=cfg_invalid_env, embedder=ControlledEmbedder(dimension=64), store=mock_reference_store
    )
    res_invalid = verifier_invalid.verify(image_bytes=image_bytes, target_location_id="clock_tower")
    assert res_invalid["result"] == "ambiguous"
    assert res_invalid["reason_code"] == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"


def test_fully_validated_config_works_normally_in_production(
    monkeypatch,
    mock_reference_store: ReferenceEmbeddingsStore,
):
    """Safeguard: fully validated configurations produce strong_match normally in production."""
    monkeypatch.setenv("CAMPUS_HUNT_ENVIRONMENT", "production")
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class ControlledEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    cfg = CampusHuntConfig()
    cfg.scoring = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.08,
        allow_unvalidated_auto_accept=False,
    )
    cfg.calibration_metadata = CalibrationMetadata(
        status="calibrated",
        is_provisional=False,
        warning=None,
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-10-30T12:00:00Z",
        minimums_used=None,
        calibration_sample_counts={"clock_tower": 5, "library": 5, "unknown": 10},
        calibration_metrics=CalibrationMetrics(
            correct_accept_rate=0.96, wrong_false_accept_rate=0.005, unknown_rejection_rate=0.99
        ),
        test_evaluation=TestEvaluationMetrics(
            status="completed",
            reason=None,
            sample_counts={"clock_tower": 5, "library": 5, "unknown": 10},
            missing_categories=None,
            minimums_used=None,
            is_provisional=False,
            metrics=CalibrationMetrics(
                correct_accept_rate=0.95, wrong_false_accept_rate=0.008, unknown_rejection_rate=0.98
            ),
        ),
    )

    # In production, validate_for_production() succeeds
    cfg.validate_for_production()

    verifier = LocationVerifier(
        config=cfg,
        embedder=ControlledEmbedder(dimension=64),
        store=mock_reference_store,
    )

    image_bytes = create_test_image_bytes()
    result = verifier.verify(image_bytes=image_bytes, target_location_id="clock_tower")

    # In production, fully validated config successfully auto-accepts strong matches
    assert result["result"] == "strong_match"
    assert result["reason_code"] == "STRONG_MATCH"
    assert result["best_matching_location_id"] == "clock_tower"
    assert result["target_score"] >= 0.60
    assert result["margin"] >= 0.08


def test_normalize_environment_safeguards():
    """Unit test for environment string normalization and safe fallback."""
    assert normalize_environment(None) == "production"
    assert normalize_environment("") == "production"
    assert normalize_environment("   ") == "production"
    assert normalize_environment("production") == "production"
    assert normalize_environment("PRODUCTION") == "production"
    assert normalize_environment("prod") == "production"
    assert normalize_environment("PROD") == "production"
    assert normalize_environment("development") == "development"
    assert normalize_environment("DEVELOPMENT") == "development"
    assert normalize_environment("dev") == "development"
    assert normalize_environment("DEV") == "development"
    assert normalize_environment("test") == "test"
    assert normalize_environment("TEST") == "test"
    assert normalize_environment("testing") == "test"
    assert normalize_environment("staging") == "production"
    assert normalize_environment("unknown_env") == "production"
    assert normalize_environment(123) == "production"


