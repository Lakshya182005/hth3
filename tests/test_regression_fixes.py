"""Regression tests verifying Fixes 1 through 6, Fix A, and Fix B."""

import json
import os
from pathlib import Path
import sys
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
    CampusHuntConfig,
    DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
    ModelConfig,
    ScoringConfig,
)
from campus_hunt_cv.decision import DecisionEngine
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.exceptions import ReferenceDataError
from campus_hunt_cv.similarity import SimilarityScorer
from campus_hunt_cv.verifier import LocationVerifier
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
                "status": "unavailable",
                "reason": "No test captures found in data/test/",
                "sample_counts": {},
                "missing_categories": ["clock_tower (count=0 < 1)", "unknown (count=0 < 1)"],
                "metrics": None,
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

