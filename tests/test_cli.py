"""Unit tests for CLI utilities and calibration metric evaluation."""

import json
from pathlib import Path
import numpy as np
import pytest

from campus_hunt_cv.cli.build_embeddings import build_reference_embeddings
from campus_hunt_cv.cli.calibrate import (
    evaluate_thresholds,
    format_cross_location_errors,
    format_focused_confusion_report,
    print_confusion_matrix,
)
from campus_hunt_cv.cli.verify import main as verify_main, verify_image_file
from campus_hunt_cv.config import CampusHuntConfig, ScoringConfig
from campus_hunt_cv.similarity import SimilarityScorer
from campus_hunt_cv.verifier import LocationVerifier
from tests.conftest import MockEmbedder, create_test_image_bytes


def test_build_embeddings_empty_dir_graceful(tmp_path: Path):
    """Empty reference directory should return False and not create corrupt files."""
    empty_ref = tmp_path / "empty_reference"
    empty_ref.mkdir()

    out_npz = tmp_path / "out.npz"
    out_meta = tmp_path / "out.meta.json"

    success = build_reference_embeddings(
        reference_dir=empty_ref,
        output_npz=out_npz,
        output_meta=out_meta,
    )

    assert success is False
    assert not out_npz.exists()
    assert not out_meta.exists()


def test_evaluate_thresholds_metric_calculation(mock_reference_store):
    """Verify that accuracy metrics and confusion matrix are calculated correctly."""
    scorer = SimilarityScorer()
    scoring_config = ScoringConfig(min_reference_images_per_location=1)

    samples = [
        # True positive clock_tower capture
        {
            "actual": "clock_tower",
            "file": "c1.jpg",
            "scores": {"clock_tower": 0.85, "library_atrium": 0.40, "science_quad": 0.20},
        },
        # Borderline clock_tower capture
        {
            "actual": "clock_tower",
            "file": "c2.jpg",
            "scores": {"clock_tower": 0.55, "library_atrium": 0.45, "science_quad": 0.20},
        },
        # True positive library capture
        {
            "actual": "library_atrium",
            "file": "l1.jpg",
            "scores": {"clock_tower": 0.30, "library_atrium": 0.88, "science_quad": 0.25},
        },
        # Unknown/distractor capture (cafeteria tray)
        {
            "actual": "unknown",
            "file": "tray.jpg",
            "scores": {"clock_tower": 0.35, "library_atrium": 0.38, "science_quad": 0.20},
        },
    ]

    metrics = evaluate_thresholds(
        samples=samples,
        store=mock_reference_store,
        scorer=scorer,
        base_scoring_config=scoring_config,
        match_threshold=0.60,
        margin_threshold=0.08,
    )

    # 3 positive samples evaluated against their actual location:
    # - c1: clock_tower target=0.85, best competitor=0.40, margin=0.45 >= 0.08 -> strong_match (Accepted)
    # - c2: clock_tower target=0.55 (<0.60) -> ambiguous (Not accepted)
    # - l1: library_atrium target=0.88, best competitor=0.30, margin=0.58 >= 0.08 -> strong_match (Accepted)
    # Correct accept rate: 2 / 3 = 66.7%
    assert metrics["correct_trials"] == 3
    assert metrics["correct_accepts"] == 2
    assert metrics["correct_accept_rate"] == pytest.approx(2 / 3, abs=1e-3)

    # Cross-location trials:
    # - c1 tested against 2 wrong locations: scores < 0.60 -> not accepted
    # - c2 tested against 2 wrong locations: scores < 0.60 -> not accepted
    # - l1 tested against 2 wrong locations: scores < 0.60 -> not accepted
    # Total wrong trials: 3 * 2 = 6, false accepts: 0 -> FAR: 0.0%
    assert metrics["wrong_trials"] == 6
    assert metrics["wrong_false_accepts"] == 0
    assert metrics["wrong_false_accept_rate"] == 0.0

    # Unknown trials: tray tested against all 3 locations
    # All scores < 0.60 -> none accepted as strong_match
    # Unknown rejection rate: 3 / 3 = 100.0%
    assert metrics["unknown_trials"] == 3
    assert metrics["unknown_rejections"] == 3
    assert metrics["unknown_rejection_rate"] == 1.0

    # Confusion matrix structure
    conf = metrics["confusion_counts"]
    assert conf["clock_tower"]["clock_tower"] == 1
    assert conf["clock_tower"]["unaccepted_ambiguous"] == 1
    assert conf["library_atrium"]["library_atrium"] == 1


def test_evaluate_thresholds_no_unknown_samples_sets_null_metrics(mock_reference_store):
    """When no unknown samples are provided, unknown metrics must be None (null in JSON)."""
    scorer = SimilarityScorer()
    scoring_config = ScoringConfig(min_reference_images_per_location=1)

    samples = [
        {
            "actual": "clock_tower",
            "file": "c1.jpg",
            "scores": {"clock_tower": 0.85, "library_atrium": 0.40, "science_quad": 0.20},
        },
        {
            "actual": "library_atrium",
            "file": "l1.jpg",
            "scores": {"clock_tower": 0.30, "library_atrium": 0.88, "science_quad": 0.25},
        },
    ]

    metrics = evaluate_thresholds(
        samples=samples,
        store=mock_reference_store,
        scorer=scorer,
        base_scoring_config=scoring_config,
        match_threshold=0.60,
        margin_threshold=0.08,
    )

    assert metrics["correct_trials"] == 2
    assert metrics["correct_accepts"] == 2
    assert metrics["correct_accept_rate"] == 1.0

    # Unknown metrics MUST be None
    assert metrics["unknown_trials"] is None
    assert metrics["unknown_rejections"] is None
    assert metrics["unknown_rejection_rate"] is None


def test_evaluate_thresholds_and_confusion_matrix_identifies_misclassifications(mock_reference_store, capsys):
    """When a capture is misclassified as a competing location, it is identified in confusion matrix output."""
    scorer = SimilarityScorer()
    scoring_config = ScoringConfig(min_reference_images_per_location=1)

    # c1 is actually clock_tower, but has score 0.85 for library_atrium and 0.40 for clock_tower
    samples = [
        {
            "actual": "clock_tower",
            "file": "c1.jpg",
            "scores": {"clock_tower": 0.40, "library_atrium": 0.85, "science_quad": 0.20},
        },
    ]

    metrics = evaluate_thresholds(
        samples=samples,
        store=mock_reference_store,
        scorer=scorer,
        base_scoring_config=scoring_config,
        match_threshold=0.60,
        margin_threshold=0.08,
    )

    # Cross-location false acceptance occurred
    assert metrics["wrong_false_accepts"] == 1
    assert metrics["confusion_counts"]["clock_tower"]["library_atrium"] == 1
    # Must NOT double-count the misclassified capture as unaccepted
    assert metrics["confusion_counts"]["clock_tower"].get("unaccepted_mismatch", 0) == 0
    assert sum(metrics["confusion_counts"]["clock_tower"].values()) == 1
    assert len(metrics["cross_location_false_acceptances"]) == 1
    err = metrics["cross_location_false_acceptances"][0]
    assert err["actual_location"] == "clock_tower"
    assert err["claimed_location"] == "library_atrium"
    assert err["best_matching_location"] == "library_atrium"
    assert err["target_score"] == 0.85
    assert err["competing_score"] == 0.40
    assert round(err["margin"], 2) == 0.45
    assert err["decision"] == "strong_match"

    print_confusion_matrix(
        confusion_counts=metrics["confusion_counts"],
        locations=["clock_tower", "library_atrium", "science_quad"],
    )

    captured = capsys.readouterr().out
    assert "MISCLASSIFICATIONS DETECTED (Cross-Location False Acceptances):" in captured
    assert "Actual 'clock_tower' misclassified as competing location 'library_atrium': 1 capture(s)" in captured


def test_format_cross_location_errors_and_focused_report():
    """Verify that cross-location error details and focused report contain all required fields."""
    mock_errors = [
        {
            "image_path": "data/test/4th_floor_near_robotics_lab/IMG20261103184842.jpg",
            "actual_location": "4th_floor_near_robotics_lab",
            "claimed_location": "a_block_near_313",
            "best_matching_location": "a_block_near_313",
            "target_score": 0.6190,
            "competing_score": 0.6046,
            "margin": 0.0144,
            "decision": "strong_match",
        }
    ]

    err_report = format_cross_location_errors(mock_errors, split_name="TEST")
    assert "Relative Image Path:    data/test/4th_floor_near_robotics_lab/IMG20261103184842.jpg" in err_report
    assert "Actual Location:        4th_floor_near_robotics_lab" in err_report
    assert "Claimed Location:       a_block_near_313" in err_report
    assert "Best-Matching Location: a_block_near_313" in err_report
    assert "Target Score:           0.6190" in err_report
    assert "Competing Score:        0.6046" in err_report
    assert "Margin:                 0.0144" in err_report
    assert "Decision Result:        strong_match" in err_report

    focused_report = format_focused_confusion_report(
        mock_errors, "4th_floor_near_robotics_lab", "a_block_near_313"
    )
    assert "FOCUSED CONFUSION REPORT: '4th_floor_near_robotics_lab' <--> 'a_block_near_313'" in focused_report
    assert "Target Location Pair:   '4th_floor_near_robotics_lab' <--> 'a_block_near_313'" in focused_report
    assert "Identified Incidents:   1" in focused_report
    assert "Relative Image Path:    data/test/4th_floor_near_robotics_lab/IMG20261103184842.jpg" in focused_report
    assert "Actual Location:        4th_floor_near_robotics_lab" in focused_report
    assert "Claimed Location:       a_block_near_313" in focused_report
    assert "Root-Cause & Operational Assessment" in focused_report

    # When no incidents match the pair
    empty_focused = format_focused_confusion_report([], "loc_x", "loc_y")
    assert "No cross-location false acceptances detected" in empty_focused


def test_cli_verify_positional_arguments_success(
    tmp_path: Path,
    calibrated_config: CampusHuntConfig,
    mock_reference_store,
    capsys,
):
    """Test campus-hunt-verify with positional arguments prints valid JSON and returns 0."""
    img_bytes = create_test_image_bytes(width=64, height=64)
    img_path = tmp_path / "query.jpg"
    img_path.write_bytes(img_bytes)

    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=calibrated_config,
        embedder=embedder,
        store=mock_reference_store,
    )

    exit_code = verify_main([str(img_path), "clock_tower"], verifier=verifier)
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert "result" in payload
    assert payload["result"] in ("strong_match", "ambiguous", "mismatch")
    assert "target_score" in payload
    assert "best_matching_location_id" in payload
    assert "best_matching_score" in payload
    assert "best_competing_location_id" in payload
    assert "best_competing_score" in payload
    assert "margin" in payload
    assert "reason_code" in payload
    assert "model_version" in payload
    assert payload["model_version"] == embedder.model_version


def test_cli_verify_flag_arguments_success(
    tmp_path: Path,
    calibrated_config: CampusHuntConfig,
    mock_reference_store,
    capsys,
):
    """Test campus-hunt-verify with --image and --target flags."""
    img_bytes = create_test_image_bytes(width=64, height=64)
    img_path = tmp_path / "query.jpg"
    img_path.write_bytes(img_bytes)

    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=calibrated_config,
        embedder=embedder,
        store=mock_reference_store,
    )

    exit_code = verify_main(["--image", str(img_path), "--target", "library_atrium"], verifier=verifier)
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["result"] in ("strong_match", "ambiguous", "mismatch")
    assert payload["target_score"] is not None


def test_cli_verify_missing_arguments_fails():
    """Missing positional or flag arguments must raise SystemExit with exit code 2."""
    with pytest.raises(SystemExit) as exc_info:
        verify_main([])
    assert exc_info.value.code == 2


def test_cli_verify_file_not_found_returns_error(capsys):
    """Non-existent image file must return exit code 1 and structured error JSON."""
    exit_code = verify_main(["nonexistent_photo_123.jpg", "clock_tower"])
    assert exit_code == 1

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["result"] == "ambiguous"
    assert payload["reason_code"] == "ERROR_FILE_NOT_FOUND"


def test_cli_verify_preserves_provisional_safety_gate(
    tmp_path: Path,
    mock_reference_store,
    capsys,
):
    """CLI must enforce production safety gate and downgrade provisional strong_match to ambiguous."""
    from campus_hunt_cv.config import CalibrationMetadata, CalibrationMetrics, TestEvaluationMetrics

    # Build provisional production configuration
    cfg = CampusHuntConfig()
    cfg.environment = "production"
    cfg.scoring.calibrated = True
    cfg.scoring.match_threshold = 0.50
    cfg.scoring.margin_threshold = 0.05
    cfg.calibration_metadata = CalibrationMetadata(
        status="provisional",
        is_provisional=True,
        warning="Provisional run without unknown/ distractors",
        provisional_reasons=["Unknown evaluation unavailable"],
        model_version="ViT-B-32/mock_test_weights",
        calibrated_at="2026-11-10T12:00:00Z",
        calibration_sample_counts={"clock_tower": 5, "library_atrium": 5},
        calibration_metrics=CalibrationMetrics(correct_accept_rate=1.0, wrong_false_accept_rate=0.0),
        test_evaluation=TestEvaluationMetrics(
            status="provisional",
            is_provisional=True,
            reason="Provisional test split",
            metrics=CalibrationMetrics(correct_accept_rate=1.0, wrong_false_accept_rate=0.0),
        ),
    )

    # Force query embedding to be identical to clock_tower reference center
    clock_ref = mock_reference_store.get_location_embeddings("clock_tower")[0]

    class PerfectMatchEmbedder(MockEmbedder):
        def embed_image(self, image):
            return clock_ref.copy()

    verifier = LocationVerifier(
        config=cfg,
        embedder=PerfectMatchEmbedder(dimension=64),
        store=mock_reference_store,
    )

    img_bytes = create_test_image_bytes()
    img_path = tmp_path / "match.jpg"
    img_path.write_bytes(img_bytes)

    exit_code = verify_main([str(img_path), "clock_tower"], verifier=verifier)
    assert exit_code == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    # Score and margin exceed thresholds, but safety gate downgrades to ambiguous
    assert payload["target_score"] >= 0.90
    assert payload["best_matching_location_id"] == "clock_tower"
    assert payload["result"] == "ambiguous"
    assert payload["reason_code"] == "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"


def test_verify_image_file_direct_helper(
    tmp_path: Path,
    calibrated_config: CampusHuntConfig,
    mock_reference_store,
):
    """Direct helper verify_image_file must load file bytes and return response dict."""
    img_bytes = create_test_image_bytes()
    img_path = tmp_path / "direct.jpg"
    img_path.write_bytes(img_bytes)

    embedder = MockEmbedder(dimension=64)
    verifier = LocationVerifier(
        config=calibrated_config,
        embedder=embedder,
        store=mock_reference_store,
    )

    result = verify_image_file(img_path, "clock_tower", verifier=verifier)
    assert isinstance(result, dict)
    assert "result" in result
    assert result["result"] in ("strong_match", "ambiguous", "mismatch")
    assert isinstance(result["reason_code"], str)
    assert result["model_version"] == embedder.model_version





