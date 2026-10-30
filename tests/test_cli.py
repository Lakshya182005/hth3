"""Unit tests for CLI utilities and calibration metric evaluation."""

from pathlib import Path
import pytest

from campus_hunt_cv.cli.build_embeddings import build_reference_embeddings
from campus_hunt_cv.cli.calibrate import evaluate_thresholds
from campus_hunt_cv.config import ScoringConfig
from campus_hunt_cv.similarity import SimilarityScorer


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

