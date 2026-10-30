"""Unit tests for the verification decision engine."""

from campus_hunt_cv.config import ScoringConfig
from campus_hunt_cv.decision import DecisionEngine


def test_uncalibrated_always_returns_ambiguous():
    """Uncalibrated configuration must NEVER return strong_match or mismatch."""
    cfg = ScoringConfig(calibrated=False, match_threshold=0.60, margin_threshold=0.10)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.95,
        best_matching_location_id="clock_tower",
        best_matching_score=0.95,
        best_competing_location_id="library_atrium",
        best_competing_score=0.30,
        margin=0.65,
        target_ref_count=5,
    )
    assert result == "ambiguous"
    assert reason == "UNCALIBRATED_THRESHOLDS"


def test_none_thresholds_returns_ambiguous():
    """Even if calibrated=True, if thresholds are None, must safely return ambiguous."""
    cfg = ScoringConfig(calibrated=True, match_threshold=None, margin_threshold=None)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.95,
        best_matching_location_id="clock_tower",
        best_matching_score=0.95,
        best_competing_location_id="library_atrium",
        best_competing_score=0.30,
        margin=0.65,
        target_ref_count=5,
    )
    assert result == "ambiguous"
    assert reason == "UNCALIBRATED_THRESHOLDS"


def test_unknown_target_location():
    """When target_score is None (location unknown in dataset), return ambiguous."""
    cfg = ScoringConfig(calibrated=True, match_threshold=0.60, margin_threshold=0.08)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="unknown_location",
        target_score=None,
        best_matching_location_id="clock_tower",
        best_matching_score=0.85,
        best_competing_location_id="library_atrium",
        best_competing_score=0.70,
        margin=None,
        target_ref_count=0,
    )
    assert result == "ambiguous"
    assert reason == "UNKNOWN_TARGET_LOCATION"


def test_insufficient_reference_data():
    """When target location has fewer than required reference photos, return ambiguous."""
    cfg = ScoringConfig(
        calibrated=True,
        match_threshold=0.60,
        margin_threshold=0.08,
        min_reference_images_per_location=3,
    )
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.85,
        best_matching_location_id="clock_tower",
        best_matching_score=0.85,
        best_competing_location_id="library_atrium",
        best_competing_score=0.50,
        margin=0.35,
        target_ref_count=2,  # 2 < 3
    )
    assert result == "ambiguous"
    assert reason == "INSUFFICIENT_REFERENCE_DATA"


def test_strong_match_success():
    """Target is best match, meets match_threshold, and exceeds margin_threshold."""
    cfg = ScoringConfig(calibrated=True, match_threshold=0.65, margin_threshold=0.10)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.82,
        best_matching_location_id="clock_tower",
        best_matching_score=0.82,
        best_competing_location_id="library_atrium",
        best_competing_score=0.60,
        margin=0.22,  # 0.22 >= 0.10
        target_ref_count=5,
    )
    assert result == "strong_match"
    assert reason == "STRONG_MATCH"


def test_strong_match_single_known_location():
    """When only 1 location exists in total, competitor and margin are None, but score passes."""
    cfg = ScoringConfig(calibrated=True, match_threshold=0.65, margin_threshold=0.10)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.80,
        best_matching_location_id="clock_tower",
        best_matching_score=0.80,
        best_competing_location_id=None,
        best_competing_score=None,
        margin=None,
        target_ref_count=3,
    )
    assert result == "strong_match"
    assert reason == "STRONG_MATCH"


def test_ambiguous_due_to_low_score():
    """Target is top-1 and margin is good, but score is below match_threshold."""
    cfg = ScoringConfig(calibrated=True, match_threshold=0.65, margin_threshold=0.10)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.58,  # < 0.65
        best_matching_location_id="clock_tower",
        best_matching_score=0.58,
        best_competing_location_id="library_atrium",
        best_competing_score=0.40,
        margin=0.18,
        target_ref_count=3,
    )
    assert result == "ambiguous"
    assert reason == "AMBIGUOUS_SCORE_BELOW_MATCH_THRESHOLD"


def test_ambiguous_due_to_narrow_margin():
    """Target score is high, but competitor score is too close (margin < margin_threshold)."""
    cfg = ScoringConfig(calibrated=True, match_threshold=0.65, margin_threshold=0.10)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.82,
        best_matching_location_id="clock_tower",
        best_matching_score=0.82,
        best_competing_location_id="library_atrium",
        best_competing_score=0.78,
        margin=0.04,  # 0.04 < 0.10
        target_ref_count=3,
    )
    assert result == "ambiguous"
    assert reason == "AMBIGUOUS_MARGIN_BELOW_THRESHOLD"


def test_mismatch_due_to_competing_location_clear_match():
    """A competing location decisively wins with strong score and wide margin."""
    cfg = ScoringConfig(calibrated=True, match_threshold=0.65, margin_threshold=0.10)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.45,
        best_matching_location_id="library_atrium",
        best_matching_score=0.88,
        best_competing_location_id="library_atrium",
        best_competing_score=0.88,
        margin=-0.43,
        target_ref_count=3,
    )
    assert result == "mismatch"
    assert reason == "MISMATCH_COMPETING_LOCATION_MATCHED"


def test_mismatch_due_to_very_low_target_score():
    """Target score is below mismatch_threshold."""
    cfg = ScoringConfig(
        calibrated=True,
        match_threshold=0.65,
        margin_threshold=0.10,
        mismatch_threshold=0.30,
    )
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.20,  # < 0.30
        best_matching_location_id="library_atrium",
        best_matching_score=0.55,
        best_competing_location_id="library_atrium",
        best_competing_score=0.55,
        margin=-0.35,
        target_ref_count=3,
    )
    assert result == "mismatch"
    assert reason == "MISMATCH_TARGET_SCORE_TOO_LOW"


def test_ambiguous_borderline_competitor():
    """Competitor scored higher than target, but by a very small margin (< margin_threshold)."""
    cfg = ScoringConfig(calibrated=True, match_threshold=0.65, margin_threshold=0.10)
    engine = DecisionEngine(config=cfg)

    result, reason = engine.decide(
        target_location_id="clock_tower",
        target_score=0.68,
        best_matching_location_id="library_atrium",
        best_matching_score=0.72,
        best_competing_location_id="library_atrium",
        best_competing_score=0.72,
        margin=-0.04,  # difference is only 0.04 < 0.10
        target_ref_count=3,
    )
    assert result == "ambiguous"
    assert reason == "AMBIGUOUS_BORDERLINE_COMPETITOR"

