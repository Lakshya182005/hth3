"""Unit tests for cosine similarity calculation, top-k aggregation, and margin extraction."""

import numpy as np
import pytest

from campus_hunt_cv.config import ScoringConfig
from campus_hunt_cv.similarity import (
    aggregate_top_k_mean,
    compute_cosine_similarity,
    SimilarityScorer,
)


def test_cosine_similarity_identical_vectors():
    """Identical unit vectors must yield a cosine similarity of 1.0."""
    v = np.array([0.6, 0.8], dtype=np.float32)
    ref = np.array([[0.6, 0.8]], dtype=np.float32)
    sim = compute_cosine_similarity(v, ref)
    assert np.isclose(sim[0], 1.0, atol=1e-5)


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors must yield a cosine similarity of 0.0."""
    v = np.array([1.0, 0.0], dtype=np.float32)
    ref = np.array([[0.0, 1.0]], dtype=np.float32)
    sim = compute_cosine_similarity(v, ref)
    assert np.isclose(sim[0], 0.0, atol=1e-5)


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors must yield a cosine similarity of -1.0."""
    v = np.array([1.0, 0.0], dtype=np.float32)
    ref = np.array([[-1.0, 0.0]], dtype=np.float32)
    sim = compute_cosine_similarity(v, ref)
    assert np.isclose(sim[0], -1.0, atol=1e-5)


def test_top_k_mean_aggregation_exact():
    """Top-3 mean should average exactly the 3 highest scores."""
    sims = np.array([0.2, 0.9, 0.5, 0.8, 0.7, 0.1], dtype=np.float32)
    # Top 3 are: 0.9, 0.8, 0.7 -> mean = 0.8
    score = aggregate_top_k_mean(sims, k=3)
    assert score is not None
    assert np.isclose(score, 0.8, atol=1e-5)


def test_top_k_mean_fallback_when_fewer_than_k():
    """When fewer than k photos exist (e.g. 2 photos, k=3), average all available."""
    sims = np.array([0.7, 0.9], dtype=np.float32)
    # N=2 < k=3 -> mean(0.7, 0.9) = 0.8
    score = aggregate_top_k_mean(sims, k=3)
    assert score is not None
    assert np.isclose(score, 0.8, atol=1e-5)


def test_top_k_mean_single_reference():
    """When exactly 1 photo exists (N=1 < k=3), score is that photo's similarity."""
    sims = np.array([0.85], dtype=np.float32)
    score = aggregate_top_k_mean(sims, k=3)
    assert score is not None
    assert np.isclose(score, 0.85, atol=1e-5)


def test_top_k_mean_empty_reference():
    """When 0 reference photos exist, score must be None."""
    sims = np.array([], dtype=np.float32)
    score = aggregate_top_k_mean(sims, k=3)
    assert score is None


def test_extract_ranking_and_margin_target_wins():
    """When target location is top-1, margin must be target_score - runner_up_score > 0."""
    scores = {
        "clock_tower": 0.82,
        "library_atrium": 0.65,
        "science_quad": 0.40,
    }
    scorer = SimilarityScorer()
    (
        target_score,
        best_id,
        best_score,
        competing_id,
        competing_score,
        margin,
    ) = scorer.extract_ranking_and_margin(scores, target_location_id="clock_tower")

    assert target_score == 0.82
    assert best_id == "clock_tower"
    assert best_score == 0.82
    assert competing_id == "library_atrium"
    assert competing_score == 0.65
    assert margin == pytest.approx(0.17, abs=1e-4)


def test_extract_ranking_and_margin_target_loses():
    """When target location is not top-1, margin must be target_score - best_competing_score < 0."""
    scores = {
        "clock_tower": 0.50,
        "library_atrium": 0.85,
        "science_quad": 0.40,
    }
    scorer = SimilarityScorer()
    (
        target_score,
        best_id,
        best_score,
        competing_id,
        competing_score,
        margin,
    ) = scorer.extract_ranking_and_margin(scores, target_location_id="clock_tower")

    assert target_score == 0.50
    assert best_id == "library_atrium"
    assert best_score == 0.85
    assert competing_id == "library_atrium"
    assert competing_score == 0.85
    assert margin == pytest.approx(-0.35, abs=1e-4)


def test_extract_ranking_and_margin_single_location():
    """When only 1 location exists in total, runner-up and margin should be None."""
    scores = {"clock_tower": 0.78}
    scorer = SimilarityScorer()
    (
        target_score,
        best_id,
        best_score,
        competing_id,
        competing_score,
        margin,
    ) = scorer.extract_ranking_and_margin(scores, target_location_id="clock_tower")

    assert target_score == 0.78
    assert best_id == "clock_tower"
    assert best_score == 0.78
    assert competing_id is None
    assert competing_score is None
    assert margin is None


def test_extract_ranking_and_margin_unknown_target():
    """When target location is not in scores dictionary, target_score is None."""
    scores = {"library_atrium": 0.75, "science_quad": 0.60}
    scorer = SimilarityScorer()
    (
        target_score,
        best_id,
        best_score,
        competing_id,
        competing_score,
        margin,
    ) = scorer.extract_ranking_and_margin(scores, target_location_id="nonexistent_loc")

    assert target_score is None
    assert best_id == "library_atrium"
    assert best_score == 0.75
    assert competing_id == "library_atrium"
    assert competing_score == 0.75
    assert margin is None

