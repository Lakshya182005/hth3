"""Cosine similarity computation and score aggregation routines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from campus_hunt_cv.config import ScoringConfig


def compute_cosine_similarity(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarities between a normalized query embedding and reference embeddings.

    Because both query and reference vectors are strictly unit L2-normalized:
        cosine_similarity(q, r) = dot(q, r) / (||q|| * ||r||) = dot(q, r)

    Args:
        query_embedding: 1D float32 array of shape (D,) or (1, D).
        reference_embeddings: 2D float32 array of shape (N, D).

    Returns:
        1D float32 array of shape (N,) containing similarity values in [-1.0, 1.0].
    """
    if query_embedding.ndim == 1:
        query_vector = query_embedding
    else:
        query_vector = query_embedding.reshape(-1)

    if reference_embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D reference_embeddings array, got shape {reference_embeddings.shape}."
        )

    if reference_embeddings.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    # Matrix-vector dot product
    sims = np.dot(reference_embeddings, query_vector)
    # Clamp to [-1.0, 1.0] to guard against floating-point epsilon leakage
    return np.clip(sims, -1.0, 1.0)


def aggregate_top_k_mean(similarities: np.ndarray, k: int = 3) -> Optional[float]:
    """Aggregate per-photo cosine similarities into a single location score using top-k mean.

    Strategy:
    1. If N >= k: take the arithmetic mean of the top-k highest similarity scores.
    2. Fallback (0 < N < k): if fewer than k reference photos exist for this location,
       take the mean of all available N reference photos.
    3. If N == 0: return None.

    Args:
        similarities: 1D array of cosine similarities for a single location.
        k: Number of highest scores to average. Default is 3.

    Returns:
        Aggregated float score in [-1.0, 1.0], or None if no reference photos exist.
    """
    n = len(similarities)
    if n == 0:
        return None

    if n <= k:
        return float(np.mean(similarities))

    # Fast partition for top-k elements without full sort
    partitioned = np.partition(similarities, -k)[-k:]
    return float(np.mean(partitioned))


class SimilarityScorer:
    """Computes similarity scores across all candidate locations and extracts margins."""

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def score_all_locations(
        self,
        query_embedding: np.ndarray,
        embeddings_by_location: Dict[str, np.ndarray],
    ) -> Dict[str, Optional[float]]:
        """Compute aggregated similarity score for every configured location.

        Args:
            query_embedding: Unit L2-normalized 1D array of shape (D,).
            embeddings_by_location: Mapping from location_id to (N_loc, D) reference arrays.

        Returns:
            Dictionary mapping each location_id to its aggregated similarity score (or None).
        """
        scores: Dict[str, Optional[float]] = {}
        k = self.config.top_k

        for loc_id, ref_embs in embeddings_by_location.items():
            if ref_embs.shape[0] == 0:
                scores[loc_id] = None
                continue

            sims = compute_cosine_similarity(query_embedding, ref_embs)
            scores[loc_id] = aggregate_top_k_mean(sims, k=k)

        return scores

    def extract_ranking_and_margin(
        self,
        scores: Dict[str, Optional[float]],
        target_location_id: str,
    ) -> Tuple[
        Optional[float],  # target_score
        Optional[str],    # best_matching_location_id
        Optional[float],  # best_matching_score
        Optional[str],    # best_competing_location_id
        Optional[float],  # best_competing_score
        Optional[float],  # margin
    ]:
        """Rank locations, identify best match, strongest competitor, and margin.

        Definition of competitor:
        - The strongest competitor is the highest-scoring location OTHER than target_location_id.
        - If target_location_id is the best match, the competitor is the runner-up.
        - If target_location_id is not the best match, the competitor is the best match.

        Definition of margin:
            margin = target_score - best_competing_score
        - A positive margin means the target is winning over all alternatives.
        - A negative margin means at least one competitor outperforms the target.

        Args:
            scores: Mapping of location_id -> aggregated score.
            target_location_id: The server-authoritative expected location ID.

        Returns:
            Tuple of (target_score, best_matching_location_id, best_matching_score,
                      best_competing_location_id, best_competing_score, margin)
        """
        target_score = scores.get(target_location_id)

        # Filter out locations with None score
        valid_scores = [(loc, score) for loc, score in scores.items() if score is not None]

        if not valid_scores:
            return target_score, None, None, None, None, None

        # Sort descending by score
        sorted_locations = sorted(valid_scores, key=lambda x: x[1], reverse=True)

        best_matching_location_id, best_matching_score = sorted_locations[0]

        # Find the highest-scoring location that is NOT target_location_id
        best_competing_location_id: Optional[str] = None
        best_competing_score: Optional[float] = None

        for loc, score in sorted_locations:
            if loc != target_location_id:
                best_competing_location_id = loc
                best_competing_score = score
                break

        # Compute margin
        margin: Optional[float] = None
        if target_score is not None and best_competing_score is not None:
            margin = round(float(target_score - best_competing_score), 6)

        return (
            round(float(target_score), 6) if target_score is not None else None,
            best_matching_location_id,
            round(float(best_matching_score), 6) if best_matching_score is not None else None,
            best_competing_location_id,
            round(float(best_competing_score), 6) if best_competing_score is not None else None,
            margin,
        )

