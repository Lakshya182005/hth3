"""Decision logic engine for campus location verification.

Translates similarity scores and margins into conservative verification outcomes:
`strong_match`, `ambiguous`, or `mismatch`.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple
from campus_hunt_cv.config import ScoringConfig

VerificationResult = Literal["strong_match", "ambiguous", "mismatch"]


class DecisionEngine:
    """Evaluates similarity scores against empirical thresholds to produce conservative verdicts."""

    def __init__(self, config: Optional[ScoringConfig] = None):
        self.config = config or ScoringConfig()

    def decide(
        self,
        target_location_id: str,
        target_score: Optional[float],
        best_matching_location_id: Optional[str],
        best_matching_score: Optional[float],
        best_competing_location_id: Optional[str],
        best_competing_score: Optional[float],
        margin: Optional[float],
        target_ref_count: int,
    ) -> Tuple[VerificationResult, str]:
        """Make a conservative verification decision based on calibrated thresholds.

        Rules:
        1. If thresholds are uncalibrated: ALWAYS return 'ambiguous' (UNCALIBRATED_THRESHOLDS).
        2. If target location is unknown or has insufficient reference photos: return 'ambiguous'.
        3. 'strong_match': target must be best match, target_score >= match_threshold,
           and margin >= margin_threshold.
        4. 'mismatch':
           - A competing location clearly matches (competing score >= match_threshold AND
             competing score - target_score >= margin_threshold).
           - OR target_score < mismatch_threshold (if configured).
        5. 'ambiguous': All other borderline cases, ties, or low-confidence results.

        Args:
            target_location_id: Server-authoritative target location ID.
            target_score: Aggregated score for target location.
            best_matching_location_id: Highest scoring location ID.
            best_matching_score: Highest aggregated score.
            best_competing_location_id: Runner-up location ID.
            best_competing_score: Runner-up aggregated score.
            margin: target_score - best_competing_score.
            target_ref_count: Number of reference photos available for target location.

        Returns:
            Tuple of (result, reason_code).
        """
        # 1. Uncalibrated safety gate: Never silently use made-up thresholds
        if (
            not self.config.calibrated
            or self.config.match_threshold is None
            or self.config.margin_threshold is None
        ):
            return "ambiguous", "UNCALIBRATED_THRESHOLDS"

        # 2. Reference data integrity checks
        if target_score is None:
            return "ambiguous", "UNKNOWN_TARGET_LOCATION"

        if target_ref_count < self.config.min_reference_images_per_location:
            return "ambiguous", "INSUFFICIENT_REFERENCE_DATA"

        match_th = self.config.match_threshold
        margin_th = self.config.margin_threshold
        mismatch_th = self.config.mismatch_threshold

        is_target_best = (best_matching_location_id == target_location_id)

        # 3. Check for Strong Match
        if is_target_best:
            score_meets_th = target_score >= match_th
            # If only 1 location exists in total, competitor is None and margin is None
            margin_meets_th = (margin is None) or (margin >= margin_th)

            if score_meets_th and margin_meets_th:
                return "strong_match", "STRONG_MATCH"
            elif not score_meets_th:
                return "ambiguous", "AMBIGUOUS_SCORE_BELOW_MATCH_THRESHOLD"
            else:
                return "ambiguous", "AMBIGUOUS_MARGIN_BELOW_THRESHOLD"

        # 4. Check for Mismatch (Target is NOT best match)
        assert best_matching_location_id is not None
        assert best_matching_score is not None

        # Check if a competing location decisively won
        competing_margin = best_matching_score - target_score
        competing_decisive_win = (
            best_matching_score >= match_th and competing_margin >= margin_th
        )

        if competing_decisive_win:
            return "mismatch", "MISMATCH_COMPETING_LOCATION_MATCHED"

        # Check if target score is unequivocally below mismatch rejection threshold
        if mismatch_th is not None and target_score < mismatch_th:
            return "mismatch", "MISMATCH_TARGET_SCORE_TOO_LOW"

        # 5. Borderline cases where target lost but competitor win wasn't decisive
        if competing_margin < margin_th:
            return "ambiguous", "AMBIGUOUS_BORDERLINE_COMPETITOR"

        return "ambiguous", "AMBIGUOUS_LOW_CONFIDENCE"

