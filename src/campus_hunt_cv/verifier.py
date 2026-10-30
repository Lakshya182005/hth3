"""Callable location verification component for the campus treasure hunt."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union
from pydantic import BaseModel, Field

from campus_hunt_cv.config import CampusHuntConfig
from campus_hunt_cv.decision import DecisionEngine, VerificationResult
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.exceptions import (
    CampusHuntError,
    ImageValidationError,
    ModelInferenceError,
    ReferenceDataError,
)
from campus_hunt_cv.image_validation import validate_and_decode_image
from campus_hunt_cv.model import EmbedderProtocol, OpenCLIPLocationEmbedder
from campus_hunt_cv.similarity import SimilarityScorer

logger = logging.getLogger(__name__)


class VerificationResponse(BaseModel):
    """Structured response conforming to the treasure hunt backend contract."""

    result: VerificationResult = Field(
        description="Verification outcome: 'strong_match', 'ambiguous', or 'mismatch'."
    )
    target_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score for the server-authoritative target location.",
    )
    best_matching_location_id: Optional[str] = Field(
        default=None,
        description="Location ID with the highest aggregated similarity score across all known hunt locations.",
    )
    best_matching_score: Optional[float] = Field(
        default=None,
        description="Aggregated similarity score of the best matching location.",
    )
    best_competing_location_id: Optional[str] = Field(
        default=None,
        description="Strongest competing location ID (runner-up if target won, or highest competitor).",
    )
    best_competing_score: Optional[float] = Field(
        default=None,
        description="Aggregated similarity score of the strongest competing location.",
    )
    margin: Optional[float] = Field(
        default=None,
        description="Score margin: target_score minus best_competing_score.",
    )
    reason_code: str = Field(
        description="Machine-readable code detailing the rationale for the verdict."
    )
    model_version: str = Field(
        description="Identifying model architecture and checkpoint string."
    )


def check_config_validation_status(config: CampusHuntConfig) -> Tuple[bool, Optional[str]]:
    """Verify whether a configuration is authorized for automatic 'strong_match' progression.

    Rules:
    1. Explicit Opt-In: If config.scoring.allow_unvalidated_auto_accept is True,
       automatic acceptance is permitted (intended strictly for development/controlled testing).
    2. Missing Metadata: If config.calibration_metadata is None, calibration provenance
       and hold-out test completion cannot be verified. Fails safe to 'HOLDOUT_TEST_NOT_COMPLETED'.
    3. Provisional Calibration: If config.calibration_metadata.status == 'provisional'
       or config.calibration_metadata.is_provisional is True, sample safeguards were lowered
       below project standards. Fails safe to 'PROVISIONAL_CALIBRATION_REQUIRES_REVIEW'.
    4. Incomplete/Unavailable/Provisional Hold-Out Evaluation: If test_evaluation is missing, or its
       status is not 'completed', or is_provisional is True, independent hold-out test evaluation
       did not verify generalizability. Fails safe to 'HOLDOUT_TEST_NOT_COMPLETED'.
    5. Fully Validated: Both calibration and hold-out test splits met all quality standards
       without provisional overrides. Returns (True, None).
    """
    if config.scoring.allow_unvalidated_auto_accept:
        return True, None

    meta = config.calibration_metadata
    if meta is None:
        return False, "HOLDOUT_TEST_NOT_COMPLETED"

    if meta.status == "provisional" or meta.is_provisional:
        return False, "PROVISIONAL_CALIBRATION_REQUIRES_REVIEW"

    test_eval = meta.test_evaluation
    if test_eval is None or test_eval.status != "completed" or test_eval.is_provisional:
        return False, "HOLDOUT_TEST_NOT_COMPLETED"

    return True, None


class LocationVerifier:
    """Production-ready location verification component.

    Designed for low-latency backend integration (e.g. FastAPI / Flask / Celery):
    - Reuses preloaded OpenCLIP model across requests.
    - Reuses in-memory reference embeddings across requests.
    - Compares submitted photos against ALL configured hunt locations.
    - Enforces calibrated thresholds and fails safe to 'ambiguous'.
    - Routes model or processing errors to manual review instead of failing participants.
    """

    def __init__(
        self,
        config: Optional[CampusHuntConfig] = None,
        embedder: Optional[EmbedderProtocol] = None,
        store: Optional[ReferenceEmbeddingsStore] = None,
    ):
        """Initialize verifier with optional preloaded embedder and embeddings store.

        Args:
            config: Pipeline configuration. If None, loaded from environment defaults.
            embedder: Embedding model instance. If None, initialized using config.
            store: Reference embeddings store. If None, loaded from configured paths if available.
        """
        self.config = config or CampusHuntConfig.from_env()

        # Model embedder
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = OpenCLIPLocationEmbedder(
                config=self.config.model,
                lazy_load=False,
            )

        self._store_load_error: Optional[ReferenceDataError] = None

        # Reference embeddings store
        if store is not None:
            self.store = store
            if self.store.model_version != self.embedder.model_version:
                logger.error(
                    "Store model version '%s' does not match embedder model version '%s'.",
                    self.store.model_version,
                    self.embedder.model_version,
                )
        else:
            self.store = self._try_load_store()

        self.scorer = SimilarityScorer(config=self.config.scoring)
        self.decision_engine = DecisionEngine(config=self.config.scoring)

    def _try_load_store(self) -> Optional[ReferenceEmbeddingsStore]:
        """Attempt to load embeddings from disk; record error if incompatible."""
        npz_p = self.config.storage.embeddings_path
        meta_p = self.config.storage.metadata_path
        if npz_p.exists() and meta_p.exists():
            try:
                store = ReferenceEmbeddingsStore.load(
                    npz_path=npz_p,
                    metadata_path=meta_p,
                    expected_model_version=self.embedder.model_version,
                )
                self._store_load_error = None
                return store
            except ReferenceDataError as e:
                self._store_load_error = e
                logger.error("Failed to load reference embeddings: %s", e.message)
                return None
            except Exception as e:
                self._store_load_error = ReferenceDataError(str(e))
                logger.error("Failed to load reference embeddings from disk: %s", str(e))
                return None
        return None

    def verify(
        self,
        image_bytes: bytes,
        target_location_id: str,
    ) -> Dict[str, Any]:
        """Verify whether submitted image bytes match the server-authoritative target location.

        IMPORTANT SECURITY CONTRACT:
        The caller must supply the team's server-authoritative current location in `target_location_id`.
        Never accept or pass an unverified client-supplied target location from the participant's device.

        Args:
            image_bytes: Raw binary bytes of the participant's submitted photo.
            target_location_id: Verified expected location ID from game backend state.

        Returns:
            Dictionary matching the treasure hunt verification schema.
        """
        model_version = self.embedder.model_version

        # 1. Check reference embeddings store availability
        if self.store is None:
            # Re-attempt loading in case files were generated after initialization
            self.store = self._try_load_store()

        # Enforce model version compatibility: never silently compare incompatible embeddings
        if self.store is not None and self.store.model_version != self.embedder.model_version:
            return VerificationResponse(
                result="ambiguous",
                target_score=None,
                best_matching_location_id=None,
                best_matching_score=None,
                best_competing_location_id=None,
                best_competing_score=None,
                margin=None,
                reason_code="ERROR_MODEL_VERSION_MISMATCH",
                model_version=model_version,
            ).model_dump()

        if self.store is None or len(self.store.locations) == 0:
            reason = (
                self._store_load_error.reason_code
                if self._store_load_error is not None
                else "ERROR_NO_REFERENCE_DATA_LOADED"
            )
            return VerificationResponse(
                result="ambiguous",
                target_score=None,
                best_matching_location_id=None,
                best_matching_score=None,
                best_competing_location_id=None,
                best_competing_score=None,
                margin=None,
                reason_code=reason,
                model_version=model_version,
            ).model_dump()

        # 2. Validate and decode submitted image
        try:
            image = validate_and_decode_image(image_bytes, limits=self.config.image_limits)
        except ImageValidationError as e:
            logger.warning("Image validation failed for target '%s': %s", target_location_id, e.message)
            return VerificationResponse(
                result="ambiguous",
                target_score=None,
                best_matching_location_id=None,
                best_matching_score=None,
                best_competing_location_id=None,
                best_competing_score=None,
                margin=None,
                reason_code=e.reason_code,
                model_version=model_version,
            ).model_dump()
        except Exception as e:
            logger.error("Unexpected error during image validation: %s", str(e))
            return VerificationResponse(
                result="ambiguous",
                target_score=None,
                best_matching_location_id=None,
                best_matching_score=None,
                best_competing_location_id=None,
                best_competing_score=None,
                margin=None,
                reason_code="ERROR_IMAGE_DECODE_FAILED",
                model_version=model_version,
            ).model_dump()

        # 3. Compute normalized query embedding
        try:
            query_embedding = self.embedder.embed_image(image)
        except Exception as e:
            logger.error("Model forward pass failed: %s", str(e))
            # AI failure routed as an error for manual review, not a player failure
            return VerificationResponse(
                result="ambiguous",
                target_score=None,
                best_matching_location_id=None,
                best_matching_score=None,
                best_competing_location_id=None,
                best_competing_score=None,
                margin=None,
                reason_code="ERROR_MODEL_INFERENCE_FAILURE",
                model_version=model_version,
            ).model_dump()

        # 4. Compare against ALL known locations
        scores = self.scorer.score_all_locations(
            query_embedding=query_embedding,
            embeddings_by_location=self.store.all_embeddings_by_location,
        )

        # 5. Extract ranking, competitor, and margin
        (
            target_score,
            best_matching_location_id,
            best_matching_score,
            best_competing_location_id,
            best_competing_score,
            margin,
        ) = self.scorer.extract_ranking_and_margin(scores, target_location_id=target_location_id)

        target_ref_count = self.store.get_location_count(target_location_id)

        # 6. Apply conservative decision logic
        result, reason_code = self.decision_engine.decide(
            target_location_id=target_location_id,
            target_score=target_score,
            best_matching_location_id=best_matching_location_id,
            best_matching_score=best_matching_score,
            best_competing_location_id=best_competing_location_id,
            best_competing_score=best_competing_score,
            margin=margin,
            target_ref_count=target_ref_count,
        )

        # 7. Runtime safety gate: provisional calibration or incomplete hold-out test
        # must never produce an automatic strong_match unless explicitly opted in.
        if result == "strong_match":
            is_validated, rejection_reason = check_config_validation_status(self.config)
            if not is_validated:
                logger.warning(
                    "Downgrading strong_match to ambiguous for target '%s': %s (target_score=%.3f, margin=%.3f)",
                    target_location_id,
                    rejection_reason,
                    target_score if target_score is not None else 0.0,
                    margin if margin is not None else 0.0,
                )
                result = "ambiguous"
                reason_code = rejection_reason or "HOLDOUT_TEST_NOT_COMPLETED"

        return VerificationResponse(
            result=result,
            target_score=target_score,
            best_matching_location_id=best_matching_location_id,
            best_matching_score=best_matching_score,
            best_competing_location_id=best_competing_location_id,
            best_competing_score=best_competing_score,
            margin=margin,
            reason_code=reason_code,
            model_version=model_version,
        ).model_dump()


# Module-level cached verifier for standalone calls
_GLOBAL_VERIFIER: Optional[LocationVerifier] = None


def get_global_verifier() -> LocationVerifier:
    """Retrieve or initialize the global shared LocationVerifier instance."""
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = LocationVerifier()
    return _GLOBAL_VERIFIER


def verify_location(
    image_bytes: bytes,
    target_location_id: str,
    *,
    verifier: Optional[LocationVerifier] = None,
) -> Dict[str, Any]:
    """Verify whether submitted image bytes appear to match the server-authoritative location.

    Standard public entrypoint for backend integration.

    Example:
        >>> result = verify_location(photo_bytes, target_location_id="clock_tower")
        >>> print(result["result"])
        'strong_match'

    Args:
        image_bytes: Raw binary photo submitted by participant.
        target_location_id: Verified server-authoritative location ID.
        verifier: Optional LocationVerifier instance. If None, uses shared singleton.

    Returns:
        Structured response dictionary matching the treasure hunt verification schema.
    """
    active_verifier = verifier if verifier is not None else get_global_verifier()
    return active_verifier.verify(image_bytes, target_location_id=target_location_id)

