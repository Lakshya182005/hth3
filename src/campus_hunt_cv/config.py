"""Configuration management for the campus hunt CV pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION = 5
DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES = 10
DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION = 5
DEFAULT_MIN_TEST_UNKNOWN_SAMPLES = 10


class ImageLimitsConfig(BaseModel):
    """Limits and security constraints for image ingestion."""

    model_config = ConfigDict(extra="forbid")

    max_image_bytes: int = Field(
        default=15 * 1024 * 1024,
        description="Maximum allowed image size in bytes (default: 15MB).",
        ge=1024,
    )
    max_image_pixels: int = Field(
        default=50_000_000,
        description="Maximum allowed total pixels (width * height) to avoid decompression bombs.",
        ge=10_000,
    )
    min_image_dimension: int = Field(
        default=32,
        description="Minimum width or height in pixels for valid image processing.",
        ge=1,
    )
    allowed_formats: Tuple[str, ...] = Field(
        default=("JPEG", "JPG", "PNG", "WEBP"),
        description="Allowed image formats.",
    )


class ModelConfig(BaseModel):
    """OpenCLIP model specification and execution settings."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        default="ViT-B-32",
        description="OpenCLIP architecture name (e.g. ViT-B-32).",
    )
    pretrained: str = Field(
        default="laion2b_s34b_b79k",
        description="Pretrained weights tag or checkpoint path.",
    )
    device: str = Field(
        default="cpu",
        description="Target compute device: 'cpu', 'cuda', 'mps', or 'auto'.",
    )
    num_threads: Optional[int] = Field(
        default=None,
        description="Optional number of PyTorch CPU threads for inference. None keeps default.",
    )

    @property
    def model_version(self) -> str:
        """Standardized model version string."""
        return f"{self.model_name}/{self.pretrained}"


class ScoringConfig(BaseModel):
    """Configuration for similarity score aggregation and decision thresholds."""

    model_config = ConfigDict(extra="forbid")

    aggregation_method: Literal["top_k_mean"] = Field(
        default="top_k_mean",
        description="Score aggregation method across reference photos for a location.",
    )
    top_k: int = Field(
        default=3,
        description="Top-k reference photos to average for a location's score.",
        ge=1,
    )
    calibrated: bool = Field(
        default=False,
        description=(
            "Must be set to True only after empirical calibration on labeled calibration data. "
            "When False, the verifier safely defaults to 'ambiguous'."
        ),
    )
    match_threshold: Optional[float] = Field(
        default=None,
        description="Calibrated minimum cosine similarity score to qualify as a strong match.",
        ge=-1.0,
        le=1.0,
    )
    margin_threshold: Optional[float] = Field(
        default=None,
        description=(
            "Calibrated minimum margin between target location score and best competing location score. "
            "Must be strictly positive to ensure distinctness over runner-up."
        ),
        ge=0.0,
        le=2.0,
    )
    mismatch_threshold: Optional[float] = Field(
        default=None,
        description=(
            "Optional calibrated score below which a submission is confidently ruled as a mismatch, "
            "provided a competing location dominates. Defaults to None unless empirically calibrated."
        ),
        ge=-1.0,
        le=1.0,
    )
    min_reference_images_per_location: int = Field(
        default=1,
        description="Minimum number of reference photos required for a location to be valid.",
        ge=1,
    )
    min_calibration_samples_per_location: int = Field(
        default=DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
        description="Minimum calibration captures per location required to run calibration (data-quality safeguard).",
        ge=1,
    )
    min_calibration_unknown_samples: int = Field(
        default=DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
        description="Minimum unknown/distractor captures required for calibration.",
        ge=1,
    )
    min_test_samples_per_location: int = Field(
        default=DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
        description="Minimum hold-out test captures per location required for complete evaluation.",
        ge=1,
    )
    min_test_unknown_samples: int = Field(
        default=DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
        description="Minimum unknown/distractor hold-out test captures required for complete evaluation.",
        ge=1,
    )
    target_max_far: float = Field(
        default=0.02,
        description="Maximum acceptable false acceptance rate constraint for calibration.",
        ge=0.0,
        le=1.0,
    )
    target_min_unknown_rejection: float = Field(
        default=0.90,
        description="Minimum acceptable unknown rejection rate constraint for calibration.",
        ge=0.0,
        le=1.0,
    )
    allow_unvalidated_auto_accept: bool = Field(
        default=False,
        description=(
            "DANGER / DEVELOPMENT ONLY: If True, permits automatic 'strong_match' outcomes even when "
            "calibration is provisional, hold-out test evaluation is unavailable or incomplete, or "
            "calibration metadata is missing. Defaults to False. Must NEVER be enabled in production."
        ),
    )


class StorageConfig(BaseModel):
    """File paths for reference images, calibration captures, test captures, and embeddings."""

    model_config = ConfigDict(extra="forbid")

    reference_data_dir: Path = Field(
        default=Path("data/reference"),
        description="Directory containing subdirectories of reference images per location.",
    )
    calibration_data_dir: Path = Field(
        default=Path("data/calibration"),
        description="Directory containing subdirectories of calibration captures (including unknown/).",
    )
    test_data_dir: Path = Field(
        default=Path("data/test"),
        description="Directory containing independent test captures (including unknown/) for final evaluation.",
    )
    embeddings_path: Path = Field(
        default=Path("data/reference_embeddings.npz"),
        description="Path to precomputed reference embeddings in safe NPZ format.",
    )
    metadata_path: Path = Field(
        default=Path("data/reference_embeddings.meta.json"),
        description="Path to JSON metadata companion for precomputed embeddings.",
    )

    @property
    def validation_data_dir(self) -> Path:
        """Backwards-compatibility alias for calibration_data_dir."""
        return self.calibration_data_dir


class CalibrationMetrics(BaseModel):
    """Performance metrics calculated on an evaluation split."""

    model_config = ConfigDict(extra="forbid")

    correct_accept_rate: float = Field(ge=0.0, le=1.0)
    wrong_false_accept_rate: float = Field(ge=0.0, le=1.0)
    unknown_rejection_rate: float = Field(ge=0.0, le=1.0)
    correct_trials: int = Field(default=0, ge=0)
    correct_accepts: int = Field(default=0, ge=0)
    wrong_trials: int = Field(default=0, ge=0)
    wrong_false_accepts: int = Field(default=0, ge=0)
    unknown_trials: int = Field(default=0, ge=0)
    unknown_rejections: int = Field(default=0, ge=0)


class TestEvaluationMetrics(BaseModel):
    """Outcome and metrics of evaluation on the independent test dataset."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "incomplete", "unavailable", "provisional"]
    reason: Optional[str] = None
    sample_counts: Optional[Dict[str, int]] = None
    missing_categories: Optional[List[str]] = None
    minimums_used: Optional[Dict[str, int]] = None
    is_provisional: bool = False
    metrics: Optional[CalibrationMetrics] = None


class CalibrationMetadata(BaseModel):
    """Provenance and audit metadata for threshold calibration."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["calibrated", "provisional"] = "calibrated"
    is_provisional: bool = False
    warning: Optional[str] = None
    minimums_used: Optional[Dict[str, int]] = None
    model_version: str
    calibrated_at: str
    calibration_sample_counts: Dict[str, int]
    calibration_metrics: CalibrationMetrics
    test_evaluation: TestEvaluationMetrics


class CampusHuntConfig(BaseModel):
    """Unified configuration for the campus hunt verification pipeline."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    image_limits: ImageLimitsConfig = Field(default_factory=ImageLimitsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    calibration_metadata: Optional[CalibrationMetadata] = None

    @classmethod
    def from_env(cls) -> CampusHuntConfig:
        """Create configuration with environment variable overrides."""
        cfg = cls()

        if "CAMPUS_HUNT_MODEL_NAME" in os.environ:
            cfg.model.model_name = os.environ["CAMPUS_HUNT_MODEL_NAME"]
        if "CAMPUS_HUNT_PRETRAINED" in os.environ:
            cfg.model.pretrained = os.environ["CAMPUS_HUNT_PRETRAINED"]
        if "CAMPUS_HUNT_DEVICE" in os.environ:
            cfg.model.device = os.environ["CAMPUS_HUNT_DEVICE"]
        if "CAMPUS_HUNT_TOP_K" in os.environ:
            cfg.scoring.top_k = int(os.environ["CAMPUS_HUNT_TOP_K"])
        if "CAMPUS_HUNT_CALIBRATED" in os.environ:
            cfg.scoring.calibrated = os.environ["CAMPUS_HUNT_CALIBRATED"].lower() in ("1", "true", "yes")
        if "CAMPUS_HUNT_MATCH_THRESHOLD" in os.environ:
            cfg.scoring.match_threshold = float(os.environ["CAMPUS_HUNT_MATCH_THRESHOLD"])
        if "CAMPUS_HUNT_MARGIN_THRESHOLD" in os.environ:
            cfg.scoring.margin_threshold = float(os.environ["CAMPUS_HUNT_MARGIN_THRESHOLD"])
        if "CAMPUS_HUNT_MISMATCH_THRESHOLD" in os.environ:
            cfg.scoring.mismatch_threshold = float(os.environ["CAMPUS_HUNT_MISMATCH_THRESHOLD"])
        if "CAMPUS_HUNT_TARGET_MAX_FAR" in os.environ:
            cfg.scoring.target_max_far = float(os.environ["CAMPUS_HUNT_TARGET_MAX_FAR"])
        if "CAMPUS_HUNT_TARGET_MIN_UNKNOWN_REJECTION" in os.environ:
            cfg.scoring.target_min_unknown_rejection = float(os.environ["CAMPUS_HUNT_TARGET_MIN_UNKNOWN_REJECTION"])
        if "CAMPUS_HUNT_ALLOW_UNVALIDATED_AUTO_ACCEPT" in os.environ:
            cfg.scoring.allow_unvalidated_auto_accept = (
                os.environ["CAMPUS_HUNT_ALLOW_UNVALIDATED_AUTO_ACCEPT"].lower() in ("1", "true", "yes")
            )
        if "CAMPUS_HUNT_MIN_CALIB_SAMPLES" in os.environ:
            cfg.scoring.min_calibration_samples_per_location = int(os.environ["CAMPUS_HUNT_MIN_CALIB_SAMPLES"])
        if "CAMPUS_HUNT_MIN_CALIB_UNKNOWN" in os.environ:
            cfg.scoring.min_calibration_unknown_samples = int(os.environ["CAMPUS_HUNT_MIN_CALIB_UNKNOWN"])
        if "CAMPUS_HUNT_MIN_TEST_SAMPLES" in os.environ:
            cfg.scoring.min_test_samples_per_location = int(os.environ["CAMPUS_HUNT_MIN_TEST_SAMPLES"])
        if "CAMPUS_HUNT_MIN_TEST_UNKNOWN" in os.environ:
            cfg.scoring.min_test_unknown_samples = int(os.environ["CAMPUS_HUNT_MIN_TEST_UNKNOWN"])
        if "CAMPUS_HUNT_EMBEDDINGS_PATH" in os.environ:
            cfg.storage.embeddings_path = Path(os.environ["CAMPUS_HUNT_EMBEDDINGS_PATH"])
        if "CAMPUS_HUNT_METADATA_PATH" in os.environ:
            cfg.storage.metadata_path = Path(os.environ["CAMPUS_HUNT_METADATA_PATH"])

        return cfg

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> CampusHuntConfig:
        """Load configuration from a documented JSON file.

        Enforces strict schema validation (`extra='forbid'`) so misplaced or
        unknown calibration fields are immediately flagged rather than silently ignored.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json_file(self, path: Union[str, Path]) -> None:
        """Persist configuration to a formatted JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
