"""Configuration management for the campus hunt CV pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator
from campus_hunt_cv.exceptions import ConfigurationError


DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION = 5
DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES = 10
DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION = 5
DEFAULT_MIN_TEST_UNKNOWN_SAMPLES = 10


def normalize_environment(val: Optional[str]) -> str:
    """Normalize runtime environment string with safe fallback to 'production'.

    Supported values: 'development', 'test', 'production' (case-insensitive).
    Defaults to 'production' if unset or invalid.
    """
    if not val or not isinstance(val, str):
        return "production"
    cleaned = val.strip().lower()
    if cleaned in ("development", "dev"):
        return "development"
    if cleaned in ("test", "testing"):
        return "test"
    if cleaned in ("production", "prod"):
        return "production"
    return "production"


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
    unknown_rejection_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    correct_trials: int = Field(default=0, ge=0)
    correct_accepts: int = Field(default=0, ge=0)
    wrong_trials: int = Field(default=0, ge=0)
    wrong_false_accepts: int = Field(default=0, ge=0)
    unknown_trials: Optional[int] = Field(default=None, ge=0)
    unknown_rejections: Optional[int] = Field(default=None, ge=0)


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

    environment: str = Field(
        default="production",
        description=(
            "Runtime deployment environment: 'development', 'test', or 'production'. "
            "Defaults strictly to 'production' if unset or invalid."
        ),
    )
    model: ModelConfig = Field(default_factory=ModelConfig)
    image_limits: ImageLimitsConfig = Field(default_factory=ImageLimitsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    calibration_metadata: Optional[CalibrationMetadata] = None

    @field_validator("environment", mode="before")
    @classmethod
    def _validate_environment(cls, v: Any) -> str:
        return normalize_environment(v)

    @property
    def effective_environment(self) -> str:
        """Determine the effective runtime environment with strict production default.

        Precedence:
        1. Host process environment variable `CAMPUS_HUNT_ENVIRONMENT` if explicitly set.
        2. Configured `environment` field on this instance if set and non-empty.
        3. Safe default: 'production'.

        All values are normalized via `normalize_environment()`.
        Unset, empty, or unrecognized values strictly resolve to 'production'.
        """
        env_from_os = os.environ.get("CAMPUS_HUNT_ENVIRONMENT")
        if env_from_os is not None and env_from_os.strip() != "":
            return normalize_environment(env_from_os)
        return normalize_environment(getattr(self, "environment", "production"))

    @property
    def is_production(self) -> bool:
        """Check whether the effective runtime environment is production."""
        return self.effective_environment == "production"

    def validate_for_production(self) -> None:
        """Validate that the configuration is safe for production deployment.

        Enforces that unsafe development overrides are not active in production.

        Raises:
            ConfigurationError: If allow_unvalidated_auto_accept is enabled in production,
                or if calibration/hold-out test metadata is incomplete or provisional.
        """
        if self.is_production and self.scoring.allow_unvalidated_auto_accept:
            raise ConfigurationError(
                "Unsafe configuration rejected: 'allow_unvalidated_auto_accept' cannot be enabled "
                f"in production environment (effective environment: '{self.effective_environment}'). "
                "Set CAMPUS_HUNT_ENVIRONMENT=development for local development testing.",
                reason_code="ERROR_UNSAFE_PRODUCTION_CONFIG",
            )
        if self.is_production:
            if self.calibration_metadata is None:
                raise ConfigurationError(
                    "Production configuration rejected: calibration metadata is missing. "
                    "Run campus-hunt-calibrate to produce a validated configuration.",
                    reason_code="ERROR_UNVALIDATED_PRODUCTION_CONFIG",
                )
            if self.calibration_metadata.status == "provisional" or self.calibration_metadata.is_provisional:
                raise ConfigurationError(
                    "Production configuration rejected: calibration status is provisional. "
                    "Adequate calibration samples must be collected before deploying to production.",
                    reason_code="ERROR_PROVISIONAL_PRODUCTION_CONFIG",
                )
            test_eval = self.calibration_metadata.test_evaluation
            if test_eval is None or test_eval.status != "completed" or test_eval.is_provisional:
                raise ConfigurationError(
                    "Production configuration rejected: hold-out test evaluation is incomplete or unavailable. "
                    "Complete independent test evaluation must pass before deploying to production.",
                    reason_code="ERROR_INCOMPLETE_TEST_PRODUCTION_CONFIG",
                )

    @classmethod
    def from_env(cls) -> CampusHuntConfig:
        """Create configuration with environment variable overrides."""
        cfg = cls()

        if "CAMPUS_HUNT_ENVIRONMENT" in os.environ:
            cfg.environment = normalize_environment(os.environ["CAMPUS_HUNT_ENVIRONMENT"])
        else:
            cfg.environment = "production"

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
        Runtime environment defaults strictly to 'production' unless overridden by host environment.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls.model_validate(data)
        if "CAMPUS_HUNT_ENVIRONMENT" in os.environ:
            cfg.environment = normalize_environment(os.environ["CAMPUS_HUNT_ENVIRONMENT"])
        elif cfg.environment != "production":
            # Config file cannot override host environment to development when unset
            cfg.environment = "production"
        return cfg

    def to_json_file(self, path: Union[str, Path]) -> None:
        """Persist configuration to a formatted JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)
