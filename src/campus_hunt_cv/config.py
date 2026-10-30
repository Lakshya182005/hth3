"""Configuration management for the campus hunt CV pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class ImageLimitsConfig(BaseModel):
    """Limits and security constraints for image ingestion."""

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
            "Must be set to True only after empirical calibration on labeled validation data. "
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
            "provided a competing location dominates."
        ),
        ge=-1.0,
        le=1.0,
    )
    min_reference_images_per_location: int = Field(
        default=1,
        description="Minimum number of reference photos required for a location to be valid.",
        ge=1,
    )


class StorageConfig(BaseModel):
    """File paths for reference images, validation images, and precomputed embeddings."""

    reference_data_dir: Path = Field(
        default=Path("data/reference"),
        description="Directory containing subdirectories of reference images per location.",
    )
    validation_data_dir: Path = Field(
        default=Path("data/validation"),
        description="Directory containing subdirectories of validation captures (including unknown/).",
    )
    embeddings_path: Path = Field(
        default=Path("data/reference_embeddings.npz"),
        description="Path to precomputed reference embeddings in safe NPZ format.",
    )
    metadata_path: Path = Field(
        default=Path("data/reference_embeddings.meta.json"),
        description="Path to JSON metadata companion for precomputed embeddings.",
    )


class CampusHuntConfig(BaseModel):
    """Unified configuration for the campus hunt verification pipeline."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    image_limits: ImageLimitsConfig = Field(default_factory=ImageLimitsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def from_env(cls) -> CampusHuntConfig:
        """Create configuration with environment variable overrides.

        Supported environment variables:
          - CAMPUS_HUNT_MODEL_NAME (str)
          - CAMPUS_HUNT_PRETRAINED (str)
          - CAMPUS_HUNT_DEVICE (str)
          - CAMPUS_HUNT_TOP_K (int)
          - CAMPUS_HUNT_CALIBRATED (bool: "1", "true", "yes")
          - CAMPUS_HUNT_MATCH_THRESHOLD (float)
          - CAMPUS_HUNT_MARGIN_THRESHOLD (float)
          - CAMPUS_HUNT_MISMATCH_THRESHOLD (float)
          - CAMPUS_HUNT_EMBEDDINGS_PATH (path)
          - CAMPUS_HUNT_METADATA_PATH (path)
        """
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
        if "CAMPUS_HUNT_EMBEDDINGS_PATH" in os.environ:
            cfg.storage.embeddings_path = Path(os.environ["CAMPUS_HUNT_EMBEDDINGS_PATH"])
        if "CAMPUS_HUNT_METADATA_PATH" in os.environ:
            cfg.storage.metadata_path = Path(os.environ["CAMPUS_HUNT_METADATA_PATH"])

        return cfg

    @classmethod
    def from_json_file(cls, path: str | Path) -> CampusHuntConfig:
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json_file(self, path: str | Path) -> None:
        """Persist configuration to a formatted JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

