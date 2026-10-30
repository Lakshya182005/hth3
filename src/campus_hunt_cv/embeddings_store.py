"""Safe, versioned storage for precomputed reference embeddings."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from campus_hunt_cv.exceptions import ReferenceDataError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


class ReferenceEmbeddingsStore:
    """Manages loading and querying precomputed location reference embeddings.

    Uses a secure storage contract:
    - NumPy NPZ format with `allow_pickle=False` to prevent arbitrary code execution vulnerabilities.
    - Accompanying JSON metadata file specifying schema version, model architecture,
      feature dimensions, L2 normalization status, and per-location image counts.
    """

    def __init__(
        self,
        embeddings_by_location: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
    ):
        """Initialize the store with verified embeddings and metadata.

        Args:
            embeddings_by_location: Mapping from location_id to (N, D) float32 L2-normalized array.
            metadata: Metadata dictionary containing model, timestamp, and versioning info.
        """
        self._embeddings_by_location = embeddings_by_location
        self._metadata = metadata

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata associated with the stored embeddings."""
        return self._metadata

    @property
    def model_version(self) -> str:
        """Model version associated with stored embeddings."""
        model_name = self._metadata.get("model_name", "unknown")
        pretrained = self._metadata.get("pretrained", "unknown")
        return f"{model_name}/{pretrained}"

    @property
    def locations(self) -> List[str]:
        """Alphabetically sorted list of all configured hunt locations."""
        return sorted(list(self._embeddings_by_location.keys()))

    def has_location(self, location_id: str) -> bool:
        """Check whether a location ID exists in the reference dataset."""
        return location_id in self._embeddings_by_location

    def get_location_embeddings(self, location_id: str) -> Optional[np.ndarray]:
        """Retrieve precomputed embeddings for a given location.

        Returns:
            (N, D) numpy array of reference embeddings, or None if location is unknown.
        """
        return self._embeddings_by_location.get(location_id)

    def get_location_count(self, location_id: str) -> int:
        """Return the number of reference photos available for a location."""
        emb = self.get_location_embeddings(location_id)
        return 0 if emb is None else emb.shape[0]

    @property
    def all_embeddings_by_location(self) -> Dict[str, np.ndarray]:
        """Copy of full location-to-embeddings mapping."""
        return dict(self._embeddings_by_location)

    @classmethod
    def save(
        cls,
        npz_path: Union[str, Path],
        metadata_path: Union[str, Path],
        embeddings_by_location: Dict[str, np.ndarray],
        model_name: str,
        pretrained: str,
        per_location_files: Optional[Dict[str, List[str]]] = None,
    ) -> ReferenceEmbeddingsStore:
        """Save reference embeddings and metadata to disk using safe NPZ and JSON formats.

        Args:
            npz_path: Destination path for compressed NPZ file.
            metadata_path: Destination path for JSON companion file.
            embeddings_by_location: Dict of location_id -> (N, D) float32 embeddings.
            model_name: OpenCLIP model name (e.g. ViT-B-32).
            pretrained: Weights tag (e.g. laion2b_s34b_b79k).
            per_location_files: Optional mapping of location_id -> list of image filenames.

        Returns:
            Initialized ReferenceEmbeddingsStore instance.
        """
        npz_path = Path(npz_path)
        metadata_path = Path(metadata_path)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        if not embeddings_by_location:
            raise ReferenceDataError(
                "Cannot save empty reference embeddings. At least one location is required.",
                reason_code="ERROR_EMPTY_REFERENCE_SET",
            )

        # Collect stacked arrays and metadata
        all_embeddings: List[np.ndarray] = []
        all_labels: List[str] = []
        location_meta: Dict[str, Any] = {}
        feature_dim: Optional[int] = None

        for loc_id in sorted(embeddings_by_location.keys()):
            emb = embeddings_by_location[loc_id]
            if not isinstance(emb, np.ndarray):
                raise ReferenceDataError(f"Embeddings for {loc_id} must be a numpy ndarray.")
            if emb.ndim != 2:
                raise ReferenceDataError(
                    f"Embeddings for {loc_id} must be 2D array of shape (N, D), got {emb.shape}."
                )
            if feature_dim is None:
                feature_dim = emb.shape[1]
            elif emb.shape[1] != feature_dim:
                raise ReferenceDataError(
                    f"Feature dimension mismatch for {loc_id}: expected {feature_dim}, got {emb.shape[1]}."
                )

            count = emb.shape[0]
            for _ in range(count):
                all_labels.append(loc_id)
            all_embeddings.append(emb.astype(np.float32))

            files = (per_location_files or {}).get(loc_id, [])
            location_meta[loc_id] = {
                "count": count,
                "files": files,
            }

        stacked_matrix = np.vstack(all_embeddings)
        labels_array = np.array(all_labels, dtype=str)

        # Save NPZ compressed without pickle
        np.savez_compressed(
            npz_path,
            embeddings=stacked_matrix,
            locations=labels_array,
        )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "pretrained": pretrained,
            "feature_dimension": feature_dim,
            "normalized": True,
            "total_images": int(stacked_matrix.shape[0]),
            "total_locations": len(embeddings_by_location),
            "locations": location_meta,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Saved %d reference embeddings across %d locations to '%s' and '%s'.",
            metadata["total_images"],
            metadata["total_locations"],
            npz_path,
            metadata_path,
        )

        return cls(embeddings_by_location=embeddings_by_location, metadata=metadata)

    @classmethod
    def load(
        cls,
        npz_path: Union[str, Path],
        metadata_path: Union[str, Path],
        expected_model_version: Optional[str] = None,
    ) -> ReferenceEmbeddingsStore:
        """Load and validate reference embeddings from NPZ and JSON files.

        Strictly enforces `allow_pickle=False` for security.

        Args:
            npz_path: Path to .npz file.
            metadata_path: Path to companion .meta.json file.
            expected_model_version: Optional string to verify model compatibility.

        Returns:
            Initialized ReferenceEmbeddingsStore.

        Raises:
            ReferenceDataError: If files are missing, corrupt, incompatible, or invalid.
        """
        npz_path = Path(npz_path)
        metadata_path = Path(metadata_path)

        if not npz_path.exists():
            raise ReferenceDataError(
                f"Reference embeddings file not found at '{npz_path}'. "
                "Run `campus-hunt-build-embeddings` to compute embeddings from `data/reference/`.",
                reason_code="ERROR_REFERENCE_EMBEDDINGS_NOT_FOUND",
            )

        if not metadata_path.exists():
            raise ReferenceDataError(
                f"Reference embeddings metadata not found at '{metadata_path}'.",
                reason_code="ERROR_REFERENCE_METADATA_NOT_FOUND",
            )

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            raise ReferenceDataError(
                f"Failed to read metadata JSON from '{metadata_path}': {str(e)}",
                reason_code="ERROR_METADATA_CORRUPT",
            ) from e

        schema_ver = metadata.get("schema_version")
        if schema_ver != SCHEMA_VERSION:
            logger.warning(
                "Metadata schema version '%s' does not match current version '%s'.",
                schema_ver,
                SCHEMA_VERSION,
            )

        # Enforce allow_pickle=False for security against arbitrary code execution
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                if "embeddings" not in npz or "locations" not in npz:
                    raise ReferenceDataError(
                        "NPZ file missing required keys 'embeddings' or 'locations'.",
                        reason_code="ERROR_NPZ_INVALID_SCHEMA",
                    )
                embeddings = npz["embeddings"]
                locations = npz["locations"]
        except Exception as e:
            if isinstance(e, ReferenceDataError):
                raise
            raise ReferenceDataError(
                f"Failed to safely load embeddings from '{npz_path}' (allow_pickle=False): {str(e)}",
                reason_code="ERROR_NPZ_LOAD_FAILED",
            ) from e

        if embeddings.ndim != 2:
            raise ReferenceDataError(
                f"Expected 2D embeddings matrix, got {embeddings.shape}.",
                reason_code="ERROR_INVALID_EMBEDDING_SHAPE",
            )

        if embeddings.shape[0] != locations.shape[0]:
            raise ReferenceDataError(
                f"Embeddings count ({embeddings.shape[0]}) does not match labels count ({locations.shape[0]}).",
                reason_code="ERROR_EMBEDDING_LABEL_MISMATCH",
            )

        # Optional model version check
        stored_version = f"{metadata.get('model_name')}/{metadata.get('pretrained')}"
        if expected_model_version and stored_version != expected_model_version:
            logger.warning(
                "Reference embeddings were generated with model '%s', but current model is '%s'. "
                "Embeddings should be regenerated with the matching model checkpoint.",
                stored_version,
                expected_model_version,
            )

        # Group embeddings by location_id
        embeddings_by_loc: Dict[str, List[np.ndarray]] = {}
        for idx in range(locations.shape[0]):
            loc = str(locations[idx])
            if loc not in embeddings_by_loc:
                embeddings_by_loc[loc] = []
            embeddings_by_loc[loc].append(embeddings[idx])

        final_mapping: Dict[str, np.ndarray] = {
            loc: np.vstack(vecs).astype(np.float32) for loc, vecs in embeddings_by_loc.items()
        }

        return cls(embeddings_by_location=final_mapping, metadata=metadata)

