"""Developer CLI tool to verify a participant-submitted image against a target checkpoint.

Calls the existing LocationVerifier and outputs structured verification results as JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Sequence, Union

from campus_hunt_cv.config import CampusHuntConfig
from campus_hunt_cv.verifier import LocationVerifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("campus_hunt_verify")


def verify_image_file(
    image_path: Union[str, Path],
    target_location_id: str,
    config: Optional[CampusHuntConfig] = None,
    verifier: Optional[LocationVerifier] = None,
) -> Dict[str, Any]:
    """Load image bytes, execute LocationVerifier, and return the structured response dictionary.

    Args:
        image_path: Path to the image file to verify.
        target_location_id: Server-authoritative target checkpoint location ID.
        config: Optional CampusHuntConfig instance.
        verifier: Optional pre-initialized LocationVerifier instance.

    Returns:
        Dictionary conforming to VerificationResponse schema.
    """
    image_path = Path(image_path)
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bytes = image_path.read_bytes()

    if verifier is None:
        verifier = LocationVerifier(config=config)

    return verifier.verify(
        image_bytes=image_bytes,
        target_location_id=target_location_id,
    )


def main(argv: Optional[Sequence[str]] = None, verifier: Optional[LocationVerifier] = None) -> int:
    """CLI entrypoint for campus-hunt-verify."""
    parser = argparse.ArgumentParser(
        prog="campus-hunt-verify",
        description=(
            "Developer-only CLI command to verify an image file against a target checkpoint "
            "using the existing LocationVerifier and output the structured result as JSON."
        ),
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        type=Path,
        default=None,
        help="Path to the image file to verify",
    )
    parser.add_argument(
        "target_location_id",
        nargs="?",
        type=str,
        default=None,
        help="Server-authoritative target checkpoint location ID",
    )
    parser.add_argument(
        "--image",
        "--image-path",
        dest="image_flag",
        type=Path,
        default=None,
        help="Alternative flag for image file path",
    )
    parser.add_argument(
        "--target",
        "--target-location-id",
        dest="target_flag",
        type=str,
        default=None,
        help="Alternative flag for target checkpoint location ID",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional path to CampusHuntConfig JSON file "
            "(default: calibrated_thresholds.json if present, else environment defaults)"
        ),
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        default=None,
        help="Optional path to reference embeddings .npz file (overrides config storage)",
    )
    parser.add_argument(
        "--embeddings-metadata",
        type=Path,
        default=None,
        help="Optional path to reference embeddings metadata .json file (overrides config storage)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override for embedding model inference (e.g. cpu, cuda)",
    )

    args = parser.parse_args(argv)

    image_path = args.image_path or args.image_flag
    target_id = args.target_location_id or args.target_flag

    if image_path is None or target_id is None:
        parser.error("Both image_path and target_location_id are required.")

    # Resolve configuration
    if args.config is not None and args.config.exists():
        cfg = CampusHuntConfig.from_json_file(args.config)
    elif Path("calibrated_thresholds.json").exists():
        cfg = CampusHuntConfig.from_json_file(Path("calibrated_thresholds.json"))
    else:
        cfg = CampusHuntConfig.from_env()

    # Apply command-line overrides
    if args.embeddings_npz is not None:
        cfg.storage.embeddings_path = args.embeddings_npz
    if args.embeddings_metadata is not None:
        cfg.storage.metadata_path = args.embeddings_metadata
    if args.device is not None:
        cfg.model.device = args.device

    try:
        result = verify_image_file(
            image_path=image_path,
            target_location_id=target_id,
            config=cfg,
            verifier=verifier,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        error_payload = {
            "result": "ambiguous",
            "target_score": None,
            "best_matching_location_id": None,
            "best_matching_score": None,
            "best_competing_location_id": None,
            "best_competing_score": None,
            "margin": None,
            "reason_code": "ERROR_FILE_NOT_FOUND",
            "model_version": getattr(cfg.model, "model_name", "unknown"),
        }
        print(json.dumps(error_payload, indent=2))
        return 1
    except Exception as e:
        logger.error("Verification execution failed: %s", str(e))
        error_payload = {
            "result": "ambiguous",
            "target_score": None,
            "best_matching_location_id": None,
            "best_matching_score": None,
            "best_competing_location_id": None,
            "best_competing_score": None,
            "margin": None,
            "reason_code": "ERROR_UNEXPECTED_FAILURE",
            "model_version": getattr(cfg.model, "model_name", "unknown"),
        }
        print(json.dumps(error_payload, indent=2))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

