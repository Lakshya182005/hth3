"""CLI tool to build and save reference embeddings from the data directory layout."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Dict, List
import numpy as np

from campus_hunt_cv.config import CampusHuntConfig, ModelConfig
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.image_validation import validate_and_decode_image
from campus_hunt_cv.model import OpenCLIPLocationEmbedder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_embeddings")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}


def build_reference_embeddings(
    reference_dir: Path,
    output_npz: Path,
    output_meta: Path,
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str = "cpu",
    num_threads: int | None = None,
) -> bool:
    """Scan reference directory, extract normalized embeddings, and save NPZ/JSON files."""
    reference_dir = Path(reference_dir)
    if not reference_dir.exists():
        logger.error("Reference directory does not exist: %s", reference_dir)
        logger.info(
            "Please create location folders inside '%s', e.g.:\n"
            "  %s/clock_tower/photo1.jpg\n"
            "  %s/library_atrium/photo1.jpg",
            reference_dir,
            reference_dir,
            reference_dir,
        )
        return False

    location_dirs = [d for d in reference_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not location_dirs:
        logger.warning(
            "No location subdirectories found in '%s'.\n"
            "Contract layout:\n"
            "  %s/<location_id>/*.jpg\n"
            "Please add reference photos before building embeddings.",
            reference_dir,
            reference_dir,
        )
        return False

    logger.info("Found %d candidate location directories.", len(location_dirs))

    model_config = ModelConfig(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        num_threads=num_threads,
    )
    embedder = OpenCLIPLocationEmbedder(config=model_config)

    embeddings_by_location: Dict[str, np.ndarray] = {}
    files_by_location: Dict[str, List[str]] = {}
    total_images_processed = 0

    for loc_dir in sorted(location_dirs):
        loc_id = loc_dir.name
        image_files = [f for f in loc_dir.iterdir() if f.suffix in SUPPORTED_EXTENSIONS]

        if not image_files:
            logger.warning("Location '%s' has 0 images. Skipping.", loc_id)
            continue

        logger.info("Processing location '%s' (%d images)...", loc_id, len(image_files))
        loc_embeddings: List[np.ndarray] = []
        valid_files: List[str] = []

        for img_path in sorted(image_files):
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()
                image = validate_and_decode_image(image_bytes)
                emb = embedder.embed_image(image)
                loc_embeddings.append(emb)
                valid_files.append(img_path.name)
                total_images_processed += 1
            except Exception as e:
                logger.error("Error processing '%s': %s", img_path, str(e))

        if loc_embeddings:
            embeddings_by_location[loc_id] = np.vstack(loc_embeddings).astype(np.float32)
            files_by_location[loc_id] = valid_files

    if not embeddings_by_location:
        logger.error("No valid reference images could be processed. No embeddings saved.")
        return False

    ReferenceEmbeddingsStore.save(
        npz_path=output_npz,
        metadata_path=output_meta,
        embeddings_by_location=embeddings_by_location,
        model_name=model_name,
        pretrained=pretrained,
        per_location_files=files_by_location,
    )

    logger.info(
        "Successfully saved reference embeddings for %d locations (%d total images).",
        len(embeddings_by_location),
        total_images_processed,
    )
    return True


def main() -> None:
    """CLI entrypoint."""
    cfg = CampusHuntConfig.from_env()

    parser = argparse.ArgumentParser(
        description="Extract and save reference embeddings for all campus hunt locations."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=cfg.storage.reference_data_dir,
        help="Path to directory containing reference subdirectories (default: data/reference)",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=cfg.storage.embeddings_path,
        help="Output path for NPZ file (default: data/reference_embeddings.npz)",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=cfg.storage.metadata_path,
        help="Output path for JSON metadata file (default: data/reference_embeddings.meta.json)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=cfg.model.model_name,
        help="OpenCLIP model name (default: ViT-B-32)",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=cfg.model.pretrained,
        help="Pretrained weights tag (default: laion2b_s34b_b79k)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.model.device,
        help="Compute device: 'cpu', 'cuda', 'mps', or 'auto' (default: cpu)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=cfg.model.num_threads,
        help="PyTorch CPU thread count (optional)",
    )

    args = parser.parse_args()

    success = build_reference_embeddings(
        reference_dir=args.reference_dir,
        output_npz=args.output_npz,
        output_meta=args.output_meta,
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=args.device,
        num_threads=args.num_threads,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()

