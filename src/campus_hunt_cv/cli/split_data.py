"""CLI tool to automatically and safely partition campus images into reference, calibration, and test sets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from campus_hunt_cv.config import (
    DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
)
from campus_hunt_cv.data_splitter import (
    apply_data_split,
    format_split_plan_report,
    plan_data_split,
)
from campus_hunt_cv.exceptions import DataSplitError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("split_data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely split campus location imagery into reference, calibration, and test datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/reference"),
        help="Source directory containing raw campus location folders",
    )
    parser.add_argument(
        "--unknown-location-ids",
        nargs="+",
        default=None,
        help="Explicit list of location IDs to designate as non-hunt checkpoints for the unknown/ pool",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/reference"),
        help="Destination directory for active reference photos",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("data/calibration"),
        help="Destination directory for calibration photos",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("data/test"),
        help="Destination directory for hold-out test photos",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path("data/backup/reference_original"),
        help="Backup destination to preserve pristine original photos outside audited splits",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/split_manifest.json"),
        help="Path for saving structured JSON split manifest",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible, deterministic image assignment",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
        help="Number of images per location to allocate for calibration",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
        help="Number of images per location to allocate for hold-out testing",
    )
    parser.add_argument(
        "--calib-unknown",
        type=int,
        default=DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
        help="Number of unknown captures to allocate for calibration",
    )
    parser.add_argument(
        "--test-unknown",
        type=int,
        default=DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
        help="Number of unknown captures to allocate for hold-out testing",
    )
    parser.add_argument(
        "--min-reference-images",
        type=int,
        default=1,
        help="Minimum remaining images required in reference for each location",
    )
    parser.add_argument(
        "--allow-missing-unknown",
        action="store_true",
        default=False,
        help="Permit provisional split without unknown/distractor locations. All locations remain active checkpoints.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Apply filesystem changes (default is dry-run mode without modifying files)",
    )

    args = parser.parse_args()

    try:
        plan = plan_data_split(
            source_dir=args.source_dir,
            unknown_location_ids=args.unknown_location_ids,
            reference_dir=args.reference_dir,
            calibration_dir=args.calibration_dir,
            test_dir=args.test_dir,
            backup_dir=args.backup_dir,
            manifest_path=args.manifest_path,
            seed=args.seed,
            calib_samples_per_location=args.calib_samples,
            test_samples_per_location=args.test_samples,
            calib_unknown_samples=args.calib_unknown,
            test_unknown_samples=args.test_unknown,
            min_reference_images=args.min_reference_images,
            allow_missing_unknown=args.allow_missing_unknown,
        )
    except Exception as e:
        logger.error("Failed to plan data split: %s", str(e))
        sys.exit(1)

    if not args.apply:
        print(format_split_plan_report(plan, is_apply=False))
        if not plan.adequacy_passed:
            sys.exit(1)
        sys.exit(0)

    # If --apply was requested
    if not plan.adequacy_passed:
        print(format_split_plan_report(plan, is_apply=False))
        logger.error("Cannot apply split because adequacy checks failed. See report above.")
        sys.exit(1)

    try:
        apply_data_split(plan)
        print(format_split_plan_report(plan, is_apply=True))
        sys.exit(0)
    except DataSplitError as e:
        logger.error("Data split application failed: %s", e.message)
        sys.exit(1)
    except Exception as e:
        logger.error("Unexpected error during split execution: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

