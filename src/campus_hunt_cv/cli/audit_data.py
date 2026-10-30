"""Command-line interface for campus hunt dataset audit."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

from campus_hunt_cv.config import CampusHuntConfig
from campus_hunt_cv.data_audit import audit_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audit_data")


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint for campus-hunt-audit-data."""
    parser = argparse.ArgumentParser(
        description=(
            "Audit Campus Hunt CV data splits (reference, calibration, test) for structural integrity, "
            "per-location sample counts, format limits, and cross-split duplicate file contents."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to unified CampusHuntConfig JSON file for defaults",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=None,
        help="Directory containing reference photos per location (default: data/reference)",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Directory containing calibration captures per location and unknown/ (default: data/calibration)",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Backwards-compatibility alias for --calibration-dir",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Directory containing hold-out test captures per location and unknown/ (default: data/test)",
    )
    parser.add_argument(
        "--min-reference-images",
        type=int,
        default=None,
        help="Minimum required reference photos per location (default from config: 1)",
    )
    parser.add_argument(
        "--min-calib-samples",
        type=int,
        default=None,
        help="Minimum calibration captures per location (default from config: 5)",
    )
    parser.add_argument(
        "--min-calib-unknown",
        type=int,
        default=None,
        help="Minimum unknown distractor captures in calibration split (default from config: 10)",
    )
    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=None,
        help="Minimum hold-out test captures per location (default from config: 5)",
    )
    parser.add_argument(
        "--min-test-unknown",
        type=int,
        default=None,
        help="Minimum unknown distractor captures in test split (default from config: 10)",
    )
    parser.add_argument(
        "--output-report",
        "--json-report",
        dest="output_report",
        type=Path,
        default=None,
        help="Optional path to write structured JSON audit report",
    )

    args = parser.parse_args(argv)

    if args.config is not None and args.config.exists():
        cfg = CampusHuntConfig.from_json_file(args.config)
    else:
        cfg = CampusHuntConfig.from_env()

    calib_dir = args.calibration_dir or args.validation_dir

    report = audit_dataset(
        reference_dir=args.reference_dir,
        calibration_dir=calib_dir,
        test_dir=args.test_dir,
        config=cfg,
        min_reference_images_per_location=args.min_reference_images,
        min_calibration_samples_per_location=args.min_calib_samples,
        min_calibration_unknown_samples=args.min_calib_unknown,
        min_test_samples_per_location=args.min_test_samples,
        min_test_unknown_samples=args.min_test_unknown,
    )

    # Print human-readable summary
    print(report.format_text_summary())

    # Write JSON report if requested
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report.to_json(), encoding="utf-8")
        logger.info("Structured JSON audit report written to: %s", args.output_report)

    return 0 if report.audit_passed else 1


if __name__ == "__main__":
    sys.exit(main())

