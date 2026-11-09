"""Dataset audit module for the Campus Hunt CV verification pipeline.

Performs offline structural, format, limit, count, and duplicate audits across
reference, calibration, and test splits without executing model inference
or calculating accuracy metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from campus_hunt_cv.config import (
    CampusHuntConfig,
    DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
    ImageLimitsConfig,
)
from campus_hunt_cv.exceptions import ImageValidationError
from campus_hunt_cv.image_validation import validate_and_decode_image

logger = logging.getLogger("data_audit")

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class FileOccurrence:
    """Provenance tracking for a scanned image file."""

    split: str
    location: str
    path: str
    sha256: str


@dataclass
class InvalidImageRecord:
    """Record of an image that failed validation or decoding."""

    filename: str
    path: str
    reason: str
    reason_code: str


@dataclass
class LocationAuditDetail:
    """Detailed audit metrics for a single location category within a split."""

    location_id: str
    valid_count: int = 0
    required_min: int = 0
    sufficient: bool = False
    valid_files: List[str] = field(default_factory=list)
    invalid_images: List[Dict[str, str]] = field(default_factory=list)
    unsupported_files: List[str] = field(default_factory=list)
    unexpected_subdirectories: List[str] = field(default_factory=list)


@dataclass
class SplitAuditResult:
    """Audit outcome for a single dataset split (reference, calibration, or test)."""

    split_name: str
    dir_path: str
    exists: bool = False
    locations: Dict[str, LocationAuditDetail] = field(default_factory=dict)
    missing_locations: List[str] = field(default_factory=list)
    insufficient_locations: List[str] = field(default_factory=list)
    unexpected_subdirectories: List[str] = field(default_factory=list)
    empty_locations: List[str] = field(default_factory=list)
    invalid_images_count: int = 0
    unsupported_files_count: int = 0


@dataclass
class DuplicateGroup:
    """A set of file paths sharing identical SHA-256 content."""

    sha256: str
    is_cross_split: bool
    files: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class DatasetAuditReport:
    """Complete report of dataset integrity and split consistency audit."""

    timestamp: str
    audit_passed: bool
    summary: Dict[str, Any]
    configuration: Dict[str, Any]
    splits: Dict[str, SplitAuditResult]
    duplicates: Dict[str, List[DuplicateGroup]]
    issues: List[str]
    allow_missing_unknown: bool = False
    provisional_no_unknown: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert audit report to a serializable dictionary."""
        return {
            "timestamp": self.timestamp,
            "audit_passed": self.audit_passed,
            "allow_missing_unknown": self.allow_missing_unknown,
            "provisional_no_unknown": self.provisional_no_unknown,
            "status": (
                "STRUCTURALLY_SPLIT_UNKNOWN_UNAVAILABLE"
                if self.provisional_no_unknown
                else ("PASSED" if self.audit_passed else "FAILED")
            ),
            "fully_validated": False if self.provisional_no_unknown else self.audit_passed,
            "unknown_evaluation": "unavailable" if self.provisional_no_unknown else "available",
            "summary": self.summary,
            "configuration": self.configuration,
            "splits": {name: asdict(split) for name, split in self.splits.items()},
            "duplicates": {
                category: [asdict(group) for group in groups]
                for category, groups in self.duplicates.items()
            },
            "issues": list(self.issues),
            "disclaimer": (
                "This audit evaluates dataset integrity, format compliance, sample counts, "
                "and split isolation. It does NOT evaluate model accuracy or claim model validation."
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        """Render report as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def format_text_summary(self) -> str:
        """Render human-readable multi-line console report."""
        lines: List[str] = []
        divider = "=" * 70
        subdivider = "-" * 70

        lines.append(divider)
        lines.append("CAMPUS HUNT DATASET INTEGRITY AUDIT")
        lines.append(divider)
        lines.append(f"Audit Timestamp: {self.timestamp}")
        if self.provisional_no_unknown:
            status_text = "STRUCTURALLY SPLIT (UNKNOWN EVALUATION UNAVAILABLE - PROVISIONAL)"
        else:
            status_text = "PASSED" if self.audit_passed else "FAILED"
        lines.append(f"Overall Status:  {status_text}")
        if self.provisional_no_unknown:
            lines.append("Notice:          Dataset is structurally partitioned across all checkpoint locations,")
            lines.append("                 but unknown/ distractor folders are absent. False Acceptance Rate (FAR)")
            lines.append("                 and unknown rejection cannot be evaluated. This split is NOT fully")
            lines.append("                 validated and must remain provisional.")
        lines.append(f"Total Issues:    {len(self.issues)}")
        lines.append(subdivider)

        lines.append("\nSPLIT BREAKDOWN:")
        for split_name in ("reference", "calibration", "test"):
            if split_name not in self.splits:
                continue
            split_res = self.splits[split_name]
            lines.append(f"\n[{split_name.upper()}] Directory: {split_res.dir_path}")
            if not split_res.exists:
                lines.append("  Status: DIRECTORY MISSING")
                continue

            lines.append(f"  Locations found: {len(split_res.locations)}")
            for loc_id, loc_detail in sorted(split_res.locations.items()):
                flag = "OK" if loc_detail.sufficient else "INSUFFICIENT"
                lines.append(
                    f"    - {loc_id:20s}: {loc_detail.valid_count:3d} valid images "
                    f"(required min: {loc_detail.required_min:2d}) -> [{flag}]"
                )
                if loc_detail.invalid_images:
                    lines.append(f"        * Invalid images: {len(loc_detail.invalid_images)}")
                if loc_detail.unsupported_files:
                    lines.append(f"        * Unsupported files: {len(loc_detail.unsupported_files)}")
                if loc_detail.unexpected_subdirectories:
                    lines.append(f"        * Unexpected subdirs: {loc_detail.unexpected_subdirectories}")

            if split_res.missing_locations:
                lines.append(f"  Missing locations: {', '.join(split_res.missing_locations)}")
            if split_res.unexpected_subdirectories:
                lines.append(f"  Unexpected split-level subdirs: {', '.join(split_res.unexpected_subdirectories)}")
            if split_res.empty_locations:
                lines.append(f"  Empty location folders: {', '.join(split_res.empty_locations)}")

        cross_dups = self.duplicates.get("cross_split_duplicates", [])
        within_dups = self.duplicates.get("within_split_duplicates", [])
        lines.append(f"\nDUPLICATE CONTENT CHECK (SHA-256):")
        lines.append(f"  Cross-split duplicate groups:  {len(cross_dups)}")
        lines.append(f"  Within-split duplicate groups: {len(within_dups)}")

        if cross_dups:
            lines.append("\n  [!] CRITICAL CROSS-SPLIT DUPLICATES DETECTED:")
            for grp in cross_dups:
                lines.append(f"    - SHA-256: {grp.sha256[:16]}...")
                for f_info in grp.files:
                    lines.append(f"        {f_info['split']:12s} | {f_info['path']}")

        if self.issues:
            lines.append("\n" + subdivider)
            lines.append(f"DETECTED PROBLEMS ({len(self.issues)}):")
            for idx, issue in enumerate(self.issues, 1):
                lines.append(f"  {idx:2d}. {issue}")

        lines.append("\n" + divider)
        lines.append(
            "DISCLAIMER: This audit evaluates dataset structure, format compliance, "
            "and split isolation.\nIt does NOT evaluate model accuracy or claim model validation."
        )
        lines.append(divider)
        return "\n".join(lines)


def audit_dataset(
    reference_dir: Optional[Path] = None,
    calibration_dir: Optional[Path] = None,
    test_dir: Optional[Path] = None,
    config: Optional[CampusHuntConfig] = None,
    min_reference_images_per_location: Optional[int] = None,
    min_calibration_samples_per_location: Optional[int] = None,
    min_calibration_unknown_samples: Optional[int] = None,
    min_test_samples_per_location: Optional[int] = None,
    min_test_unknown_samples: Optional[int] = None,
    image_limits: Optional[ImageLimitsConfig] = None,
    allow_missing_unknown: bool = False,
) -> DatasetAuditReport:
    """Execute offline data audit across reference, calibration, and test splits.

    Validates:
    1. Reference locations contain supported images meeting reference minimums.
    2. Calibration and test splits contain identical locations to reference, plus 'unknown/'.
    3. Each split meets configured per-location and unknown-image minimums.
    4. Images decode cleanly within size, dimension, and format limits.
    5. No identical image contents (SHA-256) are reused across splits or across categories.
    6. Unexpected subdirectories, empty location folders, and unsupported files are flagged.

    Args:
        reference_dir: Path to reference photos split.
        calibration_dir: Path to calibration captures split.
        test_dir: Path to hold-out test captures split.
        config: Unified configuration instance (default: CampusHuntConfig.from_env()).
        min_reference_images_per_location: Override for reference minimum count.
        min_calibration_samples_per_location: Override for calibration location minimum count.
        min_calibration_unknown_samples: Override for calibration unknown minimum count.
        min_test_samples_per_location: Override for test location minimum count.
        min_test_unknown_samples: Override for test unknown minimum count.
        image_limits: Custom image limits (defaults to config.image_limits).

    Returns:
        DatasetAuditReport containing structured results and issues list.
    """
    cfg = config or CampusHuntConfig.from_env()

    # Resolve paths
    ref_dir = Path(reference_dir or cfg.storage.reference_data_dir)
    cal_dir = Path(calibration_dir or cfg.storage.calibration_data_dir)
    tst_dir = Path(test_dir or cfg.storage.test_data_dir)

    # Resolve minimum count thresholds
    eff_min_ref = (
        min_reference_images_per_location
        if min_reference_images_per_location is not None
        else cfg.scoring.min_reference_images_per_location
    )
    eff_min_cal_loc = (
        min_calibration_samples_per_location
        if min_calibration_samples_per_location is not None
        else cfg.scoring.min_calibration_samples_per_location
    )
    eff_min_cal_unk = (
        min_calibration_unknown_samples
        if min_calibration_unknown_samples is not None
        else cfg.scoring.min_calibration_unknown_samples
    )
    eff_min_tst_loc = (
        min_test_samples_per_location
        if min_test_samples_per_location is not None
        else cfg.scoring.min_test_samples_per_location
    )
    eff_min_tst_unk = (
        min_test_unknown_samples
        if min_test_unknown_samples is not None
        else cfg.scoring.min_test_unknown_samples
    )

    limits = image_limits or cfg.image_limits

    issues: List[str] = []
    hash_occurrences: Dict[str, List[FileOccurrence]] = {}

    splits_data: Dict[str, SplitAuditResult] = {}
    total_scanned = 0
    total_valid = 0
    total_invalid = 0
    total_unsupported = 0

    # ---------------------------------------------------------
    # 1. Audit Reference Split
    # ---------------------------------------------------------
    ref_split = SplitAuditResult(split_name="reference", dir_path=str(ref_dir), exists=ref_dir.exists())
    ref_location_ids: List[str] = []

    if not ref_dir.exists():
        issues.append(f"Reference directory does not exist: '{ref_dir}'")
    else:
        # Check subdirectories
        subitems = sorted(list(ref_dir.iterdir()))
        for item in subitems:
            if item.is_dir():
                if item.name.startswith("."):
                    continue
                if item.name == "unknown":
                    issues.append("Reference split contains unexpected 'unknown/' distractor directory.")
                    ref_split.unexpected_subdirectories.append("unknown")
                    continue
                ref_location_ids.append(item.name)
            elif item.is_file():
                # Allow root README.md or .gitignore, flag loose image/data files
                if not item.name.startswith(".") and item.name.lower() not in ("readme.md", "readme.txt", "readme"):
                    issues.append(f"Unexpected loose file in reference root directory: '{item.name}'")
                    ref_split.unsupported_files_count += 1

        if not ref_location_ids:
            issues.append(f"Reference directory '{ref_dir}' contains no location subdirectories.")

        for loc_id in sorted(ref_location_ids):
            loc_dir = ref_dir / loc_id
            detail = LocationAuditDetail(location_id=loc_id, required_min=eff_min_ref)

            loc_items = sorted(list(loc_dir.iterdir()))
            if not loc_items:
                ref_split.empty_locations.append(loc_id)
                issues.append(f"Reference location folder '{loc_id}' is empty.")
                ref_split.locations[loc_id] = detail
                ref_split.insufficient_locations.append(loc_id)
                continue

            for file_path in loc_items:
                if file_path.is_dir():
                    if not file_path.name.startswith("."):
                        detail.unexpected_subdirectories.append(file_path.name)
                        issues.append(
                            f"Unexpected nested subdirectory in reference location '{loc_id}': '{file_path.name}'"
                        )
                    continue

                if file_path.name.startswith("."):
                    continue

                if file_path.name.lower() in ("readme.md", "readme.txt", "readme"):
                    continue

                ext = file_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    detail.unsupported_files.append(file_path.name)
                    ref_split.unsupported_files_count += 1
                    total_unsupported += 1
                    issues.append(
                        f"Unsupported file format in reference location '{loc_id}': '{file_path.name}'"
                    )
                    continue

                total_scanned += 1
                try:
                    data = file_path.read_bytes()
                except Exception as e:
                    detail.invalid_images.append({
                        "file": file_path.name,
                        "path": str(file_path),
                        "reason": f"File read error: {str(e)}",
                        "reason_code": "ERROR_FILE_READ",
                    })
                    ref_split.invalid_images_count += 1
                    total_invalid += 1
                    issues.append(f"Cannot read image file in reference '{loc_id}/{file_path.name}': {str(e)}")
                    continue

                # SHA-256 tracking
                sha = hashlib.sha256(data).hexdigest()
                occ = FileOccurrence(
                    split="reference",
                    location=loc_id,
                    path=str(file_path),
                    sha256=sha,
                )
                hash_occurrences.setdefault(sha, []).append(occ)

                # Safe validation and decode
                try:
                    validate_and_decode_image(data, limits=limits)
                    detail.valid_count += 1
                    detail.valid_files.append(file_path.name)
                    total_valid += 1
                except ImageValidationError as e:
                    detail.invalid_images.append({
                        "file": file_path.name,
                        "path": str(file_path),
                        "reason": e.message,
                        "reason_code": e.reason_code,
                    })
                    ref_split.invalid_images_count += 1
                    total_invalid += 1
                    issues.append(
                        f"Invalid image in reference location '{loc_id}/{file_path.name}': {e.message}"
                    )
                except Exception as e:
                    detail.invalid_images.append({
                        "file": file_path.name,
                        "path": str(file_path),
                        "reason": f"Unexpected decode failure: {str(e)}",
                        "reason_code": "ERROR_IMAGE_DECODE_FAILED",
                    })
                    ref_split.invalid_images_count += 1
                    total_invalid += 1
                    issues.append(
                        f"Decode failure in reference location '{loc_id}/{file_path.name}': {str(e)}"
                    )

            detail.sufficient = detail.valid_count >= eff_min_ref
            if not detail.sufficient:
                ref_split.insufficient_locations.append(loc_id)
                issues.append(
                    f"Insufficient reference images for '{loc_id}': found {detail.valid_count}, "
                    f"required at least {eff_min_ref}."
                )

            ref_split.locations[loc_id] = detail

    splits_data["reference"] = ref_split

    # ---------------------------------------------------------
    # 2 & 3. Audit Calibration and Test Splits
    # ---------------------------------------------------------
    split_configs = [
        ("calibration", cal_dir, eff_min_cal_loc, eff_min_cal_unk),
        ("test", tst_dir, eff_min_tst_loc, eff_min_tst_unk),
    ]

    for split_name, split_path, min_loc, min_unk in split_configs:
        split_res = SplitAuditResult(split_name=split_name, dir_path=str(split_path), exists=split_path.exists())

        if not split_path.exists():
            issues.append(f"{split_name.capitalize()} directory does not exist: '{split_path}'")
            splits_data[split_name] = split_res
            continue

        split_items = sorted(list(split_path.iterdir()))
        observed_subdirs: Set[str] = set()

        for item in split_items:
            if item.is_dir():
                if not item.name.startswith("."):
                    observed_subdirs.add(item.name)
            elif item.is_file():
                if not item.name.startswith(".") and item.name.lower() not in ("readme.md", "readme.txt", "readme"):
                    issues.append(f"Unexpected loose file in {split_name} root directory: '{item.name}'")
                    split_res.unsupported_files_count += 1

        # Check required location folders against reference
        for ref_loc in ref_location_ids:
            if ref_loc not in observed_subdirs:
                split_res.missing_locations.append(ref_loc)
                issues.append(f"{split_name.capitalize()} split is missing location folder: '{ref_loc}'")

        # Check required unknown/ folder
        if "unknown" not in observed_subdirs:
            if not allow_missing_unknown:
                split_res.missing_locations.append("unknown")
                issues.append(
                    f"{split_name.capitalize()} split is missing required 'unknown/' distractor folder."
                )

        # Check unexpected subdirectories (not in reference and not 'unknown')
        for sdir in sorted(observed_subdirs):
            if sdir != "unknown" and sdir not in ref_location_ids:
                split_res.unexpected_subdirectories.append(sdir)
                issues.append(
                    f"Unexpected location folder in {split_name} split: '{sdir}' (not found in reference split)"
                )

        # Inspect all observed subdirectories
        for loc_id in sorted(observed_subdirs):
            loc_dir = split_path / loc_id
            req_min = min_unk if loc_id == "unknown" else min_loc
            detail = LocationAuditDetail(location_id=loc_id, required_min=req_min)

            loc_items = sorted(list(loc_dir.iterdir()))
            if not loc_items:
                split_res.empty_locations.append(loc_id)
                issues.append(f"{split_name.capitalize()} folder '{loc_id}' is empty.")
                split_res.locations[loc_id] = detail
                split_res.insufficient_locations.append(loc_id)
                continue

            for file_path in loc_items:
                if file_path.is_dir():
                    if not file_path.name.startswith("."):
                        detail.unexpected_subdirectories.append(file_path.name)
                        issues.append(
                            f"Unexpected nested subdirectory in {split_name} '{loc_id}': '{file_path.name}'"
                        )
                    continue

                if file_path.name.startswith("."):
                    continue

                if file_path.name.lower() in ("readme.md", "readme.txt", "readme"):
                    continue

                ext = file_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    detail.unsupported_files.append(file_path.name)
                    split_res.unsupported_files_count += 1
                    total_unsupported += 1
                    issues.append(
                        f"Unsupported file format in {split_name} '{loc_id}': '{file_path.name}'"
                    )
                    continue

                total_scanned += 1
                try:
                    data = file_path.read_bytes()
                except Exception as e:
                    detail.invalid_images.append({
                        "file": file_path.name,
                        "path": str(file_path),
                        "reason": f"File read error: {str(e)}",
                        "reason_code": "ERROR_FILE_READ",
                    })
                    split_res.invalid_images_count += 1
                    total_invalid += 1
                    issues.append(f"Cannot read image file in {split_name} '{loc_id}/{file_path.name}': {str(e)}")
                    continue

                # SHA-256 tracking
                sha = hashlib.sha256(data).hexdigest()
                occ = FileOccurrence(
                    split=split_name,
                    location=loc_id,
                    path=str(file_path),
                    sha256=sha,
                )
                hash_occurrences.setdefault(sha, []).append(occ)

                # Safe validation and decode
                try:
                    validate_and_decode_image(data, limits=limits)
                    detail.valid_count += 1
                    detail.valid_files.append(file_path.name)
                    total_valid += 1
                except ImageValidationError as e:
                    detail.invalid_images.append({
                        "file": file_path.name,
                        "path": str(file_path),
                        "reason": e.message,
                        "reason_code": e.reason_code,
                    })
                    split_res.invalid_images_count += 1
                    total_invalid += 1
                    issues.append(
                        f"Invalid image in {split_name} '{loc_id}/{file_path.name}': {e.message}"
                    )
                except Exception as e:
                    detail.invalid_images.append({
                        "file": file_path.name,
                        "path": str(file_path),
                        "reason": f"Unexpected decode failure: {str(e)}",
                        "reason_code": "ERROR_IMAGE_DECODE_FAILED",
                    })
                    split_res.invalid_images_count += 1
                    total_invalid += 1
                    issues.append(
                        f"Decode failure in {split_name} '{loc_id}/{file_path.name}': {str(e)}"
                    )

            detail.sufficient = detail.valid_count >= req_min
            if not detail.sufficient:
                split_res.insufficient_locations.append(loc_id)
                category_label = "unknown captures" if loc_id == "unknown" else f"captures for '{loc_id}'"
                issues.append(
                    f"Insufficient {category_label} in {split_name} split: "
                    f"found {detail.valid_count}, required at least {req_min}."
                )

            split_res.locations[loc_id] = detail

        splits_data[split_name] = split_res

    # ---------------------------------------------------------
    # 4. Duplicate Content Analysis (SHA-256)
    # ---------------------------------------------------------
    cross_split_duplicates: List[DuplicateGroup] = []
    within_split_duplicates: List[DuplicateGroup] = []

    for sha, occ_list in hash_occurrences.items():
        if len(occ_list) > 1:
            distinct_splits = {o.split for o in occ_list}
            files_info = [{"split": o.split, "location": o.location, "path": o.path} for o in occ_list]

            if len(distinct_splits) > 1:
                cross_split_duplicates.append(
                    DuplicateGroup(sha256=sha, is_cross_split=True, files=files_info)
                )
                splits_str = ", ".join(f"{o.split}:{o.path}" for o in occ_list)
                issues.append(
                    f"Cross-split duplicate image detected (SHA-256: {sha[:16]}...): {splits_str}"
                )
            else:
                within_split_duplicates.append(
                    DuplicateGroup(sha256=sha, is_cross_split=False, files=files_info)
                )
                locs = {o.location for o in occ_list}
                if len(locs) > 1:
                    locs_str = ", ".join(f"{o.location}:{o.path}" for o in occ_list)
                    issues.append(
                        f"Duplicate image reused across different locations in {occ_list[0].split} split "
                        f"(SHA-256: {sha[:16]}...): {locs_str}"
                    )
                else:
                    paths_str = ", ".join(o.path for o in occ_list)
                    issues.append(
                        f"Duplicate image repeated within {occ_list[0].split}/{occ_list[0].location} "
                        f"(SHA-256: {sha[:16]}...): {paths_str}"
                    )

    duplicates_dict = {
        "cross_split_duplicates": cross_split_duplicates,
        "within_split_duplicates": within_split_duplicates,
    }

    audit_passed = len(issues) == 0

    summary = {
        "total_reference_locations": len(ref_location_ids),
        "total_images_scanned": total_scanned,
        "total_valid_images": total_valid,
        "total_invalid_images": total_invalid,
        "total_unsupported_files": total_unsupported,
        "cross_split_duplicate_groups": len(cross_split_duplicates),
        "within_split_duplicate_groups": len(within_split_duplicates),
        "total_issues_count": len(issues),
    }

    configuration = {
        "reference_dir": str(ref_dir),
        "calibration_dir": str(cal_dir),
        "test_dir": str(tst_dir),
        "min_reference_images_per_location": eff_min_ref,
        "min_calibration_samples_per_location": eff_min_cal_loc,
        "min_calibration_unknown_samples": eff_min_cal_unk,
        "min_test_samples_per_location": eff_min_tst_loc,
        "min_test_unknown_samples": eff_min_tst_unk,
        "image_limits": limits.model_dump(),
        "allow_missing_unknown": allow_missing_unknown,
    }

    cal_split = splits_data.get("calibration")
    tst_split = splits_data.get("test")
    has_unknown = (
        cal_split is not None
        and "unknown" in cal_split.locations
        and tst_split is not None
        and "unknown" in tst_split.locations
    )
    provisional_no_unknown = allow_missing_unknown and (not has_unknown)

    return DatasetAuditReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        audit_passed=audit_passed,
        allow_missing_unknown=allow_missing_unknown,
        provisional_no_unknown=provisional_no_unknown,
        summary=summary,
        configuration=configuration,
        splits=splits_data,
        duplicates=duplicates_dict,
        issues=issues,
    )
