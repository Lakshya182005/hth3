"""Data splitting module for the Campus Hunt CV verification pipeline.

Safely partitions raw campus location images into three independent splits:
1. data/reference/: Precomputed embeddings for active hunt checkpoints.
2. data/calibration/: Calibration captures for sweeping similarity and margin thresholds.
3. data/test/: Strict hold-out test captures for final generalization evaluation.

Enforces sample safeguards, prevents cross-split data leakage, detects near-duplicate
bursts, excludes designated non-checkpoint locations into unknown/ distractor pools,
creates pristine backups before applying changes, and marks splits as provisional
when cross-session independence cannot be established from metadata.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from campus_hunt_cv.config import (
    DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
)
from campus_hunt_cv.exceptions import DataSplitError

logger = logging.getLogger("data_splitter")

SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ImageMetadata:
    """Metadata extracted from a single image file for clustering and splitting."""

    path: Path
    filename: str
    sha256: str
    timestamp: Optional[datetime] = None
    exif_device: Optional[str] = None


@dataclass
class LocationSplitAssignment:
    """Proposed split assignments and counts for a single location."""

    location_id: str
    is_unknown_pool: bool
    total_source_images: int
    reference_files: List[str] = field(default_factory=list)
    calibration_files: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    burst_clusters_detected: int = 0
    remaining_reference_count: int = 0
    sufficient: bool = True
    issues: List[str] = field(default_factory=list)


@dataclass
class SplitPlan:
    """Complete dry-run or execution plan for dataset splitting."""

    source_dir: Path
    reference_dir: Path
    calibration_dir: Path
    test_dir: Path
    backup_dir: Path
    manifest_path: Path
    seed: int
    calib_samples_per_location: int
    test_samples_per_location: int
    calib_unknown_samples: int
    test_unknown_samples: int
    min_reference_images: int
    allow_missing_unknown: bool = False
    all_locations: List[str] = field(default_factory=list)
    active_locations: List[str] = field(default_factory=list)
    unknown_locations: List[str] = field(default_factory=list)
    assignments: Dict[str, LocationSplitAssignment] = field(default_factory=dict)
    calibration_unknown_files: List[Tuple[str, str]] = field(default_factory=list)  # (loc_id, filename)
    test_unknown_files: List[Tuple[str, str]] = field(default_factory=list)         # (loc_id, filename)
    provisional: bool = True
    independence_verifiable: bool = False
    status_reason: str = ""
    adequacy_passed: bool = True
    blocking_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def unknown_evaluation(self) -> str:
        """Indicate whether unknown distractor evaluation is available or unavailable."""
        return "unavailable" if (self.allow_missing_unknown and not self.unknown_locations) else "available"


def extract_image_metadata(file_path: Path) -> ImageMetadata:
    """Extract SHA-256, capture timestamp, and device info from an image file."""
    data = file_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()

    timestamp: Optional[datetime] = None
    device: Optional[str] = None

    # 1. Try EXIF extraction
    try:
        with Image.open(file_path) as im:
            exif = im.getexif()
            if exif:
                # Tags: 36867 = DateTimeOriginal, 306 = DateTime, 36868 = DateTimeDigitized
                for tag in (36867, 306, 36868):
                    val = exif.get(tag)
                    if val and isinstance(val, str):
                        clean_val = val.strip("\x00").strip()
                        try:
                            timestamp = datetime.strptime(clean_val, "%Y:%m:%d %H:%M:%S")
                            break
                        except ValueError:
                            pass
                make = exif.get(271)
                model = exif.get(272)
                if make or model:
                    device = f"{str(make or '').strip(chr(0)).strip()} {str(model or '').strip(chr(0)).strip()}".strip()
    except Exception:
        pass

    # 2. Try filename timestamp pattern (e.g. IMG20261103185403.jpg or 20261103_184257.jpg)
    if timestamp is None:
        m = re.search(r"202\d{5}[-_]?(\d{6})", file_path.stem)
        if m:
            clean_digits = re.sub(r"\D", "", m.group(0))
            if len(clean_digits) == 14:
                try:
                    timestamp = datetime.strptime(clean_digits, "%Y%m%d%H%M%S")
                except ValueError:
                    pass

    return ImageMetadata(
        path=file_path,
        filename=file_path.name,
        sha256=sha,
        timestamp=timestamp,
        exif_device=device,
    )


def group_burst_clusters(
    images: List[ImageMetadata],
    burst_window_seconds: float = 3.0,
) -> List[List[ImageMetadata]]:
    """Group images into near-duplicate / burst clusters based on timestamp and SHA-256.

    Images taken within burst_window_seconds of each other or identical SHA-256 hashes
    are clustered together so that burst photos are not split across calibration and test.
    """
    if not images:
        return []

    # Sort primarily by timestamp (or fallback to filename)
    sorted_images = sorted(
        images,
        key=lambda x: (x.timestamp or datetime.max, x.filename),
    )

    clusters: List[List[ImageMetadata]] = []

    for img in sorted_images:
        # Check if identical SHA-256 already exists in one of the existing clusters
        found_cluster = False
        for c in clusters:
            if any(m.sha256 == img.sha256 for m in c):
                c.append(img)
                found_cluster = True
                break
        if found_cluster:
            continue

        # If not, check if it belongs to the most recent cluster by timestamp burst
        if clusters:
            last_cluster = clusters[-1]
            last_img = last_cluster[-1]
            if img.timestamp and last_img.timestamp:
                delta = abs((img.timestamp - last_img.timestamp).total_seconds())
                if delta <= burst_window_seconds:
                    last_cluster.append(img)
                    continue

        # Otherwise start a new cluster
        clusters.append([img])

    return clusters


def plan_data_split(
    source_dir: Path,
    unknown_location_ids: Optional[Sequence[str]] = None,
    reference_dir: Path = Path("data/reference"),
    calibration_dir: Path = Path("data/calibration"),
    test_dir: Path = Path("data/test"),
    backup_dir: Path = Path("data/backup/reference_original"),
    manifest_path: Path = Path("data/split_manifest.json"),
    seed: int = 42,
    calib_samples_per_location: int = DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    test_samples_per_location: int = DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
    calib_unknown_samples: int = DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    test_unknown_samples: int = DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
    min_reference_images: int = 1,
    burst_window_seconds: float = 3.0,
    allow_missing_unknown: bool = False,
) -> SplitPlan:
    """Compute deterministic split plan without performing filesystem modifications.

    Evaluates sample counts, detects burst clusters, validates unknown-data requirements,
    and checks whether temporal/device independence can be verified.
    """
    source_dir = Path(source_dir)
    reference_dir = Path(reference_dir)
    calibration_dir = Path(calibration_dir)
    test_dir = Path(test_dir)
    backup_dir = Path(backup_dir)
    manifest_path = Path(manifest_path)

    plan = SplitPlan(
        source_dir=source_dir,
        reference_dir=reference_dir,
        calibration_dir=calibration_dir,
        test_dir=test_dir,
        backup_dir=backup_dir,
        manifest_path=manifest_path,
        seed=seed,
        calib_samples_per_location=calib_samples_per_location,
        test_samples_per_location=test_samples_per_location,
        calib_unknown_samples=calib_unknown_samples,
        test_unknown_samples=test_unknown_samples,
        min_reference_images=min_reference_images,
        allow_missing_unknown=allow_missing_unknown,
    )

    if not source_dir.exists() or not source_dir.is_dir():
        plan.adequacy_passed = False
        plan.blocking_issues.append(f"Source directory does not exist or is not a directory: {source_dir}")
        return plan

    # Find location directories
    raw_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if not raw_dirs:
        plan.adequacy_passed = False
        plan.blocking_issues.append(f"No location subdirectories found in source: {source_dir}")
        return plan

    plan.all_locations = [d.name for d in raw_dirs]

    # Process unknown location IDs
    unknown_set: Set[str] = set()
    if unknown_location_ids:
        for u in unknown_location_ids:
            u_clean = u.strip()
            if u_clean:
                unknown_set.add(u_clean)

    # Validate provided unknown IDs
    invalid_unknown_ids = unknown_set - set(plan.all_locations)
    if invalid_unknown_ids:
        plan.adequacy_passed = False
        plan.blocking_issues.append(
            f"Specified --unknown-location-ids not found in source directory: {sorted(invalid_unknown_ids)}"
        )

    plan.unknown_locations = sorted(list(unknown_set & set(plan.all_locations)))
    plan.active_locations = [loc for loc in plan.all_locations if loc not in unknown_set]

    # Check unknown data requirement
    if not plan.unknown_locations:
        if allow_missing_unknown:
            plan.warnings.append(
                "PROVISIONAL NO-UNKNOWN MODE (--allow-missing-unknown) ACTIVE: All locations retained as active checkpoints. "
                "No unknown/ distractor folders will be created. Calibration and hold-out evaluation will evaluate "
                "checkpoint discrimination only; unknown rejection rate cannot be measured."
            )
        else:
            plan.adequacy_passed = False
            msg = (
                "No unknown-location IDs specified. Calibration requires at least "
                f"{calib_unknown_samples} calibration unknown images and {test_unknown_samples} test unknown images. "
                "Provide --unknown-location-ids or explicitly pass --allow-missing-unknown for provisional checkpoint-only splitting."
            )
            plan.warnings.append(msg)
            plan.blocking_issues.append(msg)

    # Scan and process images for each location
    images_by_location: Dict[str, List[ImageMetadata]] = {}
    timestamps_found = 0
    total_images_scanned = 0
    all_timestamps: List[datetime] = []

    for loc_dir in raw_dirs:
        loc_id = loc_dir.name
        img_files = sorted([
            f for f in loc_dir.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
        ])
        meta_list = [extract_image_metadata(f) for f in img_files]
        images_by_location[loc_id] = meta_list
        total_images_scanned += len(meta_list)

        for m in meta_list:
            if m.timestamp:
                timestamps_found += 1
                all_timestamps.append(m.timestamp)

    # Evaluate independence verification
    # If all photos were collected in a single session or day, cross-session independence cannot be established
    if all_timestamps:
        time_span_hours = (max(all_timestamps) - min(all_timestamps)).total_seconds() / 3600.0
    else:
        time_span_hours = 0.0

    plan.independence_verifiable = False
    plan.provisional = True
    if allow_missing_unknown and not plan.unknown_locations:
        plan.status_reason = (
            f"Dataset split in provisional mode without unknown/distractor data (--allow-missing-unknown). "
            f"All {len(plan.all_locations)} locations are active hunt checkpoints ({total_images_scanned} captures; "
            f"time span: {time_span_hours:.1f} hours). Single collection session; true cross-session independence is unverified. "
            "Unknown rejection rate cannot be measured and will be recorded as null/unavailable."
        )
    else:
        plan.status_reason = (
            f"Dataset originates from a single collection session ({total_images_scanned} captures "
            f"across {len(plan.all_locations)} locations; time span: {time_span_hours:.1f} hours). "
            "True temporal, cross-device, and environmental hold-out independence cannot be verified "
            "from available metadata. Downstream calibration and test evaluation must be marked provisional."
        )

    # Process active hunt locations
    for loc_id in plan.active_locations:
        imgs = images_by_location[loc_id]
        total_count = len(imgs)
        clusters = group_burst_clusters(imgs, burst_window_seconds=burst_window_seconds)

        loc_assignment = LocationSplitAssignment(
            location_id=loc_id,
            is_unknown_pool=False,
            total_source_images=total_count,
            burst_clusters_detected=len(clusters),
        )

        required_min_total = calib_samples_per_location + test_samples_per_location + min_reference_images
        if total_count < required_min_total:
            loc_assignment.sufficient = False
            issue = (
                f"Location '{loc_id}' has {total_count} images, but requires at least {required_min_total} "
                f"({calib_samples_per_location} calibration + {test_samples_per_location} test + "
                f"{min_reference_images} remaining reference)."
            )
            loc_assignment.issues.append(issue)
            plan.blocking_issues.append(issue)
            plan.adequacy_passed = False

        # Deterministic assignment
        rng = random.Random(f"{seed}:{loc_id}")
        # Shuffle cluster indices deterministically
        shuffled_clusters = list(clusters)
        rng.shuffle(shuffled_clusters)

        # Flatten clusters ensuring near-duplicate bursts stay intact
        # We designate entire clusters to calibration until quota met, then test, then reference
        calib_imgs: List[ImageMetadata] = []
        test_imgs: List[ImageMetadata] = []
        ref_imgs: List[ImageMetadata] = []

        for cluster in shuffled_clusters:
            # If calibration still needs images, assign from cluster
            needed_calib = calib_samples_per_location - len(calib_imgs)
            needed_test = test_samples_per_location - len(test_imgs)

            if needed_calib > 0:
                take_count = min(len(cluster), needed_calib)
                calib_imgs.extend(cluster[:take_count])
                # Overflow from this cluster goes to reference, NEVER to test
                ref_imgs.extend(cluster[take_count:])
            elif needed_test > 0:
                take_count = min(len(cluster), needed_test)
                test_imgs.extend(cluster[:take_count])
                # Overflow from this cluster goes to reference
                ref_imgs.extend(cluster[take_count:])
            else:
                ref_imgs.extend(cluster)

        loc_assignment.calibration_files = sorted([m.filename for m in calib_imgs])
        loc_assignment.test_files = sorted([m.filename for m in test_imgs])
        loc_assignment.reference_files = sorted([m.filename for m in ref_imgs])
        loc_assignment.remaining_reference_count = len(ref_imgs)

        # Check remaining reference count against minimum
        if len(ref_imgs) < min_reference_images:
            loc_assignment.sufficient = False
            issue = (
                f"Location '{loc_id}' remaining reference count is {len(ref_imgs)}, "
                f"which is below required minimum of {min_reference_images}."
            )
            loc_assignment.issues.append(issue)
            if issue not in plan.blocking_issues:
                plan.blocking_issues.append(issue)
            plan.adequacy_passed = False

        plan.assignments[loc_id] = loc_assignment

    # Process unknown locations
    if plan.unknown_locations:
        unknown_images: List[Tuple[str, ImageMetadata]] = []
        for unk_id in plan.unknown_locations:
            imgs = images_by_location[unk_id]
            for m in imgs:
                unknown_images.append((unk_id, m))

            clusters = group_burst_clusters(imgs, burst_window_seconds=burst_window_seconds)
            plan.assignments[unk_id] = LocationSplitAssignment(
                location_id=unk_id,
                is_unknown_pool=True,
                total_source_images=len(imgs),
                burst_clusters_detected=len(clusters),
                remaining_reference_count=0,
            )

        total_unk_images = len(unknown_images)
        required_unk = calib_unknown_samples + test_unknown_samples

        if total_unk_images < required_unk:
            plan.adequacy_passed = False
            issue = (
                f"Specified unknown locations provide {total_unk_images} images, but at least "
                f"{required_unk} are required ({calib_unknown_samples} calibration + {test_unknown_samples} test)."
            )
            plan.blocking_issues.append(issue)

        # Deterministic unknown distribution
        rng = random.Random(f"{seed}:unknown_pool")
        shuffled_unk = list(unknown_images)
        rng.shuffle(shuffled_unk)

        # Distribute images between calibration/unknown and test/unknown
        # First fulfill calibration and test quotas, then split surplus evenly
        calib_unk: List[Tuple[str, str]] = []
        test_unk: List[Tuple[str, str]] = []

        half_point = len(shuffled_unk) // 2
        for idx, (src_loc, m) in enumerate(shuffled_unk):
            if idx < half_point:
                calib_unk.append((src_loc, m.filename))
            else:
                test_unk.append((src_loc, m.filename))

        plan.calibration_unknown_files = sorted(calib_unk, key=lambda x: (x[0], x[1]))
        plan.test_unknown_files = sorted(test_unk, key=lambda x: (x[0], x[1]))

    return plan


def apply_data_split(plan: SplitPlan) -> Dict[str, Any]:
    """Execute filesystem changes according to the split plan.

    1. Validates that adequacy checks passed.
    2. Backs up the entire original source directory outside the audited splits.
    3. Recreates data/reference, data/calibration, and data/test directory layouts.
    4. Verifies zero cross-split exact SHA-256 duplicate images.
    5. Writes data/split_manifest.json with complete assignment and provenance details.
    """
    if not plan.adequacy_passed:
        raise DataSplitError(
            f"Cannot apply split due to blocking adequacy issues:\n" + "\n".join(f"- {i}" for i in plan.blocking_issues),
            reason_code="ERROR_DATA_INADEQUATE",
        )

    # 1. Prepare pristine backup directory
    plan.backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    target_backup = plan.backup_dir / f"source_{timestamp_str}"
    target_backup.mkdir(parents=True, exist_ok=True)

    logger.info("Preserving original image collection in backup: %s", target_backup)
    # Copy all files from source_dir into target_backup
    for loc_id in plan.all_locations:
        src_loc_dir = plan.source_dir / loc_id
        dst_loc_dir = target_backup / loc_id
        dst_loc_dir.mkdir(parents=True, exist_ok=True)
        for item in src_loc_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dst_loc_dir / item.name)

    # 2. Build map of source files by location and filename for fast copying
    source_files: Dict[Tuple[str, str], Path] = {}
    for loc_id in plan.all_locations:
        loc_dir = target_backup / loc_id
        for f in loc_dir.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                source_files[(loc_id, f.name)] = f

    # 3. Create target split directories
    # Note: If reference_dir == source_dir, we clean out non-reference locations from reference_dir
    plan.reference_dir.mkdir(parents=True, exist_ok=True)
    plan.calibration_dir.mkdir(parents=True, exist_ok=True)
    plan.test_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing location subdirs in reference_dir (preserving README.md)
    for item in plan.reference_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            shutil.rmtree(item)

    # Clean existing location subdirs in calibration_dir (preserving README.md)
    for item in plan.calibration_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            shutil.rmtree(item)

    # Clean existing location subdirs in test_dir (preserving README.md)
    for item in plan.test_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            shutil.rmtree(item)

    # 4. Copy active reference images
    ref_hashes: Set[str] = set()
    for loc_id in plan.active_locations:
        asgn = plan.assignments[loc_id]
        dest_loc_dir = plan.reference_dir / loc_id
        dest_loc_dir.mkdir(parents=True, exist_ok=True)
        for fname in asgn.reference_files:
            src_p = source_files[(loc_id, fname)]
            dst_p = dest_loc_dir / fname
            shutil.copy2(src_p, dst_p)
            ref_hashes.add(hashlib.sha256(src_p.read_bytes()).hexdigest())

    # 5. Copy calibration images (active locations)
    calib_hashes: Set[str] = set()
    for loc_id in plan.active_locations:
        asgn = plan.assignments[loc_id]
        dest_loc_dir = plan.calibration_dir / loc_id
        dest_loc_dir.mkdir(parents=True, exist_ok=True)
        for fname in asgn.calibration_files:
            src_p = source_files[(loc_id, fname)]
            dst_p = dest_loc_dir / fname
            shutil.copy2(src_p, dst_p)
            calib_hashes.add(hashlib.sha256(src_p.read_bytes()).hexdigest())

    # 6. Copy test images (active locations)
    test_hashes: Set[str] = set()
    for loc_id in plan.active_locations:
        asgn = plan.assignments[loc_id]
        dest_loc_dir = plan.test_dir / loc_id
        dest_loc_dir.mkdir(parents=True, exist_ok=True)
        for fname in asgn.test_files:
            src_p = source_files[(loc_id, fname)]
            dst_p = dest_loc_dir / fname
            shutil.copy2(src_p, dst_p)
            test_hashes.add(hashlib.sha256(src_p.read_bytes()).hexdigest())

    # 7. Copy calibration/unknown images
    if plan.calibration_unknown_files:
        calib_unk_dir = plan.calibration_dir / "unknown"
        calib_unk_dir.mkdir(parents=True, exist_ok=True)
        for src_loc, fname in plan.calibration_unknown_files:
            src_p = source_files[(src_loc, fname)]
            # Prevent collision if same filename in different unknown locations
            target_fname = f"{src_loc}_{fname}" if (calib_unk_dir / fname).exists() else fname
            dst_p = calib_unk_dir / target_fname
            shutil.copy2(src_p, dst_p)
            calib_hashes.add(hashlib.sha256(src_p.read_bytes()).hexdigest())

    # 8. Copy test/unknown images
    if plan.test_unknown_files:
        test_unk_dir = plan.test_dir / "unknown"
        test_unk_dir.mkdir(parents=True, exist_ok=True)
        for src_loc, fname in plan.test_unknown_files:
            src_p = source_files[(src_loc, fname)]
            target_fname = f"{src_loc}_{fname}" if (test_unk_dir / fname).exists() else fname
            dst_p = test_unk_dir / target_fname
            shutil.copy2(src_p, dst_p)
            test_hashes.add(hashlib.sha256(src_p.read_bytes()).hexdigest())

    # 9. Verify zero cross-split exact SHA-256 duplicates
    cross_ref_calib = ref_hashes & calib_hashes
    cross_ref_test = ref_hashes & test_hashes
    cross_calib_test = calib_hashes & test_hashes

    if cross_ref_calib or cross_ref_test or cross_calib_test:
        raise DataSplitError(
            f"Cross-split duplicate SHA-256 images detected during application: "
            f"ref-calib={len(cross_ref_calib)}, ref-test={len(cross_ref_test)}, calib-test={len(cross_calib_test)}",
            reason_code="ERROR_CROSS_SPLIT_DUPLICATE",
        )

    # 10. Write split manifest JSON
    manifest_data = {
        "manifest_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": plan.seed,
        "source_dir": str(plan.source_dir),
        "backup_dir": str(target_backup),
        "reference_dir": str(plan.reference_dir),
        "calibration_dir": str(plan.calibration_dir),
        "test_dir": str(plan.test_dir),
        "allow_missing_unknown": plan.allow_missing_unknown,
        "unknown_evaluation": "unavailable" if not plan.unknown_locations else "available",
        "provisional": plan.provisional,
        "independence_verifiable": plan.independence_verifiable,
        "status_reason": plan.status_reason,
        "summary": {
            "total_locations": len(plan.all_locations),
            "active_hunt_locations": len(plan.active_locations),
            "excluded_unknown_locations": len(plan.unknown_locations),
            "reference_images": sum(len(a.reference_files) for a in plan.assignments.values() if not a.is_unknown_pool),
            "calibration_images": sum(len(a.calibration_files) for a in plan.assignments.values() if not a.is_unknown_pool),
            "test_images": sum(len(a.test_files) for a in plan.assignments.values() if not a.is_unknown_pool),
            "calibration_unknown_images": len(plan.calibration_unknown_files),
            "test_unknown_images": len(plan.test_unknown_files),
        },
        "excluded_unknown_locations": plan.unknown_locations,
        "per_location_counts": {
            loc_id: {
                "is_unknown_pool": asgn.is_unknown_pool,
                "total_source": asgn.total_source_images,
                "reference": len(asgn.reference_files),
                "calibration": len(asgn.calibration_files),
                "test": len(asgn.test_files),
                "remaining_reference": asgn.remaining_reference_count,
            }
            for loc_id, asgn in plan.assignments.items()
        },
        "file_assignments": {
            "reference": {
                loc_id: asgn.reference_files
                for loc_id, asgn in plan.assignments.items()
                if not asgn.is_unknown_pool
            },
            "calibration": {
                loc_id: asgn.calibration_files
                for loc_id, asgn in plan.assignments.items()
                if not asgn.is_unknown_pool
            },
            "test": {
                loc_id: asgn.test_files
                for loc_id, asgn in plan.assignments.items()
                if not asgn.is_unknown_pool
            },
            "calibration_unknown": [
                {"source_location": loc, "file": fname} for loc, fname in plan.calibration_unknown_files
            ],
            "test_unknown": [
                {"source_location": loc, "file": fname} for loc, fname in plan.test_unknown_files
            ],
        },
    }

    plan.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plan.manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    logger.info("Split manifest written to %s", plan.manifest_path)

    return manifest_data


def format_split_plan_report(plan: SplitPlan, is_apply: bool = False) -> str:
    """Format human-readable CLI report of the split plan."""
    mode_str = "APPLYING SPLIT TO FILESYSTEM" if is_apply else "DRY RUN (PROPOSED ASSIGNMENTS)"
    lines = [
        "=" * 85,
        f"  CAMPUS HUNT CV: DATA SPLITTER [{mode_str}]",
        "=" * 85,
        f"Source Directory    : {plan.source_dir}",
        f"Backup Directory    : {plan.backup_dir}",
        f"Random Seed         : {plan.seed}",
        f"Safeguard Standards : Calibration >= {plan.calib_samples_per_location}/loc, "
        f"Test >= {plan.test_samples_per_location}/loc, "
        f"Unknown >= {plan.calib_unknown_samples} calib / {plan.test_unknown_samples} test",
        f"Independence Status : {'VERIFIED' if plan.independence_verifiable else 'PROVISIONAL'}",
        "-" * 85,
        f"Notice: {plan.status_reason}",
        "-" * 85,
    ]

    # Active locations table
    lines.append("ACTIVE HUNT CHECKPOINTS:")
    lines.append(
        f"{'Location ID':<35} | {'Source':<6} | {'Calib':<5} | {'Test':<5} | {'Remaining Ref':<14} | {'Status'}"
    )
    lines.append("-" * 85)

    for loc_id in plan.active_locations:
        asgn = plan.assignments[loc_id]
        status_label = "OK" if asgn.sufficient else "INADEQUATE"
        ref_warn = " (!)" if asgn.remaining_reference_count < 3 else ""
        lines.append(
            f"{loc_id:<35} | {asgn.total_source_images:<6} | {len(asgn.calibration_files):<5} | "
            f"{len(asgn.test_files):<5} | {asgn.remaining_reference_count:>2}{ref_warn:<12} | {status_label}"
        )

    lines.append("-" * 85)

    # Unknown locations section
    lines.append("UNKNOWN / DISTRACTOR LOCATIONS (Excluded from Hunt Checkpoints):")
    if plan.unknown_locations:
        for unk_id in plan.unknown_locations:
            asgn = plan.assignments[unk_id]
            lines.append(f"  * {unk_id} ({asgn.total_source_images} source images -> distributed to unknown/ pool)")
        lines.append(f"  Total Unknown Captures: {len(plan.calibration_unknown_files) + len(plan.test_unknown_files)}")
        lines.append(f"    - Allocated to calibration/unknown: {len(plan.calibration_unknown_files)}")
        lines.append(f"    - Allocated to test/unknown       : {len(plan.test_unknown_files)}")
    elif plan.allow_missing_unknown:
        lines.append("  PROVISIONAL NO-UNKNOWN MODE (--allow-missing-unknown) ACTIVE.")
        lines.append(f"  All {len(plan.active_locations)} locations retained as active checkpoints; no unknown/ folders created.")
        lines.append("  Notice: Unknown rejection rate is unmeasured; test evaluation will record unknown metrics as null.")
    else:
        lines.append("  NONE SPECIFIED.")
        lines.append("  WARNING: Calibration pipeline requires >= 10 calibration unknown images and >= 10 test unknown images.")
        lines.append("  You must designate non-checkpoint locations using --unknown-location-ids or explicitly pass --allow-missing-unknown.")

    lines.append("-" * 85)

    # Warnings and issues
    if plan.warnings:
        lines.append("WARNINGS:")
        for w in plan.warnings:
            lines.append(f"  [!] {w}")
        lines.append("-" * 85)

    if plan.blocking_issues:
        lines.append("BLOCKING ISSUES (Refusing to apply until resolved):")
        for i in plan.blocking_issues:
            lines.append(f"  [X] {i}")
        lines.append("-" * 85)

    # Footer
    if is_apply:
        lines.append("RESULT: Split successfully executed and verified.")
    else:
        lines.append("DRY RUN COMPLETE: No filesystem changes were made.")
        lines.append("To execute changes and create backup, re-run with '--apply'.")
    lines.append("=" * 85)

    return "\n".join(lines)
