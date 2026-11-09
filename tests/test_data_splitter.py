"""Offline tests for the campus data splitting module (campus-hunt-split-data)."""

import hashlib
import json
from pathlib import Path
import pytest
from PIL import Image

from campus_hunt_cv.data_splitter import (
    apply_data_split,
    extract_image_metadata,
    group_burst_clusters,
    plan_data_split,
    ImageMetadata,
)
from campus_hunt_cv.exceptions import DataSplitError


def _create_mock_image(path: Path, color=(100, 100, 100)) -> Path:
    """Helper to create a small valid image on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (64, 64), color=color)
    img.save(path, format="JPEG")
    return path


def _populate_mock_source_dir(root: Path, counts: dict) -> Path:
    """Create mock source folder structure with specified image counts per location."""
    for loc_id, n in counts.items():
        loc_dir = root / loc_id
        loc_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            # Ensure globally unique RGB values for every image across all locations
            h = int(hashlib.md5(f"{loc_id}_{i}".encode()).hexdigest()[:6], 16)
            r = (h >> 16) & 0xFF
            g = (h >> 8) & 0xFF
            b = h & 0xFF
            _create_mock_image(loc_dir / f"img_{i:03d}.jpg", color=(r, g, b))
    return root


def test_dry_run_does_not_modify_filesystem(tmp_path: Path):
    """Dry-run must compute assignments without creating or modifying any files."""
    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"clock_tower": 15, "library": 15, "hallway": 25})

    ref_dir = tmp_path / "data" / "reference"
    calib_dir = tmp_path / "data" / "calibration"
    test_dir = tmp_path / "data" / "test"
    backup_dir = tmp_path / "data" / "backup"
    manifest_path = tmp_path / "manifest.json"

    plan = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=["hallway"],
        reference_dir=ref_dir,
        calibration_dir=calib_dir,
        test_dir=test_dir,
        backup_dir=backup_dir,
        manifest_path=manifest_path,
        seed=42,
    )

    assert plan.adequacy_passed is True
    assert set(plan.active_locations) == {"clock_tower", "library"}
    assert plan.unknown_locations == ["hallway"]

    # Verify that target directories were NOT created
    assert not ref_dir.exists()
    assert not calib_dir.exists()
    assert not test_dir.exists()
    assert not backup_dir.exists()
    assert not manifest_path.exists()

    # Verify assignments were planned
    for loc in ["clock_tower", "library"]:
        asgn = plan.assignments[loc]
        assert len(asgn.calibration_files) == 5
        assert len(asgn.test_files) == 5
        assert len(asgn.reference_files) == 5
        assert asgn.remaining_reference_count == 5


def test_deterministic_assignments_with_seed(tmp_path: Path):
    """Splits planned with the same seed must yield identical file assignments."""
    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"loc_a": 16, "loc_b": 18, "distractor": 22})

    plan1 = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=["distractor"],
        seed=123,
    )
    plan2 = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=["distractor"],
        seed=123,
    )
    plan3 = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=["distractor"],
        seed=999,
    )

    # Identical seed must produce identical assignments
    assert plan1.assignments["loc_a"].calibration_files == plan2.assignments["loc_a"].calibration_files
    assert plan1.assignments["loc_a"].test_files == plan2.assignments["loc_a"].test_files
    assert plan1.assignments["loc_a"].reference_files == plan2.assignments["loc_a"].reference_files
    assert plan1.calibration_unknown_files == plan2.calibration_unknown_files
    assert plan1.test_unknown_files == plan2.test_unknown_files

    # Different seed should produce different permutation
    assert plan1.assignments["loc_a"].calibration_files != plan3.assignments["loc_a"].calibration_files


def test_insufficient_image_counts_fails_closed(tmp_path: Path):
    """If an active location has fewer than required total images, planning fails closed."""
    source_dir = tmp_path / "source"
    # Needs at least 5 calib + 5 test + 1 ref = 11 images; provide only 8
    _populate_mock_source_dir(source_dir, {"small_loc": 8, "distractor": 25})

    plan = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=["distractor"],
        calib_samples_per_location=5,
        test_samples_per_location=5,
        min_reference_images=1,
    )

    assert plan.adequacy_passed is False
    assert any("small_loc" in issue and "requires at least 11" in issue for issue in plan.blocking_issues)

    # Calling apply on inadequate plan must raise DataSplitError
    with pytest.raises(DataSplitError) as exc_info:
        apply_data_split(plan)
    assert exc_info.value.reason_code == "ERROR_DATA_INADEQUATE"


def test_insufficient_unknown_images_fails_closed(tmp_path: Path):
    """If unknown locations provide fewer than required unknown captures, planning fails closed."""
    source_dir = tmp_path / "source"
    # Provide 15 images in distractor, but required is 10 calib + 10 test = 20
    _populate_mock_source_dir(source_dir, {"loc_a": 15, "distractor": 15})

    plan = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=["distractor"],
        calib_unknown_samples=10,
        test_unknown_samples=10,
    )

    assert plan.adequacy_passed is False
    assert any("at least 20 are required" in issue for issue in plan.blocking_issues)


def test_unknown_locations_not_guessed(tmp_path: Path):
    """When no unknown locations are specified, splitter must not invent or guess an unknown set."""
    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"loc_a": 15, "loc_b": 15})

    plan = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=None,
    )

    assert plan.unknown_locations == []
    assert plan.calibration_unknown_files == []
    assert plan.test_unknown_files == []
    assert any("No unknown-location IDs specified" in w for w in plan.warnings)


def test_no_exact_sha256_overlap_across_splits_and_preserves_originals(tmp_path: Path):
    """On --apply, original photos must be preserved in backup and zero cross-split duplicates created."""
    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"loc_1": 15, "loc_2": 15, "non_checkpoint": 25})

    ref_dir = tmp_path / "data" / "reference"
    calib_dir = tmp_path / "data" / "calibration"
    test_dir = tmp_path / "data" / "test"
    backup_dir = tmp_path / "data" / "backup"
    manifest_p = tmp_path / "data" / "split_manifest.json"

    plan = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=["non_checkpoint"],
        reference_dir=ref_dir,
        calibration_dir=calib_dir,
        test_dir=test_dir,
        backup_dir=backup_dir,
        manifest_path=manifest_p,
        seed=42,
    )
    assert plan.adequacy_passed is True

    manifest_data = apply_data_split(plan)

    # 1. Verify backup exists and holds all original files
    assert backup_dir.exists()
    backup_runs = list(backup_dir.iterdir())
    assert len(backup_runs) == 1
    backup_run_dir = backup_runs[0]
    for loc_name in ["loc_1", "loc_2", "non_checkpoint"]:
        assert (backup_run_dir / loc_name).exists()
        orig_count = len(list((source_dir / loc_name).iterdir()))
        bk_count = len(list((backup_run_dir / loc_name).iterdir()))
        assert orig_count == bk_count

    # 2. Check layout of audited splits
    # Active locations in reference, calibration, and test
    for active_loc in ["loc_1", "loc_2"]:
        assert (ref_dir / active_loc).exists()
        assert (calib_dir / active_loc).exists()
        assert (test_dir / active_loc).exists()
        assert len(list((calib_dir / active_loc).iterdir())) == 5
        assert len(list((test_dir / active_loc).iterdir())) == 5
        assert len(list((ref_dir / active_loc).iterdir())) == 5

    # Excluded unknown location should NOT be in reference
    assert not (ref_dir / "non_checkpoint").exists()
    assert (calib_dir / "unknown").exists()
    assert (test_dir / "unknown").exists()
    assert len(list((calib_dir / "unknown").iterdir())) >= 10
    assert len(list((test_dir / "unknown").iterdir())) >= 10

    # 3. Verify ZERO cross-split SHA-256 duplicates across audited splits
    def _collect_hashes(directory: Path):
        hashes = set()
        for p in directory.glob("**/*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
        return hashes

    ref_hashes = _collect_hashes(ref_dir)
    calib_hashes = _collect_hashes(calib_dir)
    test_hashes = _collect_hashes(test_dir)

    assert len(ref_hashes & calib_hashes) == 0, "Reference and Calibration share identical images!"
    assert len(ref_hashes & test_hashes) == 0, "Reference and Test share identical images!"
    assert len(calib_hashes & test_hashes) == 0, "Calibration and Test share identical images!"

    # 4. Check manifest details
    assert manifest_p.exists()
    loaded_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert loaded_manifest["provisional"] is True
    assert loaded_manifest["independence_verifiable"] is False
    assert loaded_manifest["seed"] == 42
    assert loaded_manifest["excluded_unknown_locations"] == ["non_checkpoint"]


def test_burst_clustering_keeps_bursts_together():
    """Near-duplicate bursts must be grouped together into the same cluster."""
    t_base = 1700000000.0
    from datetime import datetime
    img1 = ImageMetadata(path=Path("1.jpg"), filename="1.jpg", sha256="h1", timestamp=datetime.fromtimestamp(t_base))
    img2 = ImageMetadata(path=Path("2.jpg"), filename="2.jpg", sha256="h2", timestamp=datetime.fromtimestamp(t_base + 2.0))
    img3 = ImageMetadata(path=Path("3.jpg"), filename="3.jpg", sha256="h3", timestamp=datetime.fromtimestamp(t_base + 10.0))

    clusters = group_burst_clusters([img1, img2, img3], burst_window_seconds=3.0)
    assert len(clusters) == 2
    assert len(clusters[0]) == 2
    assert clusters[0][0].filename == "1.jpg"
    assert clusters[0][1].filename == "2.jpg"
    assert len(clusters[1]) == 1
    assert clusters[1][0].filename == "3.jpg"


def test_cli_split_data_dry_run_and_apply_integration(tmp_path: Path, monkeypatch, capsys):
    """Test CLI main() entrypoint in both dry-run and apply modes."""
    from campus_hunt_cv.cli.split_data import main

    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"loc_a": 15, "loc_b": 15, "distractor": 22})

    ref_dir = tmp_path / "ref"
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    backup_dir = tmp_path / "backup"
    manifest_p = tmp_path / "manifest.json"

    # 1. Dry run via CLI
    monkeypatch.setattr(
        "sys.argv",
        [
            "campus-hunt-split-data",
            "--source-dir", str(source_dir),
            "--unknown-location-ids", "distractor",
            "--reference-dir", str(ref_dir),
            "--calibration-dir", str(calib_dir),
            "--test-dir", str(test_dir),
            "--backup-dir", str(backup_dir),
            "--manifest-path", str(manifest_p),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "DRY RUN (PROPOSED ASSIGNMENTS)" in captured.out
    assert "loc_a" in captured.out
    assert not ref_dir.exists()

    # 2. Apply via CLI
    monkeypatch.setattr(
        "sys.argv",
        [
            "campus-hunt-split-data",
            "--source-dir", str(source_dir),
            "--unknown-location-ids", "distractor",
            "--reference-dir", str(ref_dir),
            "--calibration-dir", str(calib_dir),
            "--test-dir", str(test_dir),
            "--backup-dir", str(backup_dir),
            "--manifest-path", str(manifest_p),
            "--apply",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "APPLYING SPLIT TO FILESYSTEM" in captured.out
    assert ref_dir.exists()
    assert (ref_dir / "loc_a").exists()
    assert (calib_dir / "unknown").exists()
    assert manifest_p.exists()


def test_allow_missing_unknown_dry_run_and_plan(tmp_path: Path):
    """Provisional planning without unknowns marks plan provisional and passes adequacy."""
    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"loc_1": 15, "loc_2": 15})

    plan = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=None,
        allow_missing_unknown=True,
        seed=42,
    )

    assert plan.adequacy_passed is True
    assert plan.provisional is True
    assert plan.allow_missing_unknown is True
    assert plan.unknown_evaluation == "unavailable"
    assert plan.unknown_locations == []
    assert plan.calibration_unknown_files == []
    assert plan.test_unknown_files == []
    assert set(plan.active_locations) == {"loc_1", "loc_2"}
    assert any("PROVISIONAL NO-UNKNOWN MODE" in w for w in plan.warnings)


def test_allow_missing_unknown_apply_preserves_originals_and_omits_unknown_dir(tmp_path: Path):
    """Applying provisional split preserves originals in backup and never creates unknown/ folders."""
    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"loc_1": 15, "loc_2": 15})

    ref_dir = tmp_path / "data" / "reference"
    calib_dir = tmp_path / "data" / "calibration"
    test_dir = tmp_path / "data" / "test"
    backup_dir = tmp_path / "data" / "backup"
    manifest_p = tmp_path / "data" / "split_manifest.json"

    plan = plan_data_split(
        source_dir=source_dir,
        unknown_location_ids=None,
        allow_missing_unknown=True,
        reference_dir=ref_dir,
        calibration_dir=calib_dir,
        test_dir=test_dir,
        backup_dir=backup_dir,
        manifest_path=manifest_p,
        seed=42,
    )
    assert plan.adequacy_passed is True

    apply_data_split(plan)

    # 1. Originals preserved in backup
    assert backup_dir.exists()
    assert (list(backup_dir.iterdir())[0] / "loc_1").exists()
    assert len(list((list(backup_dir.iterdir())[0] / "loc_1").iterdir())) == 15

    # 2. Splits exist for all active locations
    for loc in ["loc_1", "loc_2"]:
        assert (ref_dir / loc).exists()
        assert (calib_dir / loc).exists()
        assert (test_dir / loc).exists()
        assert len(list((calib_dir / loc).iterdir())) == 5
        assert len(list((test_dir / loc).iterdir())) == 5
        assert len(list((ref_dir / loc).iterdir())) == 5

    # 3. NO unknown/ folder in calibration or test
    assert not (calib_dir / "unknown").exists()
    assert not (test_dir / "unknown").exists()

    # 4. Zero cross-split SHA-256 duplicates
    def _collect_hashes(directory: Path):
        hashes = set()
        for p in directory.glob("**/*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
        return hashes

    ref_hashes = _collect_hashes(ref_dir)
    calib_hashes = _collect_hashes(calib_dir)
    test_hashes = _collect_hashes(test_dir)

    assert len(ref_hashes & calib_hashes) == 0
    assert len(ref_hashes & test_hashes) == 0
    assert len(calib_hashes & test_hashes) == 0

    # 5. Manifest details
    assert manifest_p.exists()
    loaded_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert loaded_manifest["provisional"] is True
    assert loaded_manifest["allow_missing_unknown"] is True
    assert loaded_manifest["unknown_evaluation"] == "unavailable"
    assert loaded_manifest["excluded_unknown_locations"] == []


def test_allow_missing_unknown_insufficient_active_location_fails_closed(tmp_path: Path):
    """Even in provisional mode, active locations with insufficient photos (< 11) fail closed."""
    source_dir = tmp_path / "source"
    # Needs at least 5 calib + 5 test + 1 ref = 11; provide 10
    _populate_mock_source_dir(source_dir, {"short_loc": 10, "good_loc": 15})

    plan = plan_data_split(
        source_dir=source_dir,
        allow_missing_unknown=True,
    )

    assert plan.adequacy_passed is False
    assert any("short_loc" in issue and "requires at least 11" in issue for issue in plan.blocking_issues)

    with pytest.raises(DataSplitError) as exc_info:
        apply_data_split(plan)
    assert exc_info.value.reason_code == "ERROR_DATA_INADEQUATE"


def test_cli_split_data_provisional_flag_integration(tmp_path: Path, monkeypatch, capsys):
    """Test CLI main() with --allow-missing-unknown flag in both dry-run and apply modes."""
    from campus_hunt_cv.cli.split_data import main

    source_dir = tmp_path / "source"
    _populate_mock_source_dir(source_dir, {"loc_a": 15, "loc_b": 15})

    ref_dir = tmp_path / "ref"
    calib_dir = tmp_path / "calib"
    test_dir = tmp_path / "test"
    backup_dir = tmp_path / "backup"
    manifest_p = tmp_path / "manifest.json"

    # Dry run
    monkeypatch.setattr(
        "sys.argv",
        [
            "campus-hunt-split-data",
            "--source-dir", str(source_dir),
            "--allow-missing-unknown",
            "--reference-dir", str(ref_dir),
            "--calibration-dir", str(calib_dir),
            "--test-dir", str(test_dir),
            "--backup-dir", str(backup_dir),
            "--manifest-path", str(manifest_p),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "PROVISIONAL NO-UNKNOWN MODE" in captured.out
    assert "DRY RUN (PROPOSED ASSIGNMENTS)" in captured.out
    assert not ref_dir.exists()

    # Apply
    monkeypatch.setattr(
        "sys.argv",
        [
            "campus-hunt-split-data",
            "--source-dir", str(source_dir),
            "--allow-missing-unknown",
            "--reference-dir", str(ref_dir),
            "--calibration-dir", str(calib_dir),
            "--test-dir", str(test_dir),
            "--backup-dir", str(backup_dir),
            "--manifest-path", str(manifest_p),
            "--apply",
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "APPLYING SPLIT TO FILESYSTEM" in captured.out
    assert ref_dir.exists()
    assert not (calib_dir / "unknown").exists()
    assert manifest_p.exists()


