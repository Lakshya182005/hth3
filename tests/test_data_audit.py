"""Unit and integration tests for dataset audit module and CLI."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from campus_hunt_cv.cli.audit_data import main as audit_cli_main
from campus_hunt_cv.config import CampusHuntConfig
from campus_hunt_cv.data_audit import audit_dataset
from tests.conftest import create_test_image_bytes


def create_split_dataset(
    base_dir: Path,
    locations: list[str],
    ref_counts: dict[str, int],
    calib_counts: dict[str, int],
    test_counts: dict[str, int],
    calib_unknown_count: int = 10,
    test_unknown_count: int = 10,
    start_seed: int = 0,
):
    """Populate reference, calibration, and test directories with unique valid images."""
    ref_dir = base_dir / "reference"
    calib_dir = base_dir / "calibration"
    test_dir = base_dir / "test"

    seed = start_seed

    # 1. Reference split
    for loc in locations:
        loc_dir = ref_dir / loc
        loc_dir.mkdir(parents=True, exist_ok=True)
        count = ref_counts.get(loc, 1)
        for i in range(count):
            seed += 1
            color = (seed % 255, (seed * 7) % 255, (seed * 13) % 255)
            img_bytes = create_test_image_bytes(width=50, height=50, color=color, format="JPEG")
            (loc_dir / f"ref_{loc}_{i}.jpg").write_bytes(img_bytes)

    # 2. Calibration split
    for loc in locations:
        loc_dir = calib_dir / loc
        loc_dir.mkdir(parents=True, exist_ok=True)
        count = calib_counts.get(loc, 5)
        for i in range(count):
            seed += 1
            color = (seed % 255, (seed * 7) % 255, (seed * 13) % 255)
            img_bytes = create_test_image_bytes(width=50, height=50, color=color, format="JPEG")
            (loc_dir / f"calib_{loc}_{i}.jpg").write_bytes(img_bytes)

    if calib_unknown_count > 0:
        unk_dir = calib_dir / "unknown"
        unk_dir.mkdir(parents=True, exist_ok=True)
        for i in range(calib_unknown_count):
            seed += 1
            color = (seed % 255, (seed * 7) % 255, (seed * 13) % 255)
            img_bytes = create_test_image_bytes(width=50, height=50, color=color, format="JPEG")
            (unk_dir / f"calib_unk_{i}.jpg").write_bytes(img_bytes)

    # 3. Test split
    for loc in locations:
        loc_dir = test_dir / loc
        loc_dir.mkdir(parents=True, exist_ok=True)
        count = test_counts.get(loc, 5)
        for i in range(count):
            seed += 1
            color = (seed % 255, (seed * 7) % 255, (seed * 13) % 255)
            img_bytes = create_test_image_bytes(width=50, height=50, color=color, format="JPEG")
            (loc_dir / f"test_{loc}_{i}.jpg").write_bytes(img_bytes)

    if test_unknown_count > 0:
        unk_dir = test_dir / "unknown"
        unk_dir.mkdir(parents=True, exist_ok=True)
        for i in range(test_unknown_count):
            seed += 1
            color = (seed % 255, (seed * 7) % 255, (seed * 13) % 255)
            img_bytes = create_test_image_bytes(width=50, height=50, color=color, format="JPEG")
            (unk_dir / f"test_unk_{i}.jpg").write_bytes(img_bytes)

    return ref_dir, calib_dir, test_dir


def test_audit_valid_directory_structure_passes(tmp_path: Path):
    """A well-formed dataset with adequate counts and distinct images passes the audit."""
    locations = ["clock_tower", "library_atrium"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"clock_tower": 2, "library_atrium": 2},
        calib_counts={"clock_tower": 5, "library_atrium": 5},
        test_counts={"clock_tower": 5, "library_atrium": 5},
        calib_unknown_count=10,
        test_unknown_count=10,
    )

    report_path = tmp_path / "audit_report.json"
    report = audit_dataset(
        reference_dir=ref_dir,
        calibration_dir=calib_dir,
        test_dir=test_dir,
        min_reference_images_per_location=1,
        min_calibration_samples_per_location=5,
        min_calibration_unknown_samples=10,
        min_test_samples_per_location=5,
        min_test_unknown_samples=10,
    )

    assert report.audit_passed is True
    assert len(report.issues) == 0
    assert report.summary["total_reference_locations"] == 2
    assert report.summary["total_valid_images"] == (2*2) + (2*5 + 10) + (2*5 + 10)
    assert report.summary["cross_split_duplicate_groups"] == 0
    assert report.summary["total_invalid_images"] == 0

    # Ensure CLI runs cleanly and exits with 0
    cli_exit = audit_cli_main([
        "--reference-dir", str(ref_dir),
        "--calibration-dir", str(calib_dir),
        "--test-dir", str(test_dir),
        "--min-reference-images", "1",
        "--min-calib-samples", "5",
        "--min-calib-unknown", "10",
        "--min-test-samples", "5",
        "--min-test-unknown", "10",
        "--output-report", str(report_path),
    ])
    assert cli_exit == 0
    assert report_path.exists()

    saved_json = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_json["audit_passed"] is True
    assert "accuracy" not in saved_json  # No accuracy claims


def test_audit_missing_location_folders(tmp_path: Path):
    """Missing location folders in calibration or test splits fail the audit."""
    locations = ["clock_tower", "science_center"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"clock_tower": 2, "science_center": 2},
        calib_counts={"clock_tower": 5, "science_center": 5},
        test_counts={"clock_tower": 5, "science_center": 5},
    )

    # Delete science_center from calibration, clock_tower from test
    import shutil
    shutil.rmtree(calib_dir / "science_center")
    shutil.rmtree(test_dir / "clock_tower")

    report = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)

    assert report.audit_passed is False
    assert "science_center" in report.splits["calibration"].missing_locations
    assert "clock_tower" in report.splits["test"].missing_locations

    issues_str = " ".join(report.issues)
    assert "Calibration split is missing location folder: 'science_center'" in issues_str
    assert "Test split is missing location folder: 'clock_tower'" in issues_str

    cli_exit = audit_cli_main([
        "--reference-dir", str(ref_dir),
        "--calibration-dir", str(calib_dir),
        "--test-dir", str(test_dir),
    ])
    assert cli_exit == 1


def test_audit_unknown_folder_requirements(tmp_path: Path):
    """Enforces unknown/ in calibration and test splits, and forbids unknown/ in reference split."""
    locations = ["bell_tower"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"bell_tower": 2},
        calib_counts={"bell_tower": 5},
        test_counts={"bell_tower": 5},
        calib_unknown_count=10,
        test_unknown_count=10,
    )

    # Case A: Calibration is missing unknown/
    import shutil
    shutil.rmtree(calib_dir / "unknown")
    report = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)
    assert report.audit_passed is False
    assert any("Calibration split is missing required 'unknown/'" in issue for issue in report.issues)

    # Re-create calibration unknown, remove test unknown
    (calib_dir / "unknown").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (calib_dir / "unknown" / f"u_{i}.jpg").write_bytes(create_test_image_bytes(color=(10, 20, i)))
    shutil.rmtree(test_dir / "unknown")

    # Case B: Test is missing unknown/
    report_b = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)
    assert report_b.audit_passed is False
    assert any("Test split is missing required 'unknown/'" in issue for issue in report_b.issues)

    # Case C: Reference split unexpectedly contains unknown/
    (test_dir / "unknown").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (test_dir / "unknown" / f"tu_{i}.jpg").write_bytes(create_test_image_bytes(color=(30, 40, i)))
    (ref_dir / "unknown").mkdir(parents=True, exist_ok=True)
    (ref_dir / "unknown" / "bad.jpg").write_bytes(create_test_image_bytes(color=(50, 60, 70)))

    report_c = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)
    assert report_c.audit_passed is False
    assert any("Reference split contains unexpected 'unknown/'" in issue for issue in report_c.issues)


def test_audit_insufficient_sample_counts(tmp_path: Path):
    """Locations or unknown folders with fewer valid images than configured fail the audit."""
    locations = ["dining_hall"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"dining_hall": 1},
        calib_counts={"dining_hall": 3},  # < 5 required
        test_counts={"dining_hall": 5},
        calib_unknown_count=10,
        test_unknown_count=4,  # < 10 required
    )

    report = audit_dataset(
        reference_dir=ref_dir,
        calibration_dir=calib_dir,
        test_dir=test_dir,
        min_calibration_samples_per_location=5,
        min_test_unknown_samples=10,
    )

    assert report.audit_passed is False
    assert "dining_hall" in report.splits["calibration"].insufficient_locations
    assert "unknown" in report.splits["test"].insufficient_locations
    assert any("Insufficient captures for 'dining_hall' in calibration split: found 3, required at least 5" in i for i in report.issues)
    assert any("Insufficient unknown captures in test split: found 4, required at least 10" in i for i in report.issues)


def test_audit_invalid_and_corrupt_images(tmp_path: Path):
    """Files that cannot decode, violate limits, or are corrupt are detected and reported."""
    locations = ["chapel"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"chapel": 2},
        calib_counts={"chapel": 5},
        test_counts={"chapel": 5},
    )

    # 1. Corrupted garbage file
    (calib_dir / "chapel" / "corrupt.jpg").write_bytes(b"NOT_A_VALID_JPEG_HEADER_BYTES_12345")

    # 2. 0-byte file
    (test_dir / "unknown" / "empty.png").write_bytes(b"")

    # 3. Too small dimensions (10x10 < min 32)
    tiny_img = create_test_image_bytes(width=10, height=10)
    (ref_dir / "chapel" / "tiny.jpg").write_bytes(tiny_img)

    report = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)

    assert report.audit_passed is False
    assert report.summary["total_invalid_images"] >= 3
    assert any("Invalid image in reference location 'chapel/tiny.jpg'" in i for i in report.issues)
    assert any("Invalid image in calibration 'chapel/corrupt.jpg'" in i for i in report.issues)
    assert any("Invalid image in test 'unknown/empty.png'" in i for i in report.issues)


def test_audit_exact_duplicates_across_splits(tmp_path: Path):
    """Cross-split duplicates (SHA-256) are strictly detected and flagged as audit failures."""
    locations = ["statue"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"statue": 2},
        calib_counts={"statue": 5},
        test_counts={"statue": 5},
    )

    # Duplicate: copy an identical byte payload into both reference and test
    shared_bytes = create_test_image_bytes(width=60, height=60, color=(111, 222, 33))
    (ref_dir / "statue" / "shared_leak.jpg").write_bytes(shared_bytes)
    (test_dir / "statue" / "shared_leak_test.jpg").write_bytes(shared_bytes)

    report = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)

    assert report.audit_passed is False
    assert len(report.duplicates["cross_split_duplicates"]) >= 1

    dup_group = report.duplicates["cross_split_duplicates"][0]
    paths = [f["path"] for f in dup_group.files]
    assert any("shared_leak.jpg" in p for p in paths)
    assert any("shared_leak_test.jpg" in p for p in paths)

    assert any("Cross-split duplicate image detected" in issue for issue in report.issues)


def test_audit_duplicate_across_different_locations_within_split(tmp_path: Path):
    """Reusing the exact same photo across different locations within a split is flagged."""
    locations = ["hall_a", "hall_b"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"hall_a": 2, "hall_b": 2},
        calib_counts={"hall_a": 5, "hall_b": 5},
        test_counts={"hall_a": 5, "hall_b": 5},
    )

    # Place identical photo in calibration hall_a and calibration hall_b
    dup_bytes = create_test_image_bytes(width=64, height=64, color=(44, 55, 66))
    (calib_dir / "hall_a" / "reused.jpg").write_bytes(dup_bytes)
    (calib_dir / "hall_b" / "reused_copy.jpg").write_bytes(dup_bytes)

    report = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)
    assert report.audit_passed is False
    assert len(report.duplicates["within_split_duplicates"]) >= 1
    assert any("reused across different locations in calibration split" in i for i in report.issues)


def test_audit_unexpected_subdirectories_and_unsupported_files(tmp_path: Path):
    """Nested directories, empty folders, and non-image files are clearly reported."""
    locations = ["courtyard"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"courtyard": 2},
        calib_counts={"courtyard": 5},
        test_counts={"courtyard": 5},
    )

    # 1. Unexpected location folder in test split (not in reference and not unknown)
    (test_dir / "untracked_building").mkdir(parents=True, exist_ok=True)
    (test_dir / "untracked_building" / "p.jpg").write_bytes(create_test_image_bytes())

    # 2. Nested subdirectory inside reference courtyard/
    (ref_dir / "courtyard" / "sub_angle").mkdir(parents=True, exist_ok=True)

    # 3. Unsupported non-image file (.txt)
    (calib_dir / "courtyard" / "participant_notes.txt").write_text("Collected at noon")

    # 4. Empty location folder in reference
    (ref_dir / "empty_pavilion").mkdir(parents=True, exist_ok=True)

    report = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)

    assert report.audit_passed is False
    assert "untracked_building" in report.splits["test"].unexpected_subdirectories
    assert "sub_angle" in report.splits["reference"].locations["courtyard"].unexpected_subdirectories
    assert "participant_notes.txt" in report.splits["calibration"].locations["courtyard"].unsupported_files
    assert "empty_pavilion" in report.splits["reference"].empty_locations

    issues_str = " ".join(report.issues)
    assert "Unexpected location folder in test split: 'untracked_building'" in issues_str
    assert "Unexpected nested subdirectory in reference location 'courtyard': 'sub_angle'" in issues_str
    assert "Unsupported file format in calibration 'courtyard': 'participant_notes.txt'" in issues_str
    assert "Reference location folder 'empty_pavilion' is empty" in issues_str


def test_audit_missing_split_directory(tmp_path: Path):
    """Missing top-level split directory fails audit cleanly."""
    ref_dir = tmp_path / "non_existent_ref"
    calib_dir = tmp_path / "non_existent_cal"
    test_dir = tmp_path / "non_existent_tst"

    report = audit_dataset(reference_dir=ref_dir, calibration_dir=calib_dir, test_dir=test_dir)
    assert report.audit_passed is False
    assert any("Reference directory does not exist" in i for i in report.issues)
    assert any("Calibration directory does not exist" in i for i in report.issues)
    assert any("Test directory does not exist" in i for i in report.issues)


def test_audit_cli_config_file_integration(tmp_path: Path):
    """CLI loads configuration from JSON file and applies custom thresholds."""
    locations = ["gym"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"gym": 3},
        calib_counts={"gym": 6},
        test_counts={"gym": 6},
        calib_unknown_count=12,
        test_unknown_count=12,
    )

    cfg = CampusHuntConfig()
    cfg.storage.reference_data_dir = ref_dir
    cfg.storage.calibration_data_dir = calib_dir
    cfg.storage.test_data_dir = test_dir
    cfg.scoring.min_calibration_samples_per_location = 6
    cfg.scoring.min_calibration_unknown_samples = 12
    cfg.scoring.min_test_samples_per_location = 6
    cfg.scoring.min_test_unknown_samples = 12

    config_file = tmp_path / "custom_config.json"
    cfg.to_json_file(config_file)

    report_file = tmp_path / "cli_audit_report.json"
    exit_code = audit_cli_main([
        "--config", str(config_file),
        "--output-report", str(report_file),
    ])

    assert exit_code == 0
    assert report_file.exists()
    report_data = json.loads(report_file.read_text(encoding="utf-8"))
    assert report_data["audit_passed"] is True
    assert report_data["configuration"]["min_calibration_samples_per_location"] == 6
    assert "accuracy" not in report_data


def test_audit_allow_missing_unknown_provisional_pass(tmp_path: Path):
    """Audit without unknown folders fails by default, but passes as provisional when allow_missing_unknown is True."""
    locations = ["gym", "library"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"gym": 3, "library": 3},
        calib_counts={"gym": 5, "library": 5},
        test_counts={"gym": 5, "library": 5},
        calib_unknown_count=0,
        test_unknown_count=0,
    )

    # 1. Default strict audit fails closed
    strict_report = audit_dataset(
        reference_dir=ref_dir,
        calibration_dir=calib_dir,
        test_dir=test_dir,
        allow_missing_unknown=False,
    )
    assert strict_report.audit_passed is False
    assert any("Calibration split is missing required 'unknown/' distractor folder" in i for i in strict_report.issues)
    assert any("Test split is missing required 'unknown/' distractor folder" in i for i in strict_report.issues)

    # 2. Provisional mode passes
    prov_report = audit_dataset(
        reference_dir=ref_dir,
        calibration_dir=calib_dir,
        test_dir=test_dir,
        allow_missing_unknown=True,
    )
    assert prov_report.audit_passed is True
    assert prov_report.provisional_no_unknown is True
    assert prov_report.allow_missing_unknown is True
    assert prov_report.issues == []
    summary_text = prov_report.format_text_summary()
    assert "STRUCTURALLY SPLIT (UNKNOWN EVALUATION UNAVAILABLE - PROVISIONAL)" in summary_text


def test_audit_cli_allow_missing_unknown_integration(tmp_path: Path):
    """CLI --allow-missing-unknown flag produces passing provisional report with exit code 0."""
    locations = ["courtyard"]
    ref_dir, calib_dir, test_dir = create_split_dataset(
        base_dir=tmp_path,
        locations=locations,
        ref_counts={"courtyard": 2},
        calib_counts={"courtyard": 5},
        test_counts={"courtyard": 5},
        calib_unknown_count=0,
        test_unknown_count=0,
    )

    report_file = tmp_path / "prov_audit_report.json"
    exit_code = audit_cli_main([
        "--reference-dir", str(ref_dir),
        "--calibration-dir", str(calib_dir),
        "--test-dir", str(test_dir),
        "--allow-missing-unknown",
        "--output-report", str(report_file),
    ])

    assert exit_code == 0
    assert report_file.exists()
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert data["audit_passed"] is True
    assert data["provisional_no_unknown"] is True
    assert data["allow_missing_unknown"] is True


