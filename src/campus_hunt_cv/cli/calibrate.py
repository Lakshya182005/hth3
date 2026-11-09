"""Validation and threshold calibration script for campus hunt verification."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from campus_hunt_cv.config import (
    CalibrationMetadata,
    CalibrationMetrics,
    CampusHuntConfig,
    DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION,
    DEFAULT_MIN_TEST_UNKNOWN_SAMPLES,
    ModelConfig,
    ScoringConfig,
    StorageConfig,
    TestEvaluationMetrics,
)
from campus_hunt_cv.decision import DecisionEngine
from campus_hunt_cv.embeddings_store import ReferenceEmbeddingsStore
from campus_hunt_cv.image_validation import validate_and_decode_image
from campus_hunt_cv.model import OpenCLIPLocationEmbedder
from campus_hunt_cv.similarity import SimilarityScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("calibrate")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}


class SplitDataset:
    """Loads and computes scores for photos in a labeled split directory."""

    def __init__(
        self,
        dataset_dir: Path,
        store: ReferenceEmbeddingsStore,
        embedder: OpenCLIPLocationEmbedder,
        scorer: SimilarityScorer,
        split_name: str = "dataset",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.store = store
        self.embedder = embedder
        self.scorer = scorer
        self.split_name = split_name
        self.samples: List[Dict[str, Any]] = []
        self.sample_counts: Dict[str, int] = defaultdict(int)

    def load_and_score(self) -> bool:
        """Scan directory, embed images, and compute similarity scores."""
        if not self.dataset_dir.exists():
            return False

        subdirs = [d for d in self.dataset_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not subdirs:
            return False

        logger.info("Scanning %s directory '%s'...", self.split_name, self.dataset_dir)

        total_images = 0
        for sdir in sorted(subdirs):
            actual_label = sdir.name
            img_files = [f for f in sdir.iterdir() if f.suffix in SUPPORTED_EXTENSIONS]

            for img_path in sorted(img_files):
                try:
                    with open(img_path, "rb") as f:
                        data = f.read()
                    img = validate_and_decode_image(data)
                    emb = self.embedder.embed_image(img)
                    scores = self.scorer.score_all_locations(
                        query_embedding=emb,
                        embeddings_by_location=self.store.all_embeddings_by_location,
                    )
                    self.samples.append({
                        "actual": actual_label,
                        "file": img_path.name,
                        "scores": scores,
                    })
                    self.sample_counts[actual_label] += 1
                    total_images += 1
                except Exception as e:
                    logger.error("Failed processing %s image '%s': %s", self.split_name, img_path, str(e))

        logger.info("Successfully evaluated %d %s captures.", total_images, self.split_name)
        return total_images > 0


def evaluate_thresholds(
    samples: List[Dict[str, Any]],
    store: ReferenceEmbeddingsStore,
    scorer: SimilarityScorer,
    base_scoring_config: ScoringConfig,
    match_threshold: float,
    margin_threshold: float,
    mismatch_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate performance metrics for a candidate threshold pair using real reference counts."""
    scoring_cfg = base_scoring_config.model_copy(update={
        "calibrated": True,
        "match_threshold": match_threshold,
        "margin_threshold": margin_threshold,
        "mismatch_threshold": mismatch_threshold,
    })
    engine = DecisionEngine(config=scoring_cfg)
    all_locations = store.locations

    correct_trials = 0
    correct_accepts = 0

    wrong_trials = 0
    wrong_false_accepts = 0

    unknown_trials = 0
    unknown_rejections = 0

    confusion_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        actual = sample["actual"]
        scores = sample["scores"]
        is_unknown = (actual == "unknown")

        if is_unknown:
            for target_loc in all_locations:
                unknown_trials += 1
                (
                    target_score,
                    best_match_id,
                    best_match_score,
                    competing_id,
                    competing_score,
                    margin,
                ) = scorer.extract_ranking_and_margin(scores, target_location_id=target_loc)

                # Use actual reference count from embeddings store
                real_ref_count = store.get_location_count(target_loc)

                decision, _ = engine.decide(
                    target_location_id=target_loc,
                    target_score=target_score,
                    best_matching_location_id=best_match_id,
                    best_matching_score=best_match_score,
                    best_competing_location_id=competing_id,
                    best_competing_score=competing_score,
                    margin=margin,
                    target_ref_count=real_ref_count,
                )

                if decision == "strong_match":
                    confusion_counts["unknown"][target_loc] += 1
                else:
                    unknown_rejections += 1
            continue

        if actual in all_locations:
            correct_trials += 1
            (
                target_score,
                best_match_id,
                best_match_score,
                competing_id,
                competing_score,
                margin,
            ) = scorer.extract_ranking_and_margin(scores, target_location_id=actual)

            # Use actual reference count from embeddings store
            real_ref_count = store.get_location_count(actual)

            decision, _ = engine.decide(
                target_location_id=actual,
                target_score=target_score,
                best_matching_location_id=best_match_id,
                best_matching_score=best_match_score,
                best_competing_location_id=competing_id,
                best_competing_score=competing_score,
                margin=margin,
                target_ref_count=real_ref_count,
            )

            predicted_label = best_match_id if decision == "strong_match" else f"unaccepted_{decision}"
            confusion_counts[actual][predicted_label] += 1

            if decision == "strong_match":
                correct_accepts += 1

            # Cross-location trials: participant claims wrong location L' != actual
            for wrong_loc in all_locations:
                if wrong_loc == actual:
                    continue
                wrong_trials += 1
                (
                    w_target_score,
                    w_best_id,
                    w_best_score,
                    w_comp_id,
                    w_comp_score,
                    w_margin,
                ) = scorer.extract_ranking_and_margin(scores, target_location_id=wrong_loc)

                wrong_ref_count = store.get_location_count(wrong_loc)

                w_decision, _ = engine.decide(
                    target_location_id=wrong_loc,
                    target_score=w_target_score,
                    best_matching_location_id=w_best_id,
                    best_matching_score=w_best_score,
                    best_competing_location_id=w_comp_id,
                    best_competing_score=w_comp_score,
                    margin=w_margin,
                    target_ref_count=wrong_ref_count,
                )

                if w_decision == "strong_match":
                    wrong_false_accepts += 1

    accept_rate = (correct_accepts / correct_trials) if correct_trials > 0 else 0.0
    far = (wrong_false_accepts / wrong_trials) if wrong_trials > 0 else 0.0
    if unknown_trials > 0:
        unk_rej_rate: Optional[float] = unknown_rejections / unknown_trials
        unk_trials_val: Optional[int] = unknown_trials
        unk_rejections_val: Optional[int] = unknown_rejections
    else:
        unk_rej_rate = None
        unk_trials_val = None
        unk_rejections_val = None

    return {
        "match_threshold": match_threshold,
        "margin_threshold": margin_threshold,
        "correct_accept_rate": accept_rate,
        "wrong_false_accept_rate": far,
        "unknown_rejection_rate": unk_rej_rate,
        "correct_trials": correct_trials,
        "correct_accepts": correct_accepts,
        "wrong_trials": wrong_trials,
        "wrong_false_accepts": wrong_false_accepts,
        "unknown_trials": unk_trials_val,
        "unknown_rejections": unk_rejections_val,
        "confusion_counts": confusion_counts,
    }


def print_confusion_matrix(
    confusion_counts: Dict[str, Dict[str, int]],
    locations: List[str],
    title: str = "PER-LOCATION CONFUSION MATRIX",
) -> None:
    """Display formatted per-location confusion matrix."""
    header_cols = locations + ["rejected/ambiguous"]
    col_width = 16

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Actual Location':<{col_width}} | " + " | ".join(f"{col:>{col_width}}" for col in header_cols))
    print("-" * 80)

    all_actuals = sorted(list(confusion_counts.keys()))
    for act in all_actuals:
        row_str = f"{act:<{col_width}} | "
        vals = []
        rejected_count = 0
        for col in locations:
            cnt = confusion_counts[act].get(col, 0)
            vals.append(f"{cnt:>{col_width}}")
        for k, cnt in confusion_counts[act].items():
            if k.startswith("unaccepted_"):
                rejected_count += cnt
        vals.append(f"{rejected_count:>{col_width}}")
        row_str += " | ".join(vals)
        print(row_str)
    print("=" * 80 + "\n")


def check_split_data_adequacy(
    store: ReferenceEmbeddingsStore,
    sample_counts: Dict[str, int],
    min_samples_per_location: int = DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    min_unknown_samples: int = DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    allow_missing_unknown: bool = False,
) -> Tuple[bool, List[str]]:
    """Verify that split data contains sufficient samples across all hunt locations and unknown."""
    missing: List[str] = []
    for loc in store.locations:
        count = sample_counts.get(loc, 0)
        if count < min_samples_per_location:
            missing.append(f"{loc} (count={count} < {min_samples_per_location})")

    if not allow_missing_unknown:
        unknown_count = sample_counts.get("unknown", 0)
        if unknown_count < min_unknown_samples:
            missing.append(f"unknown (count={unknown_count} < {min_unknown_samples})")

    return (len(missing) == 0, missing)


def check_calibration_data_adequacy(
    store: ReferenceEmbeddingsStore,
    sample_counts: Dict[str, int],
    min_samples_per_location: int = DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION,
    min_unknown_samples: int = DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES,
    allow_missing_unknown: bool = False,
) -> Tuple[bool, List[str]]:
    """Backward-compatible wrapper for calibration adequacy check."""
    return check_split_data_adequacy(
        store=store,
        sample_counts=sample_counts,
        min_samples_per_location=min_samples_per_location,
        min_unknown_samples=min_unknown_samples,
        allow_missing_unknown=allow_missing_unknown,
    )


def run_calibration(
    calibration_dir: Path,
    test_dir: Path,
    embeddings_npz: Path,
    metadata_json: Path,
    output_config: Optional[Path] = None,
    diagnostic_report: Optional[Path] = None,
    candidate_match_thresholds: Optional[List[float]] = None,
    candidate_margins: Optional[List[float]] = None,
    max_far: Optional[float] = None,
    min_unknown_rejection: Optional[float] = None,
    min_calib_samples_per_location: Optional[int] = None,
    min_calib_unknown_samples: Optional[int] = None,
    min_test_samples_per_location: Optional[int] = None,
    min_test_unknown_samples: Optional[int] = None,
    device: str = "cpu",
    base_config: Optional[CampusHuntConfig] = None,
    allow_missing_unknown: bool = False,
) -> bool:
    """Run threshold calibration on calibration data, then evaluate on test data.

    Fails closed:
    - If calibration data is inadequate, fails without creating a config.
    - If no threshold candidate meets all safety constraints, fails without writing/overwriting config.
    - If test data is unavailable or incomplete, marks status accordingly and does not present partial metrics.
    """
    if not embeddings_npz.exists() or not metadata_json.exists():
        logger.error(
            "Reference embeddings not found ('%s', '%s').\n"
            "Run `campus-hunt-build-embeddings` first before calibrating.",
            embeddings_npz,
            metadata_json,
        )
        return False

    cfg = base_config or CampusHuntConfig.from_env()
    effective_max_far = max_far if max_far is not None else cfg.scoring.target_max_far
    effective_min_unknown_rejection = (
        min_unknown_rejection if min_unknown_rejection is not None else cfg.scoring.target_min_unknown_rejection
    )
    effective_min_calib_samples = (
        min_calib_samples_per_location
        if min_calib_samples_per_location is not None
        else cfg.scoring.min_calibration_samples_per_location
    )
    effective_min_calib_unknown = (
        min_calib_unknown_samples
        if min_calib_unknown_samples is not None
        else cfg.scoring.min_calibration_unknown_samples
    )
    effective_min_test_samples = (
        min_test_samples_per_location
        if min_test_samples_per_location is not None
        else cfg.scoring.min_test_samples_per_location
    )
    effective_min_test_unknown = (
        min_test_unknown_samples
        if min_test_unknown_samples is not None
        else cfg.scoring.min_test_unknown_samples
    )

    model_name = cfg.model.model_name
    pretrained = cfg.model.pretrained

    is_calib_provisional = (
        effective_min_calib_samples < DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION
        or effective_min_calib_unknown < DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES
    )
    is_test_provisional = (
        effective_min_test_samples < DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION
        or effective_min_test_unknown < DEFAULT_MIN_TEST_UNKNOWN_SAMPLES
    )

    # Check if split manifest marks dataset split as provisional / single-collection or missing unknown
    manifest_paths = [
        calibration_dir.parent / "split_manifest.json",
    ]
    manifest_provisional_reason: Optional[str] = None
    for mp in manifest_paths:
        if mp.exists() and mp.is_file():
            try:
                manifest_dict = json.loads(mp.read_text(encoding="utf-8"))
                if manifest_dict.get("allow_missing_unknown", False):
                    allow_missing_unknown = True
                if manifest_dict.get("provisional", False) or not manifest_dict.get("independence_verifiable", True):
                    is_calib_provisional = True
                    is_test_provisional = True
                    manifest_provisional_reason = manifest_dict.get(
                        "status_reason",
                        "Dataset split originates from a single photo collection session; independence is unverified.",
                    )
                    break
            except Exception:
                pass

    if allow_missing_unknown:
        is_calib_provisional = True
        is_test_provisional = True
        if manifest_provisional_reason is None:
            manifest_provisional_reason = (
                "Calibrated in provisional mode without unknown/distractor captures (--allow-missing-unknown). "
                "Evaluated positive matches and inter-checkpoint discrimination only; unknown rejection is unmeasured."
            )


    if is_calib_provisional:
        warning_msg = (
            f"PROVISIONAL CALIBRATION WARNING: Calibration sample safeguards were manually lowered below "
            f"recommended project defaults (requires >= {DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION} per location, "
            f">= {DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES} unknown; configured: {effective_min_calib_samples} per location, "
            f"{effective_min_calib_unknown} unknown).\n"
            "Resulting calibration status will be marked as PROVISIONAL rather than implying strong evidence."
        )
        logger.warning(warning_msg)
        print("\n" + "!" * 80)
        print("WARNING: PROVISIONAL CALIBRATION SAFEGUARDS IN EFFECT")
        print(warning_msg)
        print("!" * 80 + "\n")

    if is_test_provisional:
        warning_msg = (
            f"PROVISIONAL TEST SPLIT WARNING: Test sample safeguards were manually lowered below "
            f"recommended project defaults (requires >= {DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION} per location, "
            f">= {DEFAULT_MIN_TEST_UNKNOWN_SAMPLES} unknown; configured: {effective_min_test_samples} per location, "
            f"{effective_min_test_unknown} unknown).\n"
            "Resulting test evaluation status will be marked as PROVISIONAL rather than completed."
        )
        logger.warning(warning_msg)
        print("\n" + "!" * 80)
        print("WARNING: PROVISIONAL TEST SPLIT SAFEGUARDS IN EFFECT")
        print(warning_msg)
        print("!" * 80 + "\n")

    calib_minimums_used = {
        "min_calibration_samples_per_location": effective_min_calib_samples,
        "min_calibration_unknown_samples": effective_min_calib_unknown,
        "min_test_samples_per_location": effective_min_test_samples,
        "min_test_unknown_samples": effective_min_test_unknown,
    }

    try:
        store = ReferenceEmbeddingsStore.load(
            npz_path=embeddings_npz,
            metadata_path=metadata_json,
            expected_model_version=f"{model_name}/{pretrained}",
        )
    except Exception as e:
        logger.error("Failed to load reference embeddings: %s", str(e))
        return False

    model_cfg = ModelConfig(
        model_name=store.metadata.get("model_name", model_name),
        pretrained=store.metadata.get("pretrained", pretrained),
        device=device,
    )
    embedder = OpenCLIPLocationEmbedder(config=model_cfg)
    scorer = SimilarityScorer(config=cfg.scoring)

    # 1. Load Calibration Dataset
    calib_dataset = SplitDataset(
        dataset_dir=calibration_dir,
        store=store,
        embedder=embedder,
        scorer=scorer,
        split_name="calibration",
    )

    if not calib_dataset.load_and_score():
        logger.error(
            "No valid evaluation photos found in calibration directory '%s'.\n"
            "Contract layout:\n"
            "  data/calibration/<actual_location_id>/*.jpg\n"
            "  data/calibration/unknown/*.jpg\n"
            "Please collect real labeled calibration photos before calibrating thresholds.",
            calibration_dir,
        )
        return False

    # Check adequacy of calibration captures
    is_adequate, missing_locs = check_split_data_adequacy(
        store=store,
        sample_counts=calib_dataset.sample_counts,
        min_samples_per_location=effective_min_calib_samples,
        min_unknown_samples=effective_min_calib_unknown,
        allow_missing_unknown=allow_missing_unknown,
    )

    if not is_adequate:
        logger.error(
            "Inadequate calibration data! Missing captures for: %s.\n"
            "Thresholds cannot be calibrated without representative positive captures for all locations "
            "and distractor photos in 'unknown/'. Calibrated config will NOT be written.",
            ", ".join(missing_locs),
        )
        if diagnostic_report:
            diag_path = Path(diagnostic_report)
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_data = {
                "status": "calibration_failed_inadequate_data",
                "deployable": False,
                "calibrated": False,
                "is_provisional": is_calib_provisional,
                "minimums_used": calib_minimums_used,
                "calibration_sample_counts": dict(calib_dataset.sample_counts),
                "missing_categories": missing_locs,
                "model_version": store.model_version,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag_data, f, indent=2)
            logger.info("Saved non-deployable diagnostic report to '%s'.", diag_path)

        return False

    match_candidates = candidate_match_thresholds or [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    margin_candidates = candidate_margins or [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]

    print("\n" + "=" * 80)
    print("CALIBRATION SET THRESHOLD & MARGIN SEARCH (data/calibration/)")
    constraint_unk_str = (
        f"Unknown Rejection >= {effective_min_unknown_rejection * 100:.1f}%"
        if not allow_missing_unknown
        else "Unknown Rejection: N/A (provisional mode)"
    )
    print(f"Constraints: FAR <= {effective_max_far * 100:.2f}%, {constraint_unk_str}")
    print("=" * 80)
    print(
        f"{'Match Thresh':<14} {'Margin':<10} {'Accept Rate':<14} {'False Accept (FAR)':<20} {'Unknown Reject':<16} {'Meets Constraints'}"
    )
    print("-" * 80)

    best_config: Optional[Dict[str, Any]] = None
    best_accept_rate = -1.0
    all_results: List[Dict[str, Any]] = []

    for m_th in match_candidates:
        for mg_th in margin_candidates:
            res = evaluate_thresholds(
                samples=calib_dataset.samples,
                store=store,
                scorer=scorer,
                base_scoring_config=cfg.scoring,
                match_threshold=m_th,
                margin_threshold=mg_th,
                mismatch_threshold=None,
            )
            all_results.append(res)

            if res["unknown_rejection_rate"] is not None:
                meets_constraints = (
                    res["wrong_false_accept_rate"] <= effective_max_far
                    and res["unknown_rejection_rate"] >= effective_min_unknown_rejection
                )
                unk_val_str = f"{res['unknown_rejection_rate'] * 100:>14.1f}%   "
            else:
                meets_constraints = (
                    res["wrong_false_accept_rate"] <= effective_max_far
                )
                unk_val_str = "           N/A   "

            status_str = "YES" if meets_constraints else "NO"

            print(
                f"{res['match_threshold']:<14.2f} "
                f"{res['margin_threshold']:<10.2f} "
                f"{res['correct_accept_rate'] * 100:>10.1f}%   "
                f"{res['wrong_false_accept_rate'] * 100:>14.2f}%   "
                f"{unk_val_str}"
                f"{status_str:>17}"
            )

            if meets_constraints:
                if res["correct_accept_rate"] > best_accept_rate:
                    best_accept_rate = res["correct_accept_rate"]
                    best_config = res

    print("=" * 80 + "\n")

    # FIX B: FAIL CLOSED if no candidate meets all constraints
    if best_config is None:
        logger.error(
            "Calibration FAILED: No candidate threshold pair satisfied all safety constraints:\n"
            "  - Maximum False Acceptance Rate (FAR) <= %.2f%%\n"
            "  - Minimum Unknown Rejection Rate >= %.1f%%\n"
            "Tested %d candidate threshold combinations.\n"
            "No active calibrated configuration will be created or overwritten.",
            effective_max_far * 100,
            effective_min_unknown_rejection * 100,
            len(all_results),
        )

        print("!" * 80)
        print("CALIBRATION FAILURE: NO CANDIDATE MET REQUIRED SAFETY CONSTRAINTS")
        print("!" * 80)
        print(f"Required: FAR <= {effective_max_far * 100:.2f}%, Unknown Rejection >= {effective_min_unknown_rejection * 100:.1f}%")
        print("Closest candidates evaluated (for diagnosis):")
        sorted_candidates = sorted(
            all_results,
            key=lambda x: (
                x["wrong_false_accept_rate"],
                -(x["unknown_rejection_rate"] if x["unknown_rejection_rate"] is not None else 0.0),
            ),
        )
        for c in sorted_candidates[:5]:
            rej_str = f"{c['unknown_rejection_rate']*100:.1f}%" if c['unknown_rejection_rate'] is not None else "N/A"
            print(
                f"  Match={c['match_threshold']:.2f}, Margin={c['margin_threshold']:.2f} -> "
                f"FAR={c['wrong_false_accept_rate']*100:.2f}%, Rejection={rej_str}, Accept={c['correct_accept_rate']*100:.1f}%"
            )
        print("!" * 80 + "\n")

        # Save diagnostic report if requested (explicitly non-deployable, calibrated=False)
        if diagnostic_report:
            diag_path = Path(diagnostic_report)
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_data = {
                "status": "calibration_failed_constraints_unmet",
                "deployable": False,
                "calibrated": False,
                "is_provisional": is_calib_provisional,
                "minimums_used": calib_minimums_used,
                "model_version": store.model_version,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "constraints": {
                    "max_far": effective_max_far,
                    "min_unknown_rejection": effective_min_unknown_rejection,
                },
                "total_candidates_tested": len(all_results),
                "calibration_sample_counts": dict(calib_dataset.sample_counts),
                "closest_candidates": [
                    {
                        "match_threshold": r["match_threshold"],
                        "margin_threshold": r["margin_threshold"],
                        "correct_accept_rate": round(r["correct_accept_rate"], 4),
                        "wrong_false_accept_rate": round(r["wrong_false_accept_rate"], 6),
                        "unknown_rejection_rate": round(r["unknown_rejection_rate"], 4),
                    }
                    for r in sorted_candidates[:10]
                ],
            }
            with open(diag_path, "w", encoding="utf-8") as f:
                json.dump(diag_data, f, indent=2)
            logger.info("Saved non-deployable diagnostic report to '%s'.", diag_path)

        return False

    print("SELECTED THRESHOLDS (from calibration data):")
    print(f"  Selected Match Threshold:      {best_config['match_threshold']:.3f}")
    print(f"  Selected Margin Threshold:     {best_config['margin_threshold']:.3f}")
    print(f"  Calibration Acceptance Rate:   {best_config['correct_accept_rate'] * 100:.1f}%")
    print(f"  Calibration Wrong Location FAR:{best_config['wrong_false_accept_rate'] * 100:.2f}%")
    calib_unk_disp = f"{best_config['unknown_rejection_rate'] * 100:.1f}%" if best_config["unknown_rejection_rate"] is not None else "N/A (Provisional)"
    print(f"  Calibration Unknown Rejection: {calib_unk_disp}\n")

    print_confusion_matrix(best_config["confusion_counts"], store.locations, title="CALIBRATION CONFUSION MATRIX")

    # 2. FIX A: Rigorous Hold-Out Test Split Evaluation
    test_dataset = SplitDataset(
        dataset_dir=test_dir,
        store=store,
        embedder=embedder,
        scorer=scorer,
        split_name="test",
    )
    has_test_files = test_dataset.load_and_score()

    test_eval_metadata: TestEvaluationMetrics
    test_minimums_used = {
        "min_test_samples_per_location": effective_min_test_samples,
        "min_test_unknown_samples": effective_min_test_unknown,
    }

    if not has_test_files:
        print("!" * 80)
        print("NOTICE: Final evaluation on independent test data is UNAVAILABLE.")
        print(f"No test photos found in '{test_dir}'.")
        print("Calibration metrics shown above must NOT be reported as independent test performance.")
        print("Collect separate hold-out test photos across all locations before deploying.")
        print("!" * 80 + "\n")

        all_missing = [f"{loc} (count=0 < {effective_min_test_samples})" for loc in store.locations]
        all_missing.append(f"unknown (count=0 < {effective_min_test_unknown})")

        test_eval_metadata = TestEvaluationMetrics(
            status="unavailable",
            reason=f"No independent test captures found in '{test_dir}'.",
            sample_counts=dict(test_dataset.sample_counts),
            missing_categories=all_missing,
            minimums_used=test_minimums_used,
            is_provisional=is_test_provisional,
            metrics=None,
        )
    else:
        # Check completeness of test split
        is_test_complete, missing_test = check_split_data_adequacy(
            store=store,
            sample_counts=test_dataset.sample_counts,
            min_samples_per_location=effective_min_test_samples,
            min_unknown_samples=effective_min_test_unknown,
            allow_missing_unknown=allow_missing_unknown,
        )

        if not is_test_complete:
            print("!" * 80)
            print("WARNING: Final evaluation on independent test data is INCOMPLETE.")
            print(f"Test directory '{test_dir}' contains captures, but required categories or counts are missing:")
            for m in missing_test:
                print(f"  - {m}")
            print("Partial test metrics will NOT be reported as completed independent test performance.")
            print("Collect complete hold-out test captures across all locations to evaluate generalization.")
            print("!" * 80 + "\n")

            test_eval_metadata = TestEvaluationMetrics(
                status="incomplete",
                reason=f"Test split is incomplete. Missing required categories or counts: {', '.join(missing_test)}",
                sample_counts=dict(test_dataset.sample_counts),
                missing_categories=missing_test,
                minimums_used=test_minimums_used,
                is_provisional=is_test_provisional,
                metrics=None,  # Do not present partial metrics as completed!
            )
        else:
            # Complete test split: evaluate selected thresholds ONCE
            test_res = evaluate_thresholds(
                samples=test_dataset.samples,
                store=store,
                scorer=scorer,
                base_scoring_config=cfg.scoring,
                match_threshold=best_config["match_threshold"],
                margin_threshold=best_config["margin_threshold"],
                mismatch_threshold=None,
            )

            test_title = (
                "PROVISIONAL TEST SET EVALUATION (Hold-out evaluation with lowered safeguards)"
                if is_test_provisional
                else "INDEPENDENT TEST SET EVALUATION (Unbiased hold-out generalization)"
            )
            print("\n" + "=" * 80)
            print(test_title)
            print("=" * 80)
            test_unk_disp = (
                f"{test_res['unknown_rejection_rate'] * 100:.1f}%"
                if test_res["unknown_rejection_rate"] is not None
                else "N/A (Provisional)"
            )
            print(f"  Test Correct Acceptance Rate:  {test_res['correct_accept_rate'] * 100:.1f}%")
            print(f"  Test Wrong Location FAR:       {test_res['wrong_false_accept_rate'] * 100:.2f}%")
            print(f"  Test Unknown Rejection Rate:   {test_unk_disp}")
            if is_test_provisional:
                print("  NOTICE: Safeguards lowered below standard; results are provisional and do not establish verified generalization.")
            print("=" * 80 + "\n")

            print_confusion_matrix(test_res["confusion_counts"], store.locations, title="INDEPENDENT TEST CONFUSION MATRIX")

            test_status_val: Literal["completed", "provisional"] = "provisional" if is_test_provisional else "completed"
            test_reason_val: Optional[str] = (
                manifest_provisional_reason
                or "Evaluated with manually lowered sample safeguards. Results are provisional and do not guarantee generalization."
                if is_test_provisional
                else None
            )

            test_eval_metadata = TestEvaluationMetrics(
                status=test_status_val,
                reason=test_reason_val,
                sample_counts=dict(test_dataset.sample_counts),
                missing_categories=None,
                minimums_used=test_minimums_used,
                is_provisional=is_test_provisional,
                metrics=CalibrationMetrics(
                    correct_accept_rate=round(test_res["correct_accept_rate"], 4),
                    wrong_false_accept_rate=round(test_res["wrong_false_accept_rate"], 6),
                    unknown_rejection_rate=(
                        round(test_res["unknown_rejection_rate"], 4)
                        if test_res["unknown_rejection_rate"] is not None
                        else None
                    ),
                    correct_trials=test_res["correct_trials"],
                    correct_accepts=test_res["correct_accepts"],
                    wrong_trials=test_res["wrong_trials"],
                    wrong_false_accepts=test_res["wrong_false_accepts"],
                    unknown_trials=test_res["unknown_trials"],
                    unknown_rejections=test_res["unknown_rejections"],
                ),
            )

    # 3. Write compatible CampusHuntConfig JSON atomically
    if output_config:
        output_config = Path(output_config)
        output_config.parent.mkdir(parents=True, exist_ok=True)

        calib_status_val: Literal["calibrated", "provisional"] = "provisional" if is_calib_provisional else "calibrated"
        calib_warning_val: Optional[str] = (
            manifest_provisional_reason
            or (
                f"Sample safeguards were manually set below project standards "
                f"({DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION} per location, "
                f"{DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES} unknown). "
                "Thresholds are provisional and require further real campus data collection."
            )
            if is_calib_provisional
            else None
        )

        calib_cfg = CampusHuntConfig(
            environment="production",
            model=ModelConfig(
                model_name=store.metadata.get("model_name", model_name),
                pretrained=store.metadata.get("pretrained", pretrained),
                device=device,
            ),
            image_limits=cfg.image_limits,
            scoring=ScoringConfig(
                aggregation_method=cfg.scoring.aggregation_method,
                top_k=cfg.scoring.top_k,
                calibrated=True,
                match_threshold=round(best_config["match_threshold"], 4),
                margin_threshold=round(best_config["margin_threshold"], 4),
                mismatch_threshold=None,
                min_reference_images_per_location=cfg.scoring.min_reference_images_per_location,
                min_calibration_samples_per_location=effective_min_calib_samples,
                min_calibration_unknown_samples=effective_min_calib_unknown,
                min_test_samples_per_location=effective_min_test_samples,
                min_test_unknown_samples=effective_min_test_unknown,
                target_max_far=effective_max_far,
                target_min_unknown_rejection=effective_min_unknown_rejection,
                allow_unvalidated_auto_accept=cfg.scoring.allow_unvalidated_auto_accept,
            ),
            storage=StorageConfig(
                reference_data_dir=cfg.storage.reference_data_dir,
                calibration_data_dir=calibration_dir,
                test_data_dir=test_dir,
                embeddings_path=embeddings_npz,
                metadata_path=metadata_json,
            ),
            calibration_metadata=CalibrationMetadata(
                status=calib_status_val,
                is_provisional=is_calib_provisional,
                warning=calib_warning_val,
                model_version=store.model_version,
                calibrated_at=datetime.now(timezone.utc).isoformat(),
                minimums_used=calib_minimums_used,
                calibration_sample_counts=dict(calib_dataset.sample_counts),
                calibration_metrics=CalibrationMetrics(
                    correct_accept_rate=round(best_config["correct_accept_rate"], 4),
                    wrong_false_accept_rate=round(best_config["wrong_false_accept_rate"], 6),
                    unknown_rejection_rate=(
                        round(best_config["unknown_rejection_rate"], 4)
                        if best_config["unknown_rejection_rate"] is not None
                        else None
                    ),
                    correct_trials=best_config["correct_trials"],
                    correct_accepts=best_config["correct_accepts"],
                    wrong_trials=best_config["wrong_trials"],
                    wrong_false_accepts=best_config["wrong_false_accepts"],
                    unknown_trials=best_config["unknown_trials"],
                    unknown_rejections=best_config["unknown_rejections"],
                ),
                test_evaluation=test_eval_metadata,
            ),
        )

        # Atomic write: write to temp file first, then replace
        tmp_output = output_config.with_suffix(f".tmp_{os.getpid()}_{datetime.now().strftime('%f')}")
        calib_cfg.to_json_file(tmp_output)
        tmp_output.replace(output_config)

        logger.info("Saved compatible calibrated configuration atomically to '%s'.", output_config)

    return True


def main() -> None:
    """CLI entrypoint for calibration."""
    cfg = CampusHuntConfig.from_env()

    parser = argparse.ArgumentParser(
        description="Calibrate similarity and margin thresholds using labeled calibration captures."
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help="Path to labeled calibration images directory (default: data/calibration)",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Alias for --calibration-dir (backwards compatibility)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=cfg.storage.test_data_dir,
        help="Path to independent test images directory (default: data/test)",
    )
    parser.add_argument(
        "--embeddings-npz",
        type=Path,
        default=cfg.storage.embeddings_path,
        help="Path to reference embeddings NPZ file",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=cfg.storage.metadata_path,
        help="Path to reference embeddings JSON metadata",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("calibrated_thresholds.json"),
        help="Path to save recommended calibrated configuration",
    )
    parser.add_argument(
        "--diagnostic-report",
        type=Path,
        default=None,
        help="Optional path to save failure diagnostic report if constraints cannot be met",
    )
    parser.add_argument(
        "--max-far",
        type=float,
        default=None,
        help="Maximum tolerable False Acceptance Rate (default: from config, e.g. 0.02 = 2%%)",
    )
    parser.add_argument(
        "--min-unknown-rejection",
        type=float,
        default=None,
        help="Minimum tolerable Unknown Rejection Rate (default: from config, e.g. 0.90 = 90%%)",
    )
    parser.add_argument(
        "--min-calib-samples",
        type=int,
        default=None,
        help=f"Minimum calibration captures per location (default: {DEFAULT_MIN_CALIBRATION_SAMPLES_PER_LOCATION})",
    )
    parser.add_argument(
        "--min-calib-unknown",
        type=int,
        default=None,
        help=f"Minimum unknown/distractor calibration captures (default: {DEFAULT_MIN_CALIBRATION_UNKNOWN_SAMPLES})",
    )
    parser.add_argument(
        "--min-test-samples",
        type=int,
        default=None,
        help=f"Minimum test captures per location required for complete evaluation (default: {DEFAULT_MIN_TEST_SAMPLES_PER_LOCATION})",
    )
    parser.add_argument(
        "--min-test-unknown",
        type=int,
        default=None,
        help=f"Minimum unknown test captures required for complete evaluation (default: {DEFAULT_MIN_TEST_UNKNOWN_SAMPLES})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.model.device,
        help="Compute device for image embedding (default: cpu)",
    )
    parser.add_argument(
        "--allow-missing-unknown",
        action="store_true",
        default=False,
        help="Permit provisional calibration and test evaluation without unknown/distractor captures.",
    )
    parser.add_argument(
        "--allow-unvalidated-auto-accept",
        action="store_true",
        default=False,
        help="DANGER / DEVELOPMENT ONLY: Permit output config to auto-accept submissions even if provisional or test is incomplete",
    )

    args = parser.parse_args()

    if args.allow_unvalidated_auto_accept:
        cfg.scoring.allow_unvalidated_auto_accept = True
        logger.warning(
            "CRITICAL WARNING: --allow-unvalidated-auto-accept is active. "
            "The generated configuration will allow unvalidated auto-acceptance ONLY when "
            "CAMPUS_HUNT_ENVIRONMENT is explicitly set to 'development' or 'test'. "
            "This override is strictly blocked in production (the default runtime environment)."
        )

    # Resolve calibration directory
    calib_dir = args.calibration_dir or args.validation_dir or cfg.storage.calibration_data_dir
    if not calib_dir.exists() and cfg.storage.validation_data_dir.exists():
        calib_dir = cfg.storage.validation_data_dir

    success = run_calibration(
        calibration_dir=calib_dir,
        test_dir=args.test_dir,
        embeddings_npz=args.embeddings_npz,
        metadata_json=args.metadata_json,
        output_config=args.output_config,
        diagnostic_report=args.diagnostic_report,
        max_far=args.max_far,
        min_unknown_rejection=args.min_unknown_rejection,
        min_calib_samples_per_location=args.min_calib_samples,
        min_calib_unknown_samples=args.min_calib_unknown,
        min_test_samples_per_location=args.min_test_samples,
        min_test_unknown_samples=args.min_test_unknown,
        device=args.device,
        base_config=cfg,
        allow_missing_unknown=args.allow_missing_unknown,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
