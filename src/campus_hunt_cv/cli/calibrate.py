"""Validation and threshold calibration script for campus hunt verification."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from campus_hunt_cv.config import CampusHuntConfig, ModelConfig, ScoringConfig
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


class ValidationDataset:
    """Loads and computes scores for all labeled validation images."""

    def __init__(
        self,
        validation_dir: Path,
        store: ReferenceEmbeddingsStore,
        embedder: OpenCLIPLocationEmbedder,
        scorer: SimilarityScorer,
    ):
        self.validation_dir = Path(validation_dir)
        self.store = store
        self.embedder = embedder
        self.scorer = scorer
        # List of dicts: {"actual": str, "file": str, "scores": Dict[str, float]}
        self.samples: List[Dict[str, Any]] = []

    def load_and_score(self) -> bool:
        """Scan validation directory, embed images, and compute similarity scores."""
        if not self.validation_dir.exists():
            logger.error("Validation directory does not exist: %s", self.validation_dir)
            return False

        subdirs = [d for d in self.validation_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not subdirs:
            logger.warning("No subdirectories found in '%s'.", self.validation_dir)
            return False

        logger.info("Scanning validation directory '%s'...", self.validation_dir)

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
                    total_images += 1
                except Exception as e:
                    logger.error("Failed processing validation image '%s': %s", img_path, str(e))

        logger.info("Successfully evaluated %d validation captures.", total_images)
        return total_images > 0


def evaluate_thresholds(
    samples: List[Dict[str, Any]],
    all_locations: List[str],
    scorer: SimilarityScorer,
    match_threshold: float,
    margin_threshold: float,
    mismatch_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate performance metrics for a specific (match_threshold, margin_threshold) pair."""
    scoring_cfg = ScoringConfig(
        calibrated=True,
        match_threshold=match_threshold,
        margin_threshold=margin_threshold,
        mismatch_threshold=mismatch_threshold,
    )
    engine = DecisionEngine(config=scoring_cfg)

    correct_trials = 0
    correct_accepts = 0

    wrong_trials = 0
    wrong_false_accepts = 0

    unknown_trials = 0
    unknown_rejections = 0

    # For confusion matrix: actual -> predicted
    confusion_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sample in samples:
        actual = sample["actual"]
        scores = sample["scores"]
        is_unknown = (actual == "unknown")

        if is_unknown:
            # For unknown images, simulate testing against EVERY known hunt location
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

                decision, _ = engine.decide(
                    target_location_id=target_loc,
                    target_score=target_score,
                    best_matching_location_id=best_match_id,
                    best_matching_score=best_match_score,
                    best_competing_location_id=competing_id,
                    best_competing_score=competing_score,
                    margin=margin,
                    target_ref_count=3,
                )

                if decision == "strong_match":
                    confusion_counts["unknown"][target_loc] += 1
                else:
                    unknown_rejections += 1
            continue

        # Positive evaluation: participant submitted photo for their actual location
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

            decision, _ = engine.decide(
                target_location_id=actual,
                target_score=target_score,
                best_matching_location_id=best_match_id,
                best_matching_score=best_match_score,
                best_competing_location_id=competing_id,
                best_competing_score=competing_score,
                margin=margin,
                target_ref_count=3,
            )

            predicted_label = best_match_id if decision == "strong_match" else f"unaccepted_{decision}"
            confusion_counts[actual][predicted_label] += 1

            if decision == "strong_match":
                correct_accepts += 1

            # Cross-location trials: participant claims to be at wrong location L' != actual
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

                w_decision, _ = engine.decide(
                    target_location_id=wrong_loc,
                    target_score=w_target_score,
                    best_matching_location_id=w_best_id,
                    best_matching_score=w_best_score,
                    best_competing_location_id=w_comp_id,
                    best_competing_score=w_comp_score,
                    margin=w_margin,
                    target_ref_count=3,
                )

                if w_decision == "strong_match":
                    wrong_false_accepts += 1

    accept_rate = (correct_accepts / correct_trials) if correct_trials > 0 else 0.0
    far = (wrong_false_accepts / wrong_trials) if wrong_trials > 0 else 0.0
    unk_rej_rate = (unknown_rejections / unknown_trials) if unknown_trials > 0 else 1.0

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
        "unknown_trials": unknown_trials,
        "unknown_rejections": unknown_rejections,
        "confusion_counts": confusion_counts,
    }


def print_confusion_matrix(confusion_counts: Dict[str, Dict[str, int]], locations: List[str]) -> None:
    """Display formatted per-location confusion matrix."""
    header_cols = locations + ["rejected/ambiguous"]
    col_width = 16

    print("\n" + "=" * 80)
    print("PER-LOCATION CONFUSION MATRIX")
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


def run_calibration(
    validation_dir: Path,
    embeddings_npz: Path,
    metadata_json: Path,
    output_config: Optional[Path] = None,
    candidate_match_thresholds: Optional[List[float]] = None,
    candidate_margins: Optional[List[float]] = None,
    max_far: float = 0.02,
    device: str = "cpu",
) -> None:
    """Run threshold sweep and select optimal operating parameters."""
    if not embeddings_npz.exists() or not metadata_json.exists():
        logger.error(
            "Reference embeddings not found ('%s', '%s').\n"
            "Run `campus-hunt-build-embeddings` first before calibrating.",
            embeddings_npz,
            metadata_json,
        )
        return

    store = ReferenceEmbeddingsStore.load(npz_path=embeddings_npz, metadata_path=metadata_json)
    model_cfg = ModelConfig(
        model_name=store.metadata.get("model_name", "ViT-B-32"),
        pretrained=store.metadata.get("pretrained", "laion2b_s34b_b79k"),
        device=device,
    )
    embedder = OpenCLIPLocationEmbedder(config=model_cfg)
    scorer = SimilarityScorer()

    dataset = ValidationDataset(
        validation_dir=validation_dir,
        store=store,
        embedder=embedder,
        scorer=scorer,
    )

    loaded = dataset.load_and_score()
    if not loaded:
        logger.error(
            "No valid evaluation photos found in '%s'.\n"
            "Data directory contract:\n"
            "  data/validation/<actual_location_id>/*.jpg\n"
            "  data/validation/unknown/*.jpg\n"
            "Please collect real labeled validation photos before calibrating thresholds.",
            validation_dir,
        )
        return

    match_candidates = candidate_match_thresholds or [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    margin_candidates = candidate_margins or [0.00, 0.03, 0.05, 0.08, 0.10, 0.15]

    print("\n" + "=" * 80)
    print("THRESHOLD & MARGIN CALIBRATION GRID SEARCH")
    print("=" * 80)
    print(
        f"{'Match Thresh':<14} {'Margin':<10} {'Accept Rate':<14} {'False Accept (FAR)':<20} {'Unknown Reject':<16}"
    )
    print("-" * 80)

    best_config: Optional[Dict[str, Any]] = None
    best_accept_rate = -1.0
    all_results: List[Dict[str, Any]] = []

    for m_th in match_candidates:
        for mg_th in margin_candidates:
            res = evaluate_thresholds(
                samples=dataset.samples,
                all_locations=store.locations,
                scorer=scorer,
                match_threshold=m_th,
                margin_threshold=mg_th,
            )
            all_results.append(res)
            print(
                f"{res['match_threshold']:<14.2f} "
                f"{res['margin_threshold']:<10.2f} "
                f"{res['correct_accept_rate'] * 100:>10.1f}%   "
                f"{res['wrong_false_accept_rate'] * 100:>14.2f}%   "
                f"{res['unknown_rejection_rate'] * 100:>14.1f}%"
            )

            # Selection criteria: FAR <= max_far, high unknown reject, maximize accept rate
            if res["wrong_false_accept_rate"] <= max_far and res["unknown_rejection_rate"] >= 0.90:
                if res["correct_accept_rate"] > best_accept_rate:
                    best_accept_rate = res["correct_accept_rate"]
                    best_config = res

    print("=" * 80 + "\n")

    if best_config is None:
        logger.warning(
            "No candidate thresholds satisfied the strict constraint (FAR <= %.2f%%). "
            "Selecting safest threshold with lowest FAR.",
            max_far * 100,
        )
        best_config = min(all_results, key=lambda x: (x["wrong_false_accept_rate"], -x["correct_accept_rate"]))

    print("CALIBRATION RECOMMENDATION:")
    print(f"  Selected Match Threshold:  {best_config['match_threshold']:.3f}")
    print(f"  Selected Margin Threshold: {best_config['margin_threshold']:.3f}")
    print(f"  Correct Acceptance Rate:   {best_config['correct_accept_rate'] * 100:.1f}%")
    print(f"  Wrong Location FAR:        {best_config['wrong_false_accept_rate'] * 100:.2f}%")
    print(f"  Unknown Rejection Rate:    {best_config['unknown_rejection_rate'] * 100:.1f}%\n")

    print_confusion_matrix(best_config["confusion_counts"], store.locations)

    if output_config:
        calib_data = {
            "calibrated": True,
            "match_threshold": round(best_config["match_threshold"], 4),
            "margin_threshold": round(best_config["margin_threshold"], 4),
            "mismatch_threshold": 0.30,
            "model_version": store.model_version,
            "validation_metrics": {
                "correct_accept_rate": round(best_config["correct_accept_rate"], 4),
                "wrong_false_accept_rate": round(best_config["wrong_false_accept_rate"], 6),
                "unknown_rejection_rate": round(best_config["unknown_rejection_rate"], 4),
            },
        }
        with open(output_config, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=2)
        logger.info("Saved calibrated threshold configuration to '%s'.", output_config)


def main() -> None:
    """CLI entrypoint for calibration."""
    cfg = CampusHuntConfig.from_env()

    parser = argparse.ArgumentParser(
        description="Calibrate similarity and margin thresholds using labeled validation captures."
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=cfg.storage.validation_data_dir,
        help="Path to labeled validation images directory (default: data/validation)",
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
        help="Path to save recommended calibrated threshold configuration",
    )
    parser.add_argument(
        "--max-far",
        type=float,
        default=0.02,
        help="Maximum tolerable False Acceptance Rate (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=cfg.model.device,
        help="Compute device for validation image embedding (default: cpu)",
    )

    args = parser.parse_args()

    run_calibration(
        validation_dir=args.validation_dir,
        embeddings_npz=args.embeddings_npz,
        metadata_json=args.metadata_json,
        output_config=args.output_config,
        max_far=args.max_far,
        device=args.device,
    )


if __name__ == "__main__":
    main()
