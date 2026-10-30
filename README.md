# Campus Hunt CV: AI Location-Verification Component

A computer vision component for campus treasure hunts. It verifies participant-submitted photos against reference imagery for campus riddle locations using OpenCLIP feature embeddings, cosine similarity, top-$k$ mean aggregation, and calibrated threshold margins.

---

## 1. System Architecture & Scope

### Scope Boundary
- **Included**: Raw image byte validation, safe decoding, OpenCLIP normalized feature extraction, precomputed reference embeddings management, cosine similarity scoring with top-$k$ aggregation, margin calculation, conservative decision logic, threshold calibration, and independent test evaluation.
- **Excluded**: Web frontend, player authentication, database persistence, game progression logic, and external microservices.
- **No Location Permissions / Geofencing**: This system strictly analyzes visual content. It does **not** use GPS, geolocation coordinates, device telemetry, or QR codes.
- **Perceptual Match Only**: This component confirms whether a photo *appears to depict* a specific hunt location. It **never claims** that a photo proves physical presence or live capture (it does not detect printed photos or phone screen replays).

---

## 2. Directory Layout Contract

The pipeline expects data organized under the `data/` directory according to this exact layout:

```
data/
  reference/
    <location_id>/
      photo1.jpg
      photo2.jpg
      photo3.jpg
  calibration/
    <actual_location_id>/
      calib_photo1.jpg
      calib_photo2.jpg
    unknown/
      cafeteria.jpg
      dorm_hallway.jpg
      sidewalk.jpg
  test/
    <actual_location_id>/
      test_photo1.jpg
    unknown/
      off_campus.jpg
      parking_lot.jpg
```

### Data Collection & Leakage Prevention Rules:
1. **Three-Way Independent Split**:
   - `data/reference/`: Used exclusively to build precomputed embeddings for all hunt locations.
   - `data/calibration/`: Used exclusively by the calibration script to sweep candidate thresholds and margins.
   - `data/test/`: Strict hold-out dataset evaluated once to measure unbiased generalization performance.
   - **Never copy or leak images across reference, calibration, and test splits.** Photos must be independent captures, ideally collected across different capture sessions, days, and smartphone devices.
2. **Diversity**: Reference, calibration, and test photos should be captured from multiple angles, varying distances, different mobile operating systems/sensors (iOS, Android), diverse lighting conditions (sunny, overcast, twilight), and realistic campus activity (pedestrians in background, backpacks).
3. **`unknown/` Distractors**: Contains photos of non-hunt areas (dining halls, hallways, random trees, generic doors). These are essential to calibrate the unknown-image rejection rate and prevent false acceptances of off-target photos.
4. **No Fabricated Data**: This repository provides directory scaffolding and documentation only. Embeddings, calibrated thresholds, and performance metrics must be generated from real campus photos.

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10+
- `pip` or [`uv`](https://github.com/astral-sh/uv)

### Setup Virtual Environment

```bash
# Clone and enter directory
cd campus-hunt-cv

# Create virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

---

## 4. Running on CPU (Resource & Latency Guide)

The component is configured to run on **CPU by default**, eliminating the need for GPU infrastructure:

- **Model**: `ViT-B-32` pretrained on `laion2b_s34b_b79k` (~340MB weights, ~600MB peak RAM).
- **Inference Mode**: Inference runs under `torch.inference_mode()` with `model.eval()`.
- **CPU Multi-threading**: PyTorch CPU threads can be configured via `num_threads` in `ModelConfig` or via `torch.set_num_threads(N)`. On a modern multi-core CPU (e.g. 4–8 cores), inference latency is estimated at ~30–70ms per image.
- **In-Memory Dot Product**: For 20–50 campus locations with 3–5 reference photos each (~100–250 total 512D vectors), matrix dot product in NumPy takes **less than 0.2 milliseconds**. No vector database (Milvus, Pinecone, Qdrant) is required.
- **Hardware Acceleration**: Set `device="mps"` for Apple Silicon or `device="cuda"` for NVIDIA GPUs. Use `device="auto"` for automatic detection with CPU fallback.

---

## 5. Reference Embeddings Storage Contract

Embeddings are extracted once and saved into a safe, versioned pair:
1. **Compressed NumPy Archive (`.npz`)**:
   Contains stacked unit L2-normalized float32 embeddings `(N, D)` and associated string labels `(N,)`. Loaded strictly with `allow_pickle=False` to eliminate arbitrary code execution vulnerabilities.
2. **JSON Metadata Companion (`.meta.json`)**:
   Contains schema version, model architecture name (`ViT-B-32`), weights tag (`laion2b_s34b_b79k`), feature dimension (512), timestamp, and per-location image counts and file names.
3. **Model Version Compatibility Enforcement**:
   When loading reference embeddings at runtime or calibration, the stored model architecture and weights tag are verified against the active embedder. If a mismatch is detected, the pipeline fails safely with `ERROR_MODEL_VERSION_MISMATCH` and refuses to silently compare incompatible feature spaces.

### Building Reference Embeddings

```bash
# Scan data/reference/ and generate embeddings
campus-hunt-build-embeddings \
  --reference-dir data/reference \
  --output-npz data/reference_embeddings.npz \
  --output-meta data/reference_embeddings.meta.json \
  --model-name ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --device cpu
```

---

## 6. Threshold Calibration & Hold-Out Evaluation

### Why Thresholds Cannot Be Hardcoded
Cosine similarity distributions depend on:
- Model checkpoint geometry.
- Visual distinctiveness of campus landmarks (e.g. unique statues vs. similar brick building facades).
- Environmental lighting and weather conditions.

Hardcoding a generic threshold (e.g. `0.70`) is arbitrary and risks high false acceptance rates for distractors or false rejections of legitimate players.

### Calibration & Test Workflow

```bash
campus-hunt-calibrate \
  --calibration-dir data/calibration \
  --test-dir data/test \
  --embeddings-npz data/reference_embeddings.npz \
  --metadata-json data/reference_embeddings.meta.json \
  --output-config calibrated_thresholds.json \
  --max-far 0.02 \
  --min-unknown-rejection 0.90 \
  --min-calib-samples 5 \
  --min-calib-unknown 10 \
  --min-test-samples 5 \
  --min-test-unknown 10 \
  --device cpu
```

### Process Details & Safety Guarantees:
1. **Calibration Split (`data/calibration/`)**:
   - Sweeps candidate match thresholds and margins.
   - Uses the **actual reference photo count** for each location from the embeddings store.
   - **Sample-Count Safeguards**: Requires at least **5 independent captures per hunt location** and at least **10 unknown/distractor captures** by default (`--min-calib-samples 5`, `--min-calib-unknown 10`). A single photo per category cannot qualify as an adequate split.
   - **Data Safeguard Note**: Minimum capture thresholds are practical data-quality safeguards to prevent degenerate or unrepresentative calibration—they do **not** constitute statistical proof of model reliability on campus. The team must still collect varied captures (across different phones, camera angles, lighting conditions, and times of day) and review confusion matrices.
   - **Provisional Status on Lowered Minimums**: If a user manually overrides minimums below project standards (< 5 per location or < 10 unknown), the CLI emits a prominent warning and marks calibration status as `"provisional"` (`is_provisional=True`) in the metadata rather than implying verified evidence.
2. **Fail-Closed Calibration (No Satisfying Threshold or Inadequate Data)**:
   - If calibration data is inadequate (missing locations or counts below safeguard), calibration halts immediately with an error and **does not** create or overwrite any configuration.
   - If no candidate threshold pair meets all configured safety constraints (e.g. `FAR <= max_far` and `unknown_rejection >= min_unknown_rejection`), calibration **fails closed**:
     - It **does not** write or overwrite an active calibrated configuration (`scoring.calibrated=True`).
     - If an existing configuration file is present at `--output-config`, it remains untouched.
     - Files are written atomically on success.
     - The CLI prints a diagnostic summary and exits with a non-zero exit code (`1`).
     - An optional diagnostic report can be saved via `--diagnostic-report`, marked as `deployable: false, calibrated: false`.
3. **Hold-Out Test Split Outcomes (`data/test/`)**:
   - Requires at least **5 independent captures per hunt location** and at least **10 unknown/distractor captures** (`--min-test-samples 5`, `--min-test-unknown 10`) for a completed evaluation.
   - `unavailable`: No usable test captures in `data/test/`. Metrics are `null`, and missing categories are recorded.
   - `incomplete`: Test captures exist, but some reference locations or `unknown/` are missing, or counts fall below minimums. Partial metrics are **not** presented as complete independent test performance (metrics are `null`), and missing categories are recorded with a clear message.
   - `completed`: All required categories and sample counts meet or exceed the standard minimums (>= 5 per location, >= 10 unknown). Selected thresholds are evaluated once to measure unbiased generalization performance.
   - `provisional`: If test minimums were manually lowered below project standards (< 5 or < 10) and met, test evaluation runs but is recorded as `"provisional"` with `is_provisional=True`, warning that the sample size does not guarantee generalization.
4. **No Unmeasured Mismatch Threshold**:
   - Calibration tunes `match_threshold` and `margin_threshold`. It does **not** insert an arbitrary, unvalidated `mismatch_threshold: 0.30`. `mismatch_threshold` defaults to `None`. Teams advance strictly on the conservative `strong_match` criteria.

---

## 7. Calling `verify_location` (Backend Integration)

### Security Contract: Server-Authoritative Target Location
> [!IMPORTANT]
> The caller **MUST** supply the team's current expected location from the authoritative game state on the server. **NEVER** accept or trust a client-supplied location ID submitted from a player's device.

### Verification Function

```python
from campus_hunt_cv import verify_location, LocationVerifier, CampusHuntConfig

# In a FastAPI / Flask lifespan, initialize the verifier ONCE:
config = CampusHuntConfig.from_json_file("calibrated_thresholds.json")
verifier = LocationVerifier(config=config)

# In the request handler:
result = verify_location(
    image_bytes=uploaded_file_bytes,
    target_location_id="clock_tower",  # Server-authoritative riddle location
    verifier=verifier,
)
```

### Return Schema

```json
{
  "result": "strong_match",
  "target_score": 0.824512,
  "best_matching_location_id": "clock_tower",
  "best_matching_score": 0.824512,
  "best_competing_location_id": "library_atrium",
  "best_competing_score": 0.612045,
  "margin": 0.212467,
  "reason_code": "STRONG_MATCH",
  "model_version": "ViT-B-32/laion2b_s34b_b79k"
}
```

### Response Fields

| Field | Type | Description |
|---|---|---|
| `result` | `"strong_match"` \| `"ambiguous"` \| `"mismatch"` | High-level verdict for game engine review. |
| `target_score` | `float` \| `null` | Cosine similarity score for expected target location. |
| `best_matching_location_id` | `str` \| `null` | Highest scoring location ID across all configured locations. |
| `best_matching_score` | `float` \| `null` | Similarity score of the top-1 location. |
| `best_competing_location_id` | `str` \| `null` | Strongest runner-up competitor (different from target). |
| `best_competing_score` | `float` \| `null` | Similarity score of the strongest competitor. |
| `margin` | `float` \| `null` | Difference: `target_score - best_competing_score`. |
| `reason_code` | `str` | Actionable machine-readable code explaining the verdict. |
| `model_version` | `str` | Model identifier (e.g. `ViT-B-32/laion2b_s34b_b79k`). |

---

## 8. Calibrated Configuration JSON Format

The configuration file written by `campus-hunt-calibrate` uses a nested structure that strictly matches `CampusHuntConfig` with `extra="forbid"`:

```json
{
  "environment": "production",
  "model": {
    "model_name": "ViT-B-32",
    "pretrained": "laion2b_s34b_b79k",
    "device": "cpu",
    "num_threads": null
  },
  "image_limits": {
    "max_image_bytes": 15728640,
    "max_image_pixels": 50000000,
    "min_image_dimension": 32,
    "allowed_formats": ["JPEG", "JPG", "PNG", "WEBP"]
  },
  "scoring": {
    "aggregation_method": "top_k_mean",
    "top_k": 3,
    "calibrated": true,
    "match_threshold": 0.65,
    "margin_threshold": 0.08,
    "mismatch_threshold": null,
    "min_reference_images_per_location": 1,
    "allow_unvalidated_auto_accept": false
  },
  "storage": {
    "reference_data_dir": "data/reference",
    "calibration_data_dir": "data/calibration",
    "test_data_dir": "data/test",
    "embeddings_path": "data/reference_embeddings.npz",
    "metadata_path": "data/reference_embeddings.meta.json"
  },
  "calibration_metadata": {
    "model_version": "ViT-B-32/laion2b_s34b_b79k",
    "status": "calibrated",
    "is_provisional": false,
    "warning": null,
    "calibrated_at": "2026-10-30T19:00:00Z",
    "minimums_used": {
      "min_calibration_samples_per_location": 5,
      "min_calibration_unknown_samples": 10,
      "min_test_samples_per_location": 5,
      "min_test_unknown_samples": 10
    },
    "calibration_sample_counts": {
      "clock_tower": 6,
      "library_atrium": 5,
      "unknown": 12
    },
    "calibration_metrics": {
      "correct_accept_rate": 0.91,
      "wrong_false_accept_rate": 0.01,
      "unknown_rejection_rate": 0.95
    },
    "test_evaluation": {
      "status": "unavailable",
      "reason": "No independent test captures found in data/test/",
      "sample_counts": {
        "clock_tower": 0,
        "library_atrium": 0,
        "unknown": 0
      },
      "missing_categories": [
        "clock_tower (count=0 < 5)",
        "library_atrium (count=0 < 5)",
        "unknown (count=0 < 10)"
      ],
      "minimums_used": {
        "min_test_samples_per_location": 5,
        "min_test_unknown_samples": 10
      },
      "is_provisional": false,
      "metrics": null
    }
  }
}
```

---

## 9. Decision Logic & Conservative Policy

### Decision Rules:
1. **Uncalibrated Safety Gate**: If `calibrated` is `False` or thresholds are `None`, the verifier **always** returns `"ambiguous"` with `reason_code="UNCALIBRATED_THRESHOLDS"`. It will never guess or use arbitrary thresholds.
2. **`strong_match`**: Requires `target_location_id == best_matching_location_id` **and** `target_score >= match_threshold` **and** `margin >= margin_threshold`.
3. **`mismatch`**: Requires a competing location to decisively win (`best_matching_score >= match_threshold` and `best_matching_score - target_score >= margin_threshold`).
4. **`ambiguous`**: All borderline scores, narrow margins, ties, insufficient reference photos, uncalibrated states, or unvalidated configurations.
5. **Error Routing**: If image bytes are corrupted, model weights mismatch, or inference fails, the system returns `result="ambiguous"` with an `ERROR_...` reason code (e.g. `ERROR_IMAGE_TOO_LARGE_BYTES`, `ERROR_MODEL_VERSION_MISMATCH`). The backend can route this to manual staff review. **An AI error is never turned into an unfair player failure.**

### Runtime Authorization Gate: Automatic Acceptance vs. Manual Review

To guarantee competition integrity, matching thresholds alone is **not** sufficient to automatically unlock team progression. The runtime verifier checks the empirical validation provenance of the active configuration:

| Calibration Status | Hold-Out Test Status | Matching Submission Outcome | Resulting `reason_code` | Action |
|:---|:---|:---|:---|:---|
| `calibrated` (standard) | `completed` | `strong_match` | `STRONG_MATCH` | Automatic progression |
| `provisional` (lowered count) | Any (`completed`, `provisional`, etc.) | `ambiguous` | `PROVISIONAL_CALIBRATION_REQUIRES_REVIEW` | Queued for manual staff review |
| Any | `unavailable` | `ambiguous` | `HOLDOUT_TEST_NOT_COMPLETED` | Queued for manual staff review |
| Any | `incomplete` | `ambiguous` | `HOLDOUT_TEST_NOT_COMPLETED` | Queued for manual staff review |
| Any | `provisional` | `ambiguous` | `HOLDOUT_TEST_NOT_COMPLETED` | Queued for manual staff review |
| Missing / legacy metadata | Missing | `ambiguous` | `HOLDOUT_TEST_NOT_COMPLETED` | Queued for manual staff review |

> [!NOTE]
> When a submission is routed to manual review due to provisional or uncompleted hold-out status, **all computed score fields are fully preserved** in the response dictionary (`target_score`, `margin`, `best_matching_location_id`, `best_competing_score`). This gives human adjudicators complete quantitative context to evaluate the photo immediately.

#### Runtime Environment & Explicit Opt-In Override (`allow_unvalidated_auto_accept`)

The verification engine features an explicit runtime environment setting (`CAMPUS_HUNT_ENVIRONMENT`) with supported values:
- `production` (default)
- `development` (or `dev`)
- `test` (or `testing`)

If `CAMPUS_HUNT_ENVIRONMENT` is unset, empty, or set to an unrecognized string, the system **strictly defaults to `production`**.

##### Production Protection (Fail-Closed)
In `production`, the `allow_unvalidated_auto_accept` override is **strictly blocked**. Even if enabled in a JSON config file (`"allow_unvalidated_auto_accept": true`) or via environment variable (`CAMPUS_HUNT_ALLOW_UNVALIDATED_AUTO_ACCEPT=1`), the runtime safety gate will refuse to bypass validation. Any provisional or uncompleted hold-out configuration will safely fail closed to `ambiguous` with `PROVISIONAL_CALIBRATION_REQUIRES_REVIEW` or `HOLDOUT_TEST_NOT_COMPLETED`.

Calling `config.validate_for_production()` at service startup will raise a `ConfigurationError` if unsafe overrides or incomplete calibration/hold-out splits are present.

##### Local Development / Testing Use
If automatic acceptance is explicitly needed during local development, integration testing, or dry-runs before independent hold-out captures are gathered:
1. Set the runtime environment to `development` or `test`:
   ```bash
   export CAMPUS_HUNT_ENVIRONMENT=development
   ```
2. Enable the override via environment variable, config JSON, or calibration CLI:
   ```bash
   # Option A: Environment variable
   export CAMPUS_HUNT_ALLOW_UNVALIDATED_AUTO_ACCEPT=1

   # Option B: CLI flag during calibration
   campus-hunt-calibrate --allow-unvalidated-auto-accept

   # Option C: JSON config
   # "scoring": { "allow_unvalidated_auto_accept": true }
   ```

> [!CAUTION]
> `allow_unvalidated_auto_accept` works **only** when `CAMPUS_HUNT_ENVIRONMENT` is explicitly set to `development` or `test`. In production (the default), this override is completely ignored and blocked to protect competition integrity.

---

## 10. Score Aggregation (Top-$k$ Mean)

For each location $L$ with $N_L$ reference photos:
1. Compute cosine similarities between the submitted photo and each reference embedding:
   $$s_i = \mathbf{q} \cdot \mathbf{r}_i \quad (i = 1, \dots, N_L)$$
2. **Standard Case ($N_L \ge k$, default $k=3$)**:
   Take the mean of the top-$k$ highest similarities:
   $$\text{score}(L) = \frac{1}{k} \sum_{j=1}^k s_{(j)}$$
3. **Safe Fallback ($0 < N_L < k$)**:
   If fewer than $k$ reference photos exist, compute the mean over all available $N_L$ photos:
   $$\text{score}(L) = \frac{1}{N_L} \sum_{j=1}^{N_L} s_j$$
4. **Insufficient Case ($N_L < \text{min\_reference\_images\_per\_location}$)**:
   Score is returned as `ambiguous` with `reason_code="INSUFFICIENT_REFERENCE_DATA"`.

---

## 11. When is Fine-Tuning Justified?

Zero-shot OpenCLIP (`ViT-B-32`) already provides rich representations of architectural landmarks, sculptures, and campus facades. Fine-tuning on small datasets is strongly discouraged due to the high risk of catastrophic forgetting and overfitting.

Fine-tuning is **only worth considering** if all of the following conditions are met:
1. **Data Volume**: At least **2,000 to 5,000+ labeled photos** across 30+ campus locations.
2. **Temporal & Seasonal Diversity**: Imagery captured across all four seasons, varied weather (snow, rain, overcast, intense sun), daytime, dusk, and nighttime.
3. **Device Heterogeneity**: Submissions from 50+ distinct mobile phone camera models.
4. **Empirical Need**: Pretrained ViT-B/32 achieves less than 85% correct acceptance rate at FAR $\le 2\%$, specifically due to fine-grained visual confusion between campus buildings with identical brickwork or identical doors.
5. **Methodology**: Parameter-Efficient Fine-Tuning (PEFT / LoRA on image encoder projection layers) using a supervised contrastive loss (InfoNCE) with hard negative pairs, leaving the base vision backbone frozen.

---

## 12. Testing

The test suite includes 57 fast, isolated offline unit and regression tests:

```bash
# Run unit and regression tests
pytest

# Run with verbose output
pytest -v

# Run optional integration smoke tests (downloads OpenCLIP weights)
pytest -m integration
```
