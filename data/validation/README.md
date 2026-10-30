# Validation Split (Legacy / Migration Notice)

For rigorous machine learning evaluation, validation is now separated into two independent splits:
1. `data/calibration/`: Used exclusively to sweep and select similarity and margin thresholds (requires >= 5 captures per hunt location and >= 10 unknown captures).
2. `data/test/`: Used as a strict hold-out split to report unbiased test metrics (requires >= 5 captures per hunt location and >= 10 unknown captures).

If files are present here, `campus-hunt-calibrate` will fall back to reading `data/validation/` if `data/calibration/` has not yet been populated. However, we strongly recommend placing calibration photos in `data/calibration/` and holdout evaluation captures in `data/test/`.
