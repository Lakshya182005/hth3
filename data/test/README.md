# Independent Test Images Directory

Place hold-out test captures here to evaluate final generalization performance on the calibrated thresholds.

```
data/test/
  ├── clock_tower/
  │   ├── test_photo1.jpg
  │   └── test_photo2.jpg
  ├── library_atrium/
  │   └── test_photo1.jpg
  └── unknown/
      ├── off_campus.jpg
      └── parking_lot.jpg
```

## Guidelines:
1. **Strict Hold-Out**: Photos here are evaluated **only once** on the selected thresholds chosen from calibration data. They must never be used to choose or adjust thresholds.
2. **True Generalization**: Collected from different capture sessions, different times of day, and additional participant devices to measure real-world performance.
3. **`unknown/`**: Holds non-hunt distractor images used to verify generalization of the rejection rate.

