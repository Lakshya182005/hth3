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
2. **Sample Safeguards (Minimum Counts)**:
   - Provide **at least 5 independent captures per hunt location**.
   - Provide **at least 10 unknown/distractor captures** in `unknown/`.
   - If counts fall below these practical safeguards or any location is missing, the split is marked `"incomplete"` and partial metrics are withheld.
   - *Note on Reliability*: These minimums are practical data-quality safeguards—not statistical proofs of generalizability. The engineering team must review actual counts and results.
3. **True Generalization**: Collected from separate capture sessions, different times of day, lighting, and additional participant devices to measure real-world performance.
4. **`unknown/`**: Holds non-hunt distractor images used to verify generalization of the rejection rate.
5. **Provisional Reporting**: Overriding test minimums to lower values marks test evaluation as `"provisional"` rather than `"completed"`.

