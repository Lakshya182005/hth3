# Calibration Images Directory

Place labeled campus captures here to calibrate similarity and margin thresholds.

```
data/calibration/
  ├── clock_tower/
  │   ├── photo1.jpg
  │   └── photo2.jpg
  ├── library_atrium/
  │   └── photo1.jpg
  └── unknown/
      ├── cafeteria.jpg
      └── dorm_hallway.jpg
```

## Guidelines:
1. **Independent Split**: Photos must be completely separate captures from `data/reference/` and `data/test/`. **Never copy or leak images across splits.**
2. **Sample Safeguards (Minimum Counts)**:
   - Provide **at least 5 independent captures per hunt location**.
   - Provide **at least 10 unknown/distractor captures** in `unknown/`.
   - A single photo per category fails data adequacy and will refuse to calibrate.
   - *Note on Reliability*: These counts are practical data-quality safeguards to prevent unrepresentative sweeps—they are not statistical guarantees of model accuracy. The team must inspect confusion matrices and collect more captures for complex or visually ambiguous sites.
3. **Representative Variation**: Capture photos across different smartphone models, varying angles, different distances, times of day (morning, midday, dusk), and weather conditions (sunny, overcast).
4. **`unknown/` Distractors**: Essential for evaluating and controlling the False Acceptance Rate (FAR). Include non-hunt campus areas, hallways, generic objects, parking structures, and sidewalks.
5. **Purpose**: Data placed here is used solely by `campus-hunt-calibrate` to sweep candidate thresholds and margins. It must not be presented as final independent test accuracy.
6. **Provisional Warning**: Manually lowering sample minimums via CLI flags triggers a warning and marks output configs as `"provisional"`.

