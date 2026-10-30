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
2. **Representative Captures**: Include photos from different phones, angles, distances, times of day, and weather conditions.
3. **`unknown/` Distractors**: Essential for evaluating and controlling the False Acceptance Rate (FAR). Include non-hunt campus areas, hallways, generic objects, and sidewalks.
4. **Purpose**: Data placed here is used solely by `campus-hunt-calibrate` to sweep candidate thresholds and margins. It must not be presented as final independent test accuracy.

