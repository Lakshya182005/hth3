# Validation Images Directory

Place evaluation captures here to calibrate similarity thresholds and evaluate model discrimination accuracy.

```
data/validation/
  ├── clock_tower/
  │   ├── test_photo1.jpg
  │   └── test_photo2.jpg
  ├── library_atrium/
  │   └── test_photo1.jpg
  └── unknown/
      ├── cafeteria_tray.jpg
      ├── dorm_hallway.jpg
      └── random_tree.jpg
```

## Structure:
- `<actual_location_id>/`: Contains test captures taken at that actual campus location. Used to evaluate true positive acceptance rates and cross-location false acceptance rates.
- `unknown/`: Contains photos taken at non-hunt campus locations, interiors, off-campus scenes, or generic objects that should NOT be accepted as any hunt location. Used to evaluate rejection rate of distractors.

## Guidelines:
1. **Strict Split**: Never reuse reference photos here. All images must be independent captures.
2. **Realistic Variability**: Collect photos from varied phones, varying zoom levels, rainy/cloudy/sunny weather, and with slight occlusions (e.g., student bags, passing pedestrians).

