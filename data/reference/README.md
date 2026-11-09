# Reference Images Directory

Place reference images for each known campus hunt location in subdirectories named after their stable `location_id`:

```
data/reference/
  ├── 2nd_floor_clock/
  │   ├── photo1.jpg
  │   └── photo2.jpg
  ├── clock_tower/
  │   ├── photo1.jpg
  │   ├── photo2.jpg
  │   └── photo3.jpg
  └── library_atrium/
      ├── atrium1.jpg
      └── atrium2.jpg
```

## Location ID Stabilization & Mapping
Location subdirectories use stable, lowercase snake_case identifiers (e.g., `2nd_floor_clock`, `a_block_entry_gate`, `ashok_goel_library`).
The authoritative mapping from stabilized `location_id` to human-readable `display_name` is recorded in `data/location_id_map.csv`.

> [!IMPORTANT]
> **Split Consistency Requirement**: Future calibration (`data/calibration/`) and hold-out test (`data/test/`) splits **must use these exact stable IDs** as directory names so that categories align across the entire verification pipeline.

## Guidelines:
1. **Diversity**: Capture reference photos across different angles, distances, times of day, lighting conditions (sunny, overcast, twilight), and different smartphone cameras.
2. **Distinctness**: Ensure photos focus on the distinct architectural, sculpture, or landmark features of the location.
3. **No Cross-Contamination**: NEVER copy or share images between `data/reference/`, `data/calibration/`, and `data/test/`. They must represent completely separate captures.

