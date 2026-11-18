# PathSnap Person 1: data and feature pipeline

This project contains Person 1's workstream:

- per-trip readers for `.track`, `.nodes`, `.arcs`, and optional `.route` files;
- dataset inventory and EDA outputs, including a Folium map;
- local metric projection and road-candidate generation;
- candidate-level feature preparation and route-to-GPS label alignment;
- Logistic Regression and Random Forest baselines with trip-grouped cross-validation.

The guidance PDFs were reread before implementation. The dataset archive was extracted into the local, gitignored `data/` directory and used for the EDA and baseline runs. The original archive was left untouched. Actual aggregate findings and results are in [PERSON1_RESULTS.md](PERSON1_RESULTS.md).

## Documented input format

Set `--data-root` to an extracted directory containing `train/` and `student_test/`, each with one folder per trip.

- `.track`: longitude, latitude, timestamp in seconds;
- `.nodes`: longitude, latitude; row position is `node_id`;
- `.arcs`: `from_node_id`, `to_node_id`; row position is `arc_id`;
- `.route`: chronological reference `arc_id` sequence, when provided.

Malformed `.nodes` and `.arcs` rows stop loading because silently dropping them would renumber IDs. Invalid GPS rows are counted and skipped. Coordinates are projected per trip to a local azimuthal-equidistant CRS so distances and candidate radii are in metres. The loader excludes `.route` labels by default; label access must be explicitly enabled for permitted train/student-test development.

## Candidate and label assumptions

The initial candidate settings are a 60 m search radius and at most 12 nearby directed arcs per GPS point. If no arc falls inside the radius, the nearest arc is included and marked `nearest_fallback`.

The reference route is an ordered arc sequence, not a label for each GPS row. For train/student-test development labels, the pipeline aligns it to GPS observations by metric proximity with a soft route-progression penalty; it never matches rows by index. This is a working label-alignment heuristic. The candidate coverage report found serious mismatches in a small number of trips, so inspect those trips before interpreting aggregate model scores. Candidate misses are preserved for evaluation and excluded from baseline fitting. Label alignment is opt-in (`align_route=True`); inference defaults to `False`.

`candidate_feature_table(trip)` returns one row per observation/candidate pair. Model input columns are listed in `pathsnap_person1.features.FEATURE_COLUMNS`; `trip_id`, `obs_index`, `arc_id`, `target_arc_id`, and candidate-miss fields are metadata, not model features. The target is binary candidate relevance, allowing a model to rank each observation's local road candidates without treating trip-local `arc_id` values as globally consistent classes.

## Setup and commands (PowerShell)

From this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Run the trip inventory, quality summaries, EDA plot, and a representative interactive map:

```powershell
pathsnap-eda --data-root "C:\path\to\extracted\Student folder" --out outputs\eda
```

Train and compare the two baselines. Five-fold validation keeps trips grouped; student-testing results are development-stage evaluation only:

```powershell
pathsnap-train-baselines --data-root "C:\path\to\extracted\Student folder" --out outputs\baselines
```

Produce per-trip candidate-coverage diagnostics:

```powershell
pathsnap-candidate-coverage --data-root "C:\path\to\extracted\Student folder" --out outputs\eda\candidate_coverage_by_trip.csv
```

The scripts fit only on `train/`, use `student_test/` for development comparison, and do not discover or read an evaluator partition.

## Generated outputs

- `outputs/eda/trip_inventory.csv`, `summary.json`, and `eda_summary.png`;
- `outputs/eda/map_<trip-id>.html` with observed GPS, nearby road candidates, and reference-route layers;
- `outputs/eda/candidate_coverage_by_trip.csv` to identify trips where reference-route arcs are absent from generated candidates;
- `outputs/baselines/baseline_cv_folds.csv`, `baseline_student_test_results.csv`, and `baseline_comparison.png`;
- `outputs/baselines/models/` with model artifacts and hyperparameter metadata.

All reported scores and figures come from actual runs. Evaluator data was not used.
