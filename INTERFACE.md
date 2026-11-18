# PathSnap team interface

This file maps Shreya's agreed interface contract to Person 1's actual preprocessing and feature pipeline. The original file from Shreya on the Desktop is left unchanged.

## 1. Person 1 code and data outputs

- Trip reader: `src/pathsnap_person1/io.py`
- Projection and road index: `src/pathsnap_person1/geometry.py`
- Candidate generation, route alignment, and features: `src/pathsnap_person1/features.py`
- Feature order: `pathsnap_person1.features.FEATURES` (alias of `FEATURE_COLUMNS`)
- Actual labeled development exports:
  - `outputs/shivanand_handoff/train_candidates.csv.gz`
  - `outputs/shivanand_handoff/student_test_candidates.csv.gz`
- Machine-readable schema and export counts: `outputs/shivanand_handoff/candidate_schema.json`
- Detailed EDA, model results, and caveats: `PERSON1_RESULTS.md`

The exports contain actual data-derived candidate rows and can replace the empty placeholder feature CSVs. Read them with `pandas.read_csv(path, compression="gzip")`.

## 2. Candidate table contract

Each row represents one `(trip_id, point, arc)` candidate. Rows are ordered by trip, then GPS observation, then candidate distance from nearest to farthest. The agreed interface columns are present directly:

| Interface field | Person 1 column | Meaning |
|---|---|---|
| `point` | `point` | Zero-based observation index in retained `.track` order. |
| `arc` | `arc` | Zero-based arc row index from `.arcs`. |
| `t_proj` | `t_proj` | Projection fraction along the directed arc, clamped to [0, 1]. |

For internal grouping and evaluation, the same identifiers are also present as `trip_id`, `obs_index`, and `arc_id`. `source_track_row` preserves the original nonblank input record index if rows are later filtered. In the inspected partitions, no GPS rows were invalid and no timestamp steps were non-increasing, so `point` follows the supplied track order.

Every usable GPS observation gets at least one candidate. Search uses a 60 m radius and retains at most 12 arcs; if no arc is within the radius, the nearest arc is added and `nearest_fallback` is true. Candidates are directed arcs built from each `.arcs` row's from/to nodes.

## 3. Final model feature columns and definitions

Use this exact order via `FEATURES`:

| Column | Definition |
|---|---|
| `cross_track_m` | Metric distance from GPS point to candidate arc. |
| `gps_step_m` | Straight-line metric distance from the previous point to this point; zero at the start. |
| `next_gps_step_m` | Straight-line metric distance from this point to the next point; zero at the end. |
| `time_gap_s` | Positive timestamp difference from the previous point; zero when unavailable or non-positive. |
| `gps_speed_mps` | `gps_step_m / time_gap_s`; zero when no positive time gap is available. |
| `arc_length_m` | Metric length between the arc's endpoint nodes. |
| `projection_fraction` | Closest point's fraction along the directed arc, from 0 at from-node to 1 at to-node. This also supplies `t_proj`. |
| `heading_cos` | Cosine of the wrapped angle from candidate bearing to incoming GPS heading. |
| `heading_sin` | Sine of that same incoming heading difference. |
| `prev_heading_cos` | Duplicate of `heading_cos` in this version; retained for compatibility with Person 1's trained baselines. |
| `next_heading_cos` | Cosine of the wrapped angle from candidate bearing to outgoing GPS heading. |
| `prev_heading_available` | 1 if a previous point and positive time gap give an incoming heading, otherwise 0. |
| `next_heading_available` | 1 if a next point and positive time gap give an outgoing heading, otherwise 0. |
| `from_degree` | Directed out-degree of the candidate arc's from-node. |
| `to_degree` | Directed in-degree of the candidate arc's to-node. |

`point`, `arc`, `t_proj`, identifiers, and label fields are metadata; do **not** include them in `X`. The model input is `candidate_table[FEATURES]`.

## 4. Target label format

In train and student-test exports, `is_target` is a boolean binary target: `True` when the candidate arc equals the observation's aligned reference arc, otherwise `False`. `target_arc_id` stores that aligned arc. `label_available` indicates that a route label was available. `candidate_miss` is true for every candidate row belonging to an observation whose aligned target arc is absent from the generated candidate set.

The `.route` file is an ordered arc sequence, not a row-for-row GPS label. Person 1 aligns it to GPS observations with spatial proximity and a soft route-progression penalty; this is a heuristic, not verified physical ground truth. Exclude candidate-miss observation groups from fitting because they contain no positive candidate, but retain them in validation to report candidate miss rate and count their Top-1 prediction as incorrect.

The train export has **1,598,287 candidate rows for 208,417 observations**; 22,860 observations (10.968%) miss their aligned target candidate. The student-test export has **169,278 candidate rows for 19,160 observations**; 47 observations (0.245%) miss. Most train misses are in trips `00000054` and `00000033`; see `outputs/eda/candidate_coverage_by_trip.csv` and `PERSON1_RESULTS.md` before interpreting grouped-CV results.

## 5. Model prediction input and decoder mapping

Inference does not load `.route` and does not create target columns. `load_trip` excludes route labels by default, and `align_route=False` is the default for candidate generation.

```python
from joblib import load
from pathsnap_person1.features import FEATURES, candidate_feature_table
from pathsnap_person1.io import load_trip

trip = load_trip(trip_dir, partition="evaluator_test")  # include_route=False by default
candidates = candidate_feature_table(trip, align_route=False)
X = candidates[FEATURES]
probability = load("models/final_model.joblib").predict_proba(X)[:, 1]

scored = candidates[["trip_id", "point", "arc", "t_proj"]].copy()
scored["probability"] = probability
```

Preserve row order. Group scores by `(trip_id, point)` for the decoder and retain `arc` as the candidate-to-hidden-state mapping. These probabilities are HMM emissions. Person 2's final XGBoost/LightGBM model must use this feature order and the same positive-class meaning; regenerate out-of-fold probabilities from the final model before tuning the decoder.

## 6. Split and leakage rules

- Use `train/` for fitting and GroupKFold; groups are `trip_id`, never individual candidate rows.
- Use `student_test/` only for development comparison; report those results separately.
- Do not use evaluator trips in training, feature selection, tuning, or the UI.
- The evaluator API may score against its server-side reference route after prediction, but the predictor and UI must not receive reference routes, geometry, or per-point labels. Return only aggregate evaluation numbers.

## 7. Person 2 API and Person 3 UI contract

Laptop 1 serves the API; Laptop 2 hosts the UI. Enable CORS for Laptop 2's UI origin.

- `GET /health` returns `{ "status": "ok", "model": "<selected model>" }`.
- `GET /trips` returns evaluator trip IDs available on Laptop 1.
- `POST /predict` accepts `{ "trip_id": "<id>" }`; return HTTP 404 for an unknown ID.
- Coordinates in responses are `[lat, lon]` for Leaflet.

`POST /predict` should return these field names:

```json
{
  "trip_id": "<id>",
  "model": "<selected model>",
  "n_points": 0,
  "gps": [[0.0, 0.0]],
  "predicted_route_coords": [[0.0, 0.0]],
  "predicted_route_arcs": [0],
  "predicted_arcs_per_point": [0],
  "classifier_only_arcs_per_point": [0],
  "network_segments": [[[0.0, 0.0], [0.0, 0.0]]],
  "decoder": {
    "chain_breaks": 0,
    "route_continuity_hmm": 0.0,
    "route_continuity_classifier_only": 0.0,
    "gps_gaps_over_5s": 0
  },
  "timing_ms": {"features": 0, "classifier": 0, "viterbi": 0, "total": 0},
  "evaluation": {
    "points_scored": 0,
    "accuracy_hmm": 0.0,
    "accuracy_classifier_only": 0.0,
    "route_precision": 0.0,
    "route_recall": 0.0,
    "route_f1": 0.0,
    "route_length_f1": 0.0
  }
}
```

The values above only describe the response schema; they are not performance results. Evaluation is computed server-side after `RoutePredictor.predict()` has run without opening `.route`. Never return the reference route, its geometry, or per-point labels. The `evaluation` object contains aggregate numbers only.

Person 2's current implementation is in `src/pathsnap_person2/`. It loads the selected tuned model from `outputs/person2/models/final_model.joblib`, builds candidates with this Person 1 pipeline, calls the HMM/Viterbi adapter, and exposes `/health`, `/trips`, and `/predict`. Environment variables and run commands are documented in [PERSON2_INTEGRATION.md](PERSON2_INTEGRATION.md); model results and tuning settings are in [PERSON2_RESULTS.md](PERSON2_RESULTS.md).

## 8. Person 1 baseline reference

Person 1 trained Logistic Regression and Random Forest using this 15-feature candidate schema. On student-test, Top-1 accuracy was 0.789 (Logistic Regression) and 0.834 (Random Forest); per-trip observation Macro F1 was 0.630 and 0.735. These are two of the team's five experiments, not final model selection. Full grouped-CV metrics and configurations are in `outputs/baselines/` and `PERSON1_RESULTS.md`.
