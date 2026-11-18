# PathSnap: GPS Map Matching and Route Reconstruction

PathSnap reconstructs road routes from GPS trajectories by generating road candidates, ranking them with trained classifiers, and applying HMM/Viterbi route decoding. This repository contains the complete team implementation, experiments, API, and evaluator UI. Person 1's data and feature work includes:

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
- `outputs/shivanand_handoff/train_candidates.csv.gz` and `student_test_candidates.csv.gz` with actual labeled candidate rows;
- `outputs/shivanand_handoff/candidate_schema.json` with the interface fields, exact feature order, definitions, target format, and export counts;
- `outputs/baselines/baseline_cv_folds.csv`, `baseline_student_test_results.csv`, and `baseline_comparison.png`;
- `outputs/baselines/models/` with configuration metadata and available model artifacts. The 684 MB Random Forest joblib is excluded from GitHub due to the file-size limit; regenerate it with `pathsnap-train-baselines` when needed. The selected XGBoost and MLP artifacts are included in their respective model folders.

The team-facing contract that maps the agreed `point`/`arc`/`t_proj` schema to this implementation is in [INTERFACE.md](INTERFACE.md). To regenerate the candidate exports, run `pathsnap-export-candidates --data-root "C:\path\to\extracted\Student folder" --out outputs\shivanand_handoff`.

All reported scores and figures come from actual runs. Evaluator data was not used.

## Person 2: LightGBM, XGBoost, tuning, and API

Person 2's model workflow and current results are documented in [PERSON2_RESULTS.md](PERSON2_RESULTS.md). Reproduction steps and two-laptop API instructions are in [PERSON2_INTEGRATION.md](PERSON2_INTEGRATION.md). The implementation is under `src/pathsnap_person2/`; results and model artifacts are under `outputs/person2/`.

Train and compare the boosted models from the shared candidate exports:

```powershell
python -m pathsnap_person2.training --candidate-dir outputs\shivanand_handoff --out outputs\person2 --folds 5 --trials 5
```

After training, export final-model OOF scores and retune/evaluate the decoder:

```powershell
python -m pathsnap_person2.oof --candidate-dir outputs\shivanand_handoff --out outputs\person2 --folds 5
python -m pathsnap_person2.decoder_tuning --candidate-dir outputs\shivanand_handoff --data-root "data\Student folder" --out outputs\person2 --trials 30
```

## Hackathon deliverables

- [Consolidated Technical Report (PDF)](reports/PathSnap_Technical_Report.pdf): dataset and EDA findings, 15-feature definitions, five-model comparison, XGBoost tuning, HMM/Viterbi evaluation, route map, API/UI evidence, and reproducibility notes.
- [Person 1 results](PERSON1_RESULTS.md), [Person 2 results and API instructions](PERSON2_RESULTS.md), and [Person 2 integration guide](PERSON2_INTEGRATION.md).
- `reports/technical_report_assets/` contains the Folium route map and the screenshots used as demo evidence. The report-generation analyses and combined comparison plots are in `reports/`.

The API and UI were demonstrated together on two laptops: the API serves on port 8000 and the evaluator UI on port 5500. Follow [PERSON2_INTEGRATION.md](PERSON2_INTEGRATION.md) for the commands, CORS setting, and network setup.
