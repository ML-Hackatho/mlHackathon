# Person 1 handoff

## Data contract for Shivanand and Shreya

```python
from pathsnap_person1.io import discover_trip_dirs, load_trip
from pathsnap_person1.features import FEATURE_COLUMNS, candidate_feature_table

trip = load_trip(trip_dir, partition="train", include_route=True)
training_candidates = candidate_feature_table(trip, align_route=True)

inference_trip = load_trip(trip_dir, partition="evaluator_test")  # labels excluded by default
inference_candidates = candidate_feature_table(inference_trip)  # align_route=False by default
```

Each candidate table has one row for each GPS-observation/road-arc pair. Use `FEATURE_COLUMNS` as the model input. The trip loader excludes `.route` labels unless `include_route=True`; keep that default for evaluator inference and do not pass ground-truth fields to the UI. Use `include_route=True` plus `align_route=True` only for permitted train/student-test development labels. Keep the other fields as IDs and evaluation metadata:

- `trip_id`, `partition`, `obs_index`, `source_track_row`;
- `arc_id` for the candidate;
- `target_arc_id`, `label_available`, `is_target`, `candidate_miss` for supervised development/evaluation;
- `nearest_fallback` to identify observations with no road arc inside the configured radius.

Model scores should be returned in the same row order as the input candidate table so the decoder can group by `(trip_id, obs_index)` and retain the `arc_id` mapping. The UI/API prediction path should not expose ground-truth fields.

Before tuning or drawing conclusions from grouped CV, review the candidate-coverage outliers in trips `00000054` and `00000033`: their aligned target arcs are far from GPS observations for many points. This may reflect reference-route limitations or route/trajectory mismatch; it is not automatically fixed by widening the candidate radius.

## Person 1 baseline experiments

`pathsnap-train-baselines` trained Logistic Regression and Random Forest candidate rankers. It used five-fold group-aware CV by trip on `train/`, excluded candidate-miss observations from fitting, reported candidate misses during evaluation, and used `student_test/` only for development comparison. Fold metrics, student-test metrics, a chart, model artifacts, and parameter metadata are in `outputs/baselines/`.

The initial student-test results were:

| Model | 5-fold CV candidate Macro F1 | Student-test Top-1 accuracy | Student-test Macro F1 (mean per trip) | Candidate Log Loss | Candidate miss rate |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.741 ± 0.020 | 0.789 | 0.630 | 0.335 | 0.245% |
| Random Forest | 0.875 ± 0.034 | 0.834 | 0.735 | 0.144 | 0.245% |

The detailed EDA, fold metrics, timing, and route-label alignment caveats are in [PERSON1_RESULTS.md](PERSON1_RESULTS.md). These are two of the team's five required model experiments, not a final model selection.
