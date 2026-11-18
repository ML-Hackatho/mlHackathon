# PathSnap – interface contract (Person 1 ↔ Person 2 ↔ Person 3)

Status: implemented. Person 1's `pathsnap_person1` provides `point`, `arc`, `t_proj` and `FEATURES` directly
(see the team-level `../INTERFACE.md` and `../HANDOFF.md`); this folder calls it through `src/pathsnap/team.py`.

## 1. What the decoder needs from the feature pipeline (Person 1 → Person 3)

A candidate table (pandas DataFrame), **one row per (GPS point, candidate arc)**, sorted by `point`:

| column   | type  | meaning                                                                      |
|----------|-------|------------------------------------------------------------------------------|
| `point`  | int   | 0-based index of the GPS observation in Person 1's retained `.track` order     |
| `arc`    | int   | candidate arc id (0-based row index in `.arcs`)                             |
| `t_proj` | float | projection of the GPS point onto the arc, clamped to [0, 1] (0 = from-node) |
| feature columns | float | Person 1's 15 `FEATURES`, in that exact order                       |

Every GPS point must have at least one candidate row. Labels (`y`) are only added for training and evaluation; they are never needed at prediction time.

## 2. What the decoder needs from the model (Person 2 → Person 3)

A fitted model with `predict_proba(X)[:, 1]` = P(candidate arc is the reference arc). This is the HMM emission.
The decoder settings are stored in `models/decoder_config.json`:

```json
{"model_name": "mlp", "hmm": {"beta": 1.83, "w_emit": 1.0, "back_tol": 3.18, "sp_limit": 3000.0, "min_prob": 0.0001}}
```

If the team's final model is not the MLP, save its out-of-fold probabilities as `results/oof_<name>.npy` (row order of Person 1's `train_candidates.csv.gz`) and run `PATHSNAP_MODEL=<name> python scripts/03_tune_decoder.py` (Bayesian optimisation, ~10 min), so the decoder is tuned to the final model's probability scale.

## 3. Python call the API makes (Person 3 → Person 2)

```python
from pathsnap.inference import RoutePredictor, score_against_reference
predictor = RoutePredictor("models/<final_model>.joblib", "models/decoder_config.json")   # once, at start-up

out, trip = predictor.predict(trip_dir)                        # never opens the .route file
out["evaluation"], _ = score_against_reference(trip_dir, out)    # server side only, returns numbers only
return out
```

## 4. HTTP API used by the UI (Person 2 implements, Person 3's UI consumes)

CORS must allow the UI's origin (the UI runs on Laptop 2).

### `GET /health`
```json
{"status": "ok", "model": "mlp"}
```

### `GET /trips`
```json
{"trips": ["00000000", "00000007", "..."]}
```
Lists the trip folders in the evaluation directory on Laptop 1.

### `POST /predict`   body: `{"trip_id": "00000040"}`
```json
{
  "trip_id": "00000040",
  "model": "mlp",
  "n_points": 2995,
  "gps": [[lat, lon], ...],
  "predicted_route_coords": [[lat, lon], ...],
  "predicted_route_arcs": [arc_id, ...],
  "predicted_arcs_per_point": [arc_id, ...],
  "classifier_only_arcs_per_point": [arc_id, ...],
  "network_segments": [[[lat, lon], [lat, lon]], ...],
  "decoder": {"chain_breaks": 0, "route_continuity_hmm": 0.959,
              "route_continuity_classifier_only": 0.812, "gps_gaps_over_5s": 3},
  "timing_ms": {"features": 120, "classifier": 15, "viterbi": 480, "total": 640},
  "evaluation": {"points_scored": 2995, "accuracy_hmm": 0.972, "accuracy_classifier_only": 0.947,
                 "route_precision": 0.99, "route_recall": 1.0, "route_f1": 0.995, "route_length_f1": 0.998}
}
```
Unknown trip → HTTP 404.

**Leakage rule:** the response must never include the reference route, its geometry or per-point labels. `evaluation` contains summary numbers only. Coordinates are `[lat, lon]`, which is the order Leaflet expects.

`dev_api/test_api.py` is a working implementation of this contract, for testing the UI. Person 2 can reuse it or reimplement it in their own API.
