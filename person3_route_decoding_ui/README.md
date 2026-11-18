# PathSnap – Person 3: Model 5 (MLP), HMM/Viterbi route decoding & evaluator UI

Shreya's part of the team project:
1. **Model 5**: an MLP candidate classifier, trained on Person 1's shared features and labels.
2. **HMM + Viterbi route reconstruction**, with Bayesian-tuned hyperparameters. It handles GPS gaps and disconnected road segments, and is compared against classifier-only predictions.
3. **Evaluator UI** (Laptop 2): pick a trip, call the API on Laptop 1, and show the predicted route and scores.

It uses **Person 1's pipeline** (`../src/pathsnap_person1`) for loading trips, candidates, the 15 features and the route-alignment labels. This folder adds no feature engineering of its own.

```
mlHackathon/
  src/pathsnap_person1/            Person 1: loader, candidates, FEATURES, label alignment
  outputs/shivanand_handoff/       Person 1: exported candidate tables (train / student_test)
  person3_route_decoding_ui/       <- this folder
    src/pathsnap/
      team.py          adapter: calls Person 1's loader / candidate_feature_table / labels
      hmm.py           ** HMM + Viterbi decoder, network shortest paths, gap filling **
      inference.py     ** RoutePredictor (used by the API) + server-side scoring **
      io.py, geo.py    decoder geometry (metre projection, arc lengths, bearings)
      metrics.py       Top-1, macro F1, log loss, route continuity, route P/R/F1
      config.py        paths (finds ../src and ../outputs automatically)
    scripts/
      01_prepare_team_candidates.py   read Person 1's exported candidate tables
      02_train_mlp.py                 Model 5: MLP, 5-fold GroupKFold by trip, student-test eval
      03_tune_decoder.py              Bayesian optimisation (Optuna TPE) of decoder hyperparameters
      04_hmm_eval.py                  classifier-only vs HMM on the 10 student-test trips
      05_folium_maps.py               Folium maps: GPS, network, reference, classifier-only, HMM route
    ui/index.html      evaluator UI (Leaflet bundled in ui/vendor)
    dev_api/test_api.py  test API following INTERFACE.md (Person 2 owns the real API)
    models/ results/ figures/ maps/   actual outputs
```

## Setup and run

The folder must sit inside the team repo, because it imports `../src` and reads `../outputs/shivanand_handoff`.
Put the raw trips in `person3_route_decoding_ui/data/train/<trip_id>/` and `.../data/student_test/<trip_id>/`. The decoder needs the road network, and scoring needs `.route`.

```bash
pip install -r requirements.txt
python scripts/01_prepare_team_candidates.py   # ~10 s
python scripts/02_train_mlp.py                 # ~2 min (Colab ~5 min)
python scripts/03_tune_decoder.py              # ~10 min (30 Bayesian trials; Colab ~30 min)
python scripts/04_hmm_eval.py                  # ~1 min
python scripts/05_folium_maps.py               # ~30 s
```

## Results (seed 42; logs in `results/log_*.txt`)

### Model 5 – MLP (64, 32), ReLU, Adam, early stopping, on Person 1's 15 features
| metric | value |
|---|---|
| CV macro F1 (5-fold GroupKFold by trip) | 0.874 ± 0.024 |
| CV Top-1 accuracy | 0.736 ± 0.132 |
| CV log loss | 0.128 |
| Train macro F1 / Top-1 (mean over folds) | 0.884 / 0.825 |
| Student-test Top-1 accuracy (all points pooled) | 0.828 |
| Student-test macro P / R / F1 | 0.907 / 0.874 / 0.890 |
| Student-test log loss | 0.112 |
| Classifier-only route continuity | 0.744 |
| Epochs (final fit) | 41 (early stopping) |
| Inference | 2.9 ms per 1000 GPS points |

* Scaling (`StandardScaler`) is part of the saved pipeline in `models/mlp.joblib`.
* Candidate-miss points are excluded from fitting and counted as errors in evaluation (Person 1's rule).
* The large CV Top-1 spread comes from folds 3 and 4 (val Top-1 0.63 and 0.56), which contain trips 00000054 and 00000033. For many points in those trips, the aligned reference arc is not among the candidates (Person 1 flagged this). Train and validation macro F1 are close (0.884 vs 0.874), so there is no strong overfitting.
* See `figures/mlp_training_history.png`, `figures/mlp_cv_folds.png` and `figures/mlp_confusion_student_test.png`.

### Decoder hyperparameters: Bayesian optimisation (training trips only)
* **Search:** Optuna **TPE** sampler, 30 trials, seed 42, with trial 0 the untuned default. No grid or random search is used.
* **Objective:** mean decoded observation-level accuracy on the 80 training trips, using the MLP's **out-of-fold** probabilities. Each trip is decoded by a model that never saw it.
* **Search space:** `beta` log-uniform 0.5–64 m, and `back_tol` uniform 0–40 m. `w_emit` is fixed at 1, because the Viterbi path depends on `w_emit` and `beta` only through their product.
* **Best:** beta = 1.83 m, back_tol = 3.18 m (`models/decoder_config.json`, all trials in `results/decoder_bayesopt_trials.csv`, plot in `figures/decoder_bayesopt_history.png`).

| | train OOF accuracy | train OOF continuity |
|---|---|---|
| classifier only | 0.806 | – |
| HMM untuned (beta 8, back_tol 12) | 0.865 | 0.896 |
| HMM Bayesian-tuned | **0.868** | 0.893 |

| student test (mean of 10 trips) | accuracy | continuity | route F1 |
|---|---|---|---|
| HMM untuned | 0.897 | **0.918** | 0.963 |
| HMM Bayesian-tuned | **0.903** | 0.913 | **0.969** |

Tuning optimises accuracy, which improves slightly, while continuity drops slightly. This is a real trade-off and should be reported as one.

### Classifier-only vs classifier + HMM (10 student-test trips, mean per trip)
| | classifier only | + HMM/Viterbi |
|---|---|---|
| Observation-level accuracy | 0.850 | **0.903** |
| Route continuity | 0.744 | **0.913** |
| Route F1 (arc set) | 0.851 | 0.935 (no gap fill) / **0.969** (with gap fill) |
| Accuracy within 3 points of a >5 s GPS gap | 0.692 | **0.803** |

* HMM decoding improved accuracy on 9 of 10 trips. On trip 00000007 it dropped slightly (0.897 → 0.891), although continuity reached 1.0.
* Trip 00000077 is the hardest (0.644 → 0.753).
* Trip 00000000 had 2 Viterbi chain breaks and still reached 0.960.
* Per-trip numbers are in `results/hmm_vs_classifier_per_trip.csv`; maps are in `maps/decoded_<trip>.html`.

## How the decoder works (for the report and the viva)
* **Hidden states:** the candidate arcs of each GPS point (from Person 1's candidate generator: up to 12 directed arcs within 60 m), together with the point's projection `t_proj` on that arc.
* **Emission:** `w_emit · log p`, where p is the classifier's `predict_proba(X)[:, 1]` for that (point, arc). This works with **any** classifier: MLP, XGBoost, Random Forest, and so on.
* **Transition** (Newson & Krumm style). For candidate a at t−1 and b at t: `d_route` = metres driven along the directed network (rest of a + shortest path between the arcs + start of b), and `d_gc` = straight-line distance between the two GPS points. Then `log P = −|d_route − d_gc| / (beta·√Δt)`. On the same arc, backward jitter up to `back_tol` m is free, and larger backward moves are penalised. Unreachable pairs get −∞.
* **GPS gaps:** beta is scaled by √Δt, so longer gaps tolerate a larger mismatch. Dijkstra is bounded at 3 km.
* **Disconnected segments:** if the GPS jump is beyond the search limit, or no transition is finite, the chain **breaks**. Viterbi restarts from the emissions, and the break is counted (0.2 per test trip on average).
* **Gap filling:** 1 Hz points can skip very short arcs, so `fill_route` joins non-adjacent consecutive arcs with the shortest path. This is why route F1 rises from 0.935 to 0.969.
* **Recurrence:** `δ_t(b) = max_a [δ_{t−1}(a) + log P(b|a)] + emission_t(b)`, followed by backtracking. Network distances are cached per node, and transition matrices are cached when consecutive points have the same candidate sets.

## For Person 2 (API integration)
```python
import sys; sys.path.insert(0, "person3_route_decoding_ui/src")
from pathsnap.inference import RoutePredictor, score_against_reference

predictor = RoutePredictor("models/xgboost.joblib",                               # any model with predict_proba
                           "person3_route_decoding_ui/models/decoder_config.json")
out, trip = predictor.predict(trip_dir)                 # Person 1 loader, route NOT read
out["evaluation"], _ = score_against_reference(trip_dir, out)   # server-side, numbers only
return out                                              # JSON format in INTERFACE.md
```
* The model must take `candidate_table[FEATURES]` (Person 1's 15 features, in order) and return `predict_proba(X)[:, 1]`.
* `dev_api/test_api.py` is a complete working FastAPI app: `/health`, `/trips`, `/predict`, CORS and request logging. Reuse it and change only `PATHSNAP_MODEL_FILE`.
* **Re-tune the decoder for XGBoost:** save XGBoost's 5-fold out-of-fold probabilities, in the row order of `outputs/shivanand_handoff/train_candidates.csv.gz`, as `person3_route_decoding_ui/results/oof_xgboost.npy`. Then run:
  `PATHSNAP_MODEL=xgboost python scripts/03_tune_decoder.py`
  On Windows PowerShell: `$env:PATHSNAP_MODEL="xgboost"; python scripts/03_tune_decoder.py`.
  This writes `models/decoder_config.json` for XGBoost. Until then, the MLP-tuned config works as a reasonable default.

## Two-laptop demo
**Laptop 1 (API):** `PATHSNAP_EVAL_DIR=<folder with evaluator trips> uvicorn dev_api.test_api:app --host 0.0.0.0 --port 8000`. Use Person 2's API once it's ready.
**Laptop 2 (UI):** `cd ui && python -m http.server 5500`, then open http://localhost:5500, enter `http://<laptop1-ip>:8000`, click Connect, pick a trip, and click Run prediction.
Both laptops must be on the same network, with port 8000 allowed through Laptop 1's firewall.

## Leakage safeguards
* Prediction uses Person 1's `load_trip(..., include_route=False)` and `candidate_feature_table(align_route=False)`, so `.route` is never read.
* `score_against_reference` runs only on the server and returns summary numbers. No reference geometry or labels reach the UI.
* Decoder hyperparameters are tuned only on training trips (out-of-fold probabilities). Student-test trips are used only for the final comparison, and evaluator trips are never touched.
