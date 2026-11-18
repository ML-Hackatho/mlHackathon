# Person 1: EDA and baseline results

Generated from the `train/` and `student_test/` partitions only. No evaluator partition was read. The inventory, figures, trip-level candidate coverage, CV results, and fitted model files are under `outputs/`.

## EDA findings

| Measure | Training | Student testing |
|---|---:|---:|
| Trips | 80 | 10 |
| GPS observations | 208,417 | 19,160 |
| Observations per trip, median (min–max) | 1,529 (361–19,313) | 1,420 (262–3,958) |
| Road nodes across trip files | 1,933,485 | 90,689 |
| Directed road arcs across trip files | 3,876,904 | 185,041 |
| Invalid or impossible GPS rows | 0 | 0 |
| Non-increasing timestamp steps | 0 | 0 |
| Consecutive timestamps exactly 1 second apart | 96.504% | 98.621% |
| Median of each trip's largest positive time gap | 11 s | 2 s |
| Largest positive time gap in a trip | 153,064 s | 210 s |
| Isolated road nodes | 0 | 0 |
| Median undirected network components per trip | 20 | 11 |
| Maximum undirected network components in one trip | 421 | 100 |

Node and arc totals are sums across trip files, not counts of unique global roads. The largest training gap is about 42.5 hours and should be treated as an anomaly or a sequence break by the route decoder. The per-trip maximum-gap distribution is visible in `outputs/eda/eda_summary.png`.

## Candidate-coverage review

The initial candidate generator uses a 60 m search radius and retains at most 12 nearby directed arcs per observation. Across all labeled observations, candidate misses were **22,860 / 208,417 (10.968%)** in training and **47 / 19,160 (0.245%)** in student testing.

Most training misses are concentrated in two trips:

| Trip | Missed observations | Miss rate | Median distance from GPS to aligned target arc | Notes |
|---|---:|---:|---:|---|
| `00000054` | 17,099 / 19,313 | 88.5% | 146 m | 95th percentile target-arc distance 211 m; largest time gap 79 s |
| `00000033` | 3,907 / 5,844 | 66.9% | 133 m | 95th percentile target-arc distance 143 m; largest time gap 3 s |

Those two trips account for about 92% of the training candidate misses. This suggests a route/reference mismatch or an issue needing inspection; increasing the candidate radius blindly may add many irrelevant arcs. The other large miss rate is trip `00000002` (11.4%; target-arc distance median 209 m, 95th percentile 519 m, and a 160 s maximum gap). The exact per-trip results are in `outputs/eda/candidate_coverage_by_trip.csv`.

These labels were aligned from an ordered `.route` sequence to GPS observations with a soft monotonic-progression heuristic; the code does not assume row-for-row correspondence. The route alignment for trips `00000054` and `00000033` should be visually reviewed before treating their observation labels as reliable. The dataset guide itself cautions that supplied reference routes may have limitations.

## Baseline experiments

Five-fold validation grouped by trip was run on the 80 training trips. Candidate-miss observations were excluded from fitting but retained in validation so the miss rate remained visible. The 10 student-testing trips were used only for development comparison.

| Model | CV candidate Macro F1 (mean ± SD) | CV observation Top-1 (mean ± SD) | Student-test candidate Macro F1 | Student-test Top-1 accuracy | Student-test observation Macro F1, mean per trip | Candidate Log Loss | Candidate miss rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.741 ± 0.020 | 0.713 ± 0.119 | 0.732 | 0.789 | 0.630 | 0.335 | 0.245% |
| Random Forest | 0.875 ± 0.034 | 0.751 ± 0.120 | 0.880 | 0.834 | 0.735 | 0.144 | 0.245% |

The Logistic Regression pipeline used `StandardScaler` followed by `LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs", random_state=42)`. Random Forest used `n_estimators=250`, `max_depth=18`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`, `n_jobs=-1`, and `random_state=42`.

Random Forest was stronger on these two baselines. The candidate-level Log Loss is for binary “is this candidate the aligned target arc?” probabilities; it is not a multiclass route-level loss. Observation Macro F1 is calculated per trip over the trip-local arc IDs, then averaged across trips.

| Model | Fitting time on all candidate-hit training rows | Student-test model-scoring time |
|---|---:|---:|
| Logistic Regression | 2.52 s | 0.038 s |
| Random Forest | 227.32 s | 0.979 s |

Scoring time is only `predict_proba` over 169,278 student-test candidate rows; it excludes candidate generation and HMM/Viterbi decoding. Random Forest artifact size is about 718 MB, so do not commit it as the final artifact unless the team decides its size is acceptable. The trained models and configs are in `outputs/baselines/models/`.

These results cover only Person 1's two baselines. The team still needs at least three additional model experiments, Bayesian optimisation, and the classifier-versus-HMM comparison before selecting a final model.
