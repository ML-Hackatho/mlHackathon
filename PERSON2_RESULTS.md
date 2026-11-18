# Person 2: boosted-model experiments and inference API

## Data and protocol

The experiments used Person 1's `train_candidates.csv.gz` and `student_test_candidates.csv.gz` exports and the exact 15 columns in `FEATURES`. The positive class is `is_target=True`. Training contains 1,598,287 candidate rows across 80 trips and 208,417 GPS observations. Candidate-miss groups (22,860 observations) were excluded from fitting; they remained in validation and count as Top-1 errors. The final full-training fit used 1,522,804 candidate-hit rows. The student-test partition contains 169,278 candidate rows across 10 trips and 19,160 observations.

Five-fold `GroupKFold` used `trip_id` as the grouping variable. Model selection and Optuna tuning used only the training folds. Student-test metrics are held-out development comparisons; evaluator data was not read. The reference route labels come from Person 1's route-to-GPS alignment heuristic and are not verified per-observation ground truth. Training candidate misses are concentrated in trips `00000054` and `00000033`; interpret fold scores in light of that coverage issue.

## Model comparison

Candidate-level Macro F1 is binary candidate classification at threshold 0.5. Observation-level Top-1 selects the highest-probability candidate per GPS observation. Observation Macro F1 is computed per trip over arc IDs and averaged across trips. Log Loss is candidate-level binary log loss. Times are measured from actual model fits and `predict_proba` calls on the student-test candidate matrix; feature generation and HMM decoding are excluded.

The saved in-fold training scores are calculated on candidate-hit rows only, while each validation fold retains candidate-miss observations. Therefore, train-vs-validation score gaps are descriptive and partly reflect the candidate-miss distribution; they are not a pure overfit estimate.

| Model | CV candidate Macro F1 (mean ± SD) | CV observation Macro F1, mean per trip (mean ± SD) | CV Top-1 (mean ± SD) | Student-test candidate Macro F1 | Student-test observation Macro F1, mean per trip | Student-test Top-1 | Candidate Log Loss | Student-test scoring time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM baseline | 0.8408 ± 0.0371 | 0.7019 ± 0.0486 | 0.7506 ± 0.1189 | 0.8455 | 0.7326 | 0.8347 | 0.1842 | 1.301 s |
| XGBoost baseline | 0.8427 ± 0.0362 | 0.7042 ± 0.0502 | 0.7514 ± 0.1187 | 0.8476 | 0.7321 | 0.8349 | 0.1830 | 0.184 s |
| XGBoost tuned | 0.8428 ± 0.0370 | 0.7069 ± 0.0488 | 0.7528 ± 0.1200 | 0.8488 | 0.7362 | 0.8352 | 0.1839 | 0.519 s |

Tuned XGBoost was selected over the two Person 2 baselines because it had the highest mean per-trip observation Macro F1 in grouped CV. Its advantage was small; these folds do not establish statistically significant superiority. Student-test Top-1 improved only slightly over untuned XGBoost (0.8352 vs 0.8349), while candidate Log Loss was slightly worse (0.1839 vs 0.1830). The saved comparison plot and fold table are generated from the CSV result files.

## LightGBM

LightGBM was considered suitable because gradient-boosted trees can model nonlinear relationships among GPS-to-road distance, headings, movement, and network-degree features. Its histogram-based, leaf-wise tree growth is fast on this large tabular candidate set. A limitation is that leaf-wise growth and class weighting can overfit rare candidate patterns; tree-size and child-sample controls matter.

Initial configuration: binary objective with binary log-loss training objective; 300 estimators; learning rate 0.05; 31 leaves; unlimited depth (`max_depth=-1`); 30 minimum child samples; row and feature subsampling 0.8; L2 regularization 1.0; positive-class weight 7.2067; seed 42; four CPU threads. The mean train-vs-CV candidate Macro F1 gap was about 0.011 and the observation Macro F1 gap about 0.016. This is not strong evidence of overfitting, though CV variation across trip folds is material.

## XGBoost baseline

XGBoost was suitable for the same tabular ranking problem because its regularized boosted trees can represent interactions between candidate geometry and trajectory context. The histogram tree method made training practical on CPU. Limitations include training cost, sensitivity to depth/regularization, and probabilities that may need calibration before they are used as HMM emissions.

Initial configuration: `binary:logistic` objective with log-loss metric; histogram trees on CPU; 300 estimators; depth 6; learning rate 0.05; minimum child weight 2; row and feature subsampling 0.8; L2 regularization 1.0; gamma 0; positive-class weight 7.2067; seed 42; four CPU threads. Mean train-vs-CV candidate Macro F1 differed by about 0.013 and observation Macro F1 by about 0.018. Top-1 train scores were higher than CV, and one validation fold was substantially harder, so generalization varies by held-out trips.

## XGBoost Bayesian hyperparameter optimisation

Optuna TPE performed five trials, seed 42, using five-fold trip-grouped cross-validation. The objective was mean per-trip observation-level Macro F1. The model's training objective remained binary log loss. Search ranges were: estimators 200-500 in steps of 100; depth 3-9; learning rate log-uniform 0.02-0.15; minimum child weight log-uniform 1-20; row and column subsampling uniform 0.7-1.0; L2 regularization log-uniform 0.1-10; gamma uniform 0-2.

Best settings: 200 estimators; depth 9; learning rate 0.02236; minimum child weight 3.1157; row subsampling 0.9351; column subsampling 0.9531; L2 regularization 5.0374; gamma 1.5807. The tuning run took 701.4 seconds. Tuned CV observation Macro F1 increased from 0.7042 to 0.7069; CV candidate Macro F1 increased from 0.8427 to 0.8428. The change is modest. Student-test candidate Log Loss became slightly worse, so tuning did not improve every metric.

## HMM/Viterbi and API integration

The integrated predictor uses the HMM/Viterbi decoder shared by Person 3, adapted to Person 1's candidate table and the selected XGBoost probability output. Its hidden states are per-point candidate arcs and projected positions. Emissions are the model's positive-class log probabilities. Directed-network transition costs compare routed distance with GPS straight-line distance, scaled by `beta * sqrt(max(dt, 1))`; small backward motion on the same arc is controlled by `back_tol`. The implementation restarts Viterbi when no transition remains reachable or a GPS jump exceeds the 3 km search limit, and `fill_route` connects short disconnected output gaps through shortest paths.

The decoder was retuned for XGBoost rather than reusing the MLP-tuned configuration. Optuna TPE ran 30 trials (seed 42) on five-fold training OOF probabilities. It maximised mean decoded observation-level accuracy over the 80 training trips. The search used `beta` log-uniform on [0.5, 64] metres and `back_tol` uniform on [0, 40] metres; `w_emit=1`, `sp_limit=3000` metres and `min_prob=0.0001` were fixed. The best values were `beta=5.3161` and `back_tol=6.9179`. OOF mean accuracy was 0.8165 for classifier-only, 0.8708 for the default HMM, and 0.8719 for the tuned HMM. The tuning improvement over the default was small.

On the 10 student-test trips (19,160 observations), classifier-only accuracy averaged 0.8579 and HMM accuracy 0.9033. Mean directed route continuity increased from 0.7587 to 0.9162. Mean route F1 after shortest-path gap filling increased from 0.8606 to 0.9703; HMM length-weighted route F1 averaged 0.9879. HMM accuracy improved on 9 of 10 trips; trip `00000007` decreased from 0.9121 to 0.8929, although continuity rose from 0.9908 to 1.0000. The mean per-trip runtime was 910.8 ms for feature generation, 42.5 ms for classifier scoring, 576.5 ms for Viterbi, and 1.53 s total. These timing values are from the predictor's student-test runs, not network round-trip time.

`outputs/person2/hmm_vs_classifier_student_test_by_trip.csv`, `hmm_vs_classifier_student_test_summary.json`, and `hmm_vs_classifier_student_test.png` contain the per-trip and aggregate comparison. API setup and endpoint instructions are in `PERSON2_INTEGRATION.md`. The FastAPI endpoints were manually exercised in Windows PowerShell 5.1: `/health` returned `ok` with `xgboost`, `/trips` listed the 10 student-test trips, and `/predict` returned trip `00000000` with 2,503 observations. Its aggregate evaluation reported HMM accuracy 0.9600 vs. classifier-only accuracy 0.9417, route precision 0.8995, recall 1.0000, and route F1 0.9471. The two-laptop demo is captured in Figures 6 and 7 of the consolidated report.

## Reproducibility files

- `src/pathsnap_person2/training.py` and `src/pathsnap_person2/oof.py`;
- `src/pathsnap_person2/decoder.py`, `decoder_tuning.py`, `inference.py`, and `api.py`;
- `outputs/person2/model_comparison_results.csv`, `grouped_cv_folds.csv`, `optuna_trials.csv`, `training_summary.json`, and `training.log`;
- `outputs/person2/oof_xgboost.npy` and its metadata JSON, plus decoder tuning logs/trials and student-test HMM comparisons;
- `outputs/person2/models/final_model.joblib`, `final_model.ubj`, `final_model.json`, `lightgbm_baseline.json`, and `xgboost_baseline.json`;
- `outputs/person2/model_comparison.png`, `decoder_tuning_history.png`, and `hmm_vs_classifier_student_test.png`.

All numerical values above are from the executed experiment outputs. The fitted XGBoost `.joblib` artifact is approximately 2.3 MB and the native UBJSON copy is approximately 6.1 MB.
