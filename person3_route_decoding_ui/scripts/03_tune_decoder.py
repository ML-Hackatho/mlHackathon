"""Bayesian optimisation of the HMM decoder hyperparameters on TRAINING trips only.

Search strategy : Optuna TPE sampler (Tree-structured Parzen Estimator), seed 42.
                  No grid or random search is used.
Emissions       : the MLP's out-of-fold (OOF) probabilities from 5-fold GroupKFold, so each
                  training trip is decoded with a model that never saw that trip.
Objective       : maximise mean observation-level accuracy of the decoded route over the
                  80 training trips (Person 1's aligned labels; candidate-miss points count as
                  errors). Student-test / evaluator trips are not used.
Search space    :
  beta      log-uniform [0.5, 64] m   transition scale |d_route - d_gc| / (beta*sqrt(dt))
  back_tol  uniform     [0, 40]   m   backward jitter allowed on the same arc
  w_emit is fixed to 1.0: the Viterbi argmax depends on w_emit and beta only through
  their product (multiplying every score by a constant does not change the argmax),
  so searching both would be redundant.

Outputs: results/decoder_bayesopt_trials.csv, figures/decoder_bayesopt_history.png,
         models/decoder_config.json
"""
import json
import os
import sys
import time

import matplotlib
import numpy as np
import optuna
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.hmm import HMMConfig, NetworkDistance, viterbi_decode  # noqa: E402
from pathsnap.team import load_for_prediction, point_targets  # noqa: E402
from pathsnap.metrics import pointwise_accuracy, route_continuity  # noqa: E402

MODEL = os.environ.get("PATHSNAP_MODEL", "mlp")
N_TRIALS = int(os.environ.get("PATHSNAP_TRIALS", "30"))
DEFAULT = HMMConfig()          # untuned starting point: beta=8, w_emit=1, back_tol=12


def load_cache():
    tr = pd.read_parquet(os.path.join(C.PROCESSED, "train_candidates.parquet"))
    tr["p"] = np.load(os.path.join(C.RESULTS, f"oof_{MODEL}.npy"))
    cache = []
    for tid, g in tr.groupby("trip"):
        trip = load_for_prediction(os.path.join(C.TRAIN_DIR, tid))[1]   # geometry only
        lab = point_targets(g, trip.n_points)                         # Person 1's aligned labels
        cache.append((trip, g, lab, NetworkDistance(trip, DEFAULT.sp_limit)))
    return cache


def decode_all(cache, cfg):
    acc, cont, br = [], [], 0
    for trip, g, lab, net in cache:
        pred, b = viterbi_decode(trip, g, g["p"].to_numpy(), cfg, net)
        acc.append(pointwise_accuracy(pred, lab))
        cont.append(route_continuity(pred, trip.arcs))
        br += b
    return float(np.mean(acc)), float(np.mean(cont)), br


if __name__ == "__main__":
    cache = load_cache()
    # classifier-only baseline on the same trips
    accs = []
    for trip, g, lab, _ in cache:
        best = g.loc[g.groupby("point")["p"].idxmax()]
        clf = np.full(trip.n_points, -1)
        clf[best["point"]] = best["arc"]
        accs.append(pointwise_accuracy(clf, lab))
    clf_acc = float(np.mean(accs))
    t0 = time.time()
    base_acc, base_cont, _ = decode_all(cache, DEFAULT)
    print(f"classifier-only accuracy {clf_acc:.4f} | untuned decoder (beta=8, back_tol=12) "
          f"accuracy {base_acc:.4f} continuity {base_cont:.4f} ({time.time() - t0:.0f}s)")

    def objective(trial):
        cfg = HMMConfig(beta=trial.suggest_float("beta", 0.5, 64.0, log=True),
                        back_tol=trial.suggest_float("back_tol", 0.0, 40.0), w_emit=1.0)
        acc, cont, br = decode_all(cache, cfg)
        trial.set_user_attr("continuity", cont)
        trial.set_user_attr("chain_breaks", br)
        return acc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="hmm_decoder",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED, n_startup_trials=8))
    study.enqueue_trial({"beta": DEFAULT.beta, "back_tol": DEFAULT.back_tol})   # trial 0 = untuned
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS,
                   callbacks=[lambda s, t: print(f"trial {t.number:2d}: beta={t.params['beta']:.2f} "
                                                 f"back_tol={t.params['back_tol']:.1f} acc={t.value:.4f} "
                                                 f"(best {s.best_value:.4f})")])
    secs = time.time() - t0

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    df["duration_s"] = df.pop("duration").dt.total_seconds()
    df.to_csv(os.path.join(C.RESULTS, "decoder_bayesopt_trials.csv"), index=False)
    best = study.best_params
    tuned_acc, tuned_cont, tuned_br = decode_all(cache, HMMConfig(beta=best["beta"], back_tol=best["back_tol"]))
    summary = pd.DataFrame([
        {"setting": "classifier only", "beta": None, "back_tol": None, "train_oof_accuracy": clf_acc,
         "train_oof_continuity": None},
        {"setting": "HMM untuned", "beta": DEFAULT.beta, "back_tol": DEFAULT.back_tol,
         "train_oof_accuracy": base_acc, "train_oof_continuity": base_cont},
        {"setting": "HMM Bayesian-tuned", "beta": best["beta"], "back_tol": best["back_tol"],
         "train_oof_accuracy": tuned_acc, "train_oof_continuity": tuned_cont},
    ])
    summary.to_csv(os.path.join(C.RESULTS, "decoder_tuning_summary.csv"), index=False)
    print(summary.round(4).to_string(index=False))
    print(f"{N_TRIALS} trials in {secs:.0f}s; best {best}")

    vals = df["value"].to_numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(df["number"], vals, "o", alpha=.6, label="trial")
    ax[0].plot(df["number"], np.maximum.accumulate(vals), "-", color="tab:red", label="best so far")
    ax[0].axhline(clf_acc, ls="--", color="gray", label="classifier only")
    ax[0].set_xlabel("trial")
    ax[0].set_ylabel("mean decoded accuracy (80 train trips)")
    ax[0].set_title("TPE optimisation history")
    ax[0].legend(fontsize=8)
    sc = ax[1].scatter(df["params_beta"], df["params_back_tol"], c=vals, cmap="viridis")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("beta (m, log)")
    ax[1].set_ylabel("back_tol (m)")
    ax[1].set_title("Explored hyperparameters")
    fig.colorbar(sc, ax=ax[1], label="accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "decoder_bayesopt_history.png"), dpi=130)
    plt.close(fig)

    cfg = {"model_name": MODEL,
           "selected_by": f"Optuna TPE, {N_TRIALS} trials, objective = mean decoded accuracy on 80 "
                          f"training trips using OOF probabilities",
           "hmm": {"beta": float(best["beta"]), "w_emit": 1.0, "back_tol": float(best["back_tol"]),
                   "sp_limit": DEFAULT.sp_limit, "min_prob": DEFAULT.min_prob}}
    json.dump(cfg, open(os.path.join(C.MODELS, "decoder_config.json"), "w"), indent=2)
