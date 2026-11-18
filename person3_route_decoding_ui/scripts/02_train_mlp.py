"""Person 3's model: Multi-Layer Perceptron candidate classifier.

Task: binary classification of (GPS point, candidate arc) pairs -> P(candidate is the reference arc).
Features, candidates and labels: Person 1's shared pipeline (15 FEATURES, target is_target).
Protocol (same as the team's):
  * fit on the 80 training trips (every TRAIN_POINT_STRIDE-th point, candidate-miss points excluded)
  * 5-fold GroupKFold, grouped by trip id -> out-of-fold (OOF) probabilities for ALL training rows
    (these OOF probabilities are later used to tune the HMM decoder without touching test data)
  * refit on all training rows, evaluate on the 10 student-testing trips.

Outputs: models/mlp.joblib, results/mlp_metrics.json, results/mlp_cv_folds.csv,
         results/oof_mlp.npy, results/test_probs_mlp.npy, figures/mlp_*.png
"""
import json
import os
import sys
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.team import FEATURES, load_for_prediction  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_continuity, top1_accuracy  # noqa: E402

PARAMS = dict(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=1e-4,
              learning_rate_init=1e-3, batch_size=512, max_iter=60, early_stopping=True,
              validation_fraction=0.1, n_iter_no_change=6, random_state=C.SEED)


def make():
    return make_pipeline(StandardScaler(), MLPClassifier(**PARAMS))


def fit_rows(df):
    """Every TRAIN_POINT_STRIDE-th point; candidate-miss points are excluded from fitting
    (they have no positive candidate) - Person 1's rule."""
    return df[(df["point"] % C.TRAIN_POINT_STRIDE == 0) & df["reliable"] & ~df["candidate_miss"]]


def X(df):
    return df[FEATURES].astype(np.float32)


def scores(df, p):
    m = candidate_metrics(df["y"].to_numpy(), p)
    m["top1_accuracy"] = top1_accuracy(df.assign(p=p))
    return m


def continuity(df, p, root):
    vals = []
    d = df.assign(p=p)
    for tid, g in d.groupby("trip"):
        arcs = load_for_prediction(os.path.join(root, tid))[1].arcs
        best = g.loc[g.groupby("point")["p"].idxmax()].sort_values("point")
        vals.append(route_continuity(best["arc"].to_numpy(), arcs))
    return float(np.mean(vals))


if __name__ == "__main__":
    tr = pd.read_parquet(os.path.join(C.PROCESSED, "train_candidates.parquet"))
    te = pd.read_parquet(os.path.join(C.PROCESSED, "student_test_candidates.parquet"))
    print(f"train rows {len(tr)}, student-test rows {len(te)}, positives {tr['y'].mean():.3f}")

    oof = np.full(len(tr), np.nan)
    folds = []
    for k, (i_tr, i_va) in enumerate(GroupKFold(C.N_FOLDS).split(tr, groups=tr["trip"])):
        dtr, dva = fit_rows(tr.iloc[i_tr]), tr.iloc[i_va]
        m = make()
        t0 = time.time()
        m.fit(X(dtr), dtr["y"])
        tfit = time.time() - t0
        oof[i_va] = m.predict_proba(X(dva))[:, 1]
        mt = scores(dtr, m.predict_proba(X(dtr))[:, 1])
        mv = scores(dva, oof[i_va])
        folds.append({"fold": k, "val_trips": dva["trip"].nunique(), "epochs": m[-1].n_iter_,
                      "fit_seconds": tfit, **{f"train_{a}": b for a, b in mt.items()},
                      **{f"val_{a}": b for a, b in mv.items()}})
        print(f"fold {k}: train F1={mt['macro_f1']:.4f} top1={mt['top1_accuracy']:.4f} | "
              f"val F1={mv['macro_f1']:.4f} top1={mv['top1_accuracy']:.4f} ll={mv['log_loss']:.4f} "
              f"({m[-1].n_iter_} epochs, {tfit:.0f}s)")
    folds = pd.DataFrame(folds)
    folds.to_csv(os.path.join(C.RESULTS, "mlp_cv_folds.csv"), index=False)
    np.save(os.path.join(C.RESULTS, "oof_mlp.npy"), oof)

    dall = fit_rows(tr)
    m = make()
    t0 = time.time()
    m.fit(X(dall), dall["y"])
    tfit = time.time() - t0
    t0 = time.time()
    p_te = m.predict_proba(X(te))[:, 1]
    tinf = time.time() - t0
    np.save(os.path.join(C.RESULTS, "test_probs_mlp.npy"), p_te)
    joblib.dump(m, os.path.join(C.MODELS, "mlp.joblib"))
    st = scores(te, p_te)
    n_pts = te.groupby(["trip", "point"]).ngroups
    res = {
        "model": "MLP", "family": "Neural network",
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in PARAMS.items()},
        "loss": "binary cross-entropy (log loss) + L2 penalty alpha, optimised with Adam",
        "cv_macro_f1_mean": folds["val_macro_f1"].mean(), "cv_macro_f1_std": folds["val_macro_f1"].std(),
        "cv_top1_mean": folds["val_top1_accuracy"].mean(), "cv_top1_std": folds["val_top1_accuracy"].std(),
        "cv_log_loss_mean": folds["val_log_loss"].mean(),
        "train_macro_f1_mean": folds["train_macro_f1"].mean(), "train_top1_mean": folds["train_top1_accuracy"].mean(),
        "test_macro_precision": st["macro_precision"], "test_macro_recall": st["macro_recall"],
        "test_macro_f1": st["macro_f1"], "test_log_loss": st["log_loss"], "test_top1": st["top1_accuracy"],
        "test_route_continuity_classifier_only": continuity(te, p_te, C.STUDENT_TEST_DIR),
        "fit_seconds_full": tfit, "epochs_full": int(m[-1].n_iter_),
        "inference_ms_per_1000_points": tinf * 1000 / n_pts * 1000,
    }
    json.dump(res, open(os.path.join(C.RESULTS, "mlp_metrics.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

    mlp = m[-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, label="training loss (log loss)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training log loss")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(mlp.validation_scores_) + 1), mlp.validation_scores_,
             color="tab:orange", label="validation accuracy (candidate pairs, 10% internal hold-out)")
    ax2.set_ylabel("validation accuracy")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.78), fontsize=8)
    ax.set_title("MLP training history (final fit on 80 training trips)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "mlp_training_history.png"), dpi=130)
    plt.close(fig)

    cm = confusion_matrix(te["y"], (p_te >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ConfusionMatrixDisplay(cm, display_labels=["other arc", "reference arc"]).plot(ax=ax, colorbar=False)
    ax.set_title("MLP - student test (thr 0.5)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "mlp_confusion_student_test.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = folds["fold"]
    ax.plot(x, folds["train_macro_f1"], "o-", label="train macro F1")
    ax.plot(x, folds["val_macro_f1"], "o-", label="validation macro F1")
    ax.plot(x, folds["val_top1_accuracy"], "s--", label="validation Top-1")
    ax.set_xlabel("CV fold (grouped by trip)")
    ax.set_xticks(x)
    ax.legend(fontsize=8)
    ax.set_title("MLP: train vs validation per fold")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIGURES, "mlp_cv_folds.png"), dpi=130)
    plt.close(fig)
