"""Classifier-only vs classifier + HMM/Viterbi on the 10 student-testing trips.

Uses the frozen model and decoder config (no tuning on these trips).
Metrics per trip:
  accuracy           observation-level accuracy (reliable points)
  continuity         fraction of consecutive distinct predicted arcs that are directly connected
  route P/R/F1       predicted arc set (after shortest-path gap filling) vs reference route
  near_gap_accuracy  accuracy on points within 3 observations of a GPS gap > 5 s
Outputs: results/hmm_vs_classifier_per_trip.csv, results/hmm_vs_classifier_summary.csv, figures/hmm_*.png
"""
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathsnap import config as C  # noqa: E402
from pathsnap.inference import RoutePredictor, score_against_reference  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.metrics import collapse, route_overlap  # noqa: E402

MODEL = os.environ.get("PATHSNAP_MODEL", "mlp")


def near_gap_mask(t, k=3, gap=5):
    g = np.flatnonzero(np.diff(t) > gap)
    m = np.zeros(len(t), bool)
    for i in g:
        m[max(0, i - k + 1):i + k + 1] = True
    return m


if __name__ == "__main__":
    pred = RoutePredictor(os.path.join(C.MODELS, f"{MODEL}.joblib"),
                          os.path.join(C.MODELS, "decoder_config.json"))
    rows = []
    for tid in list_trips(C.STUDENT_TEST_DIR):
        d = os.path.join(C.STUDENT_TEST_DIR, tid)
        out, trip = pred.predict(d, include_network=False)
        ev, lab = score_against_reference(d, out, partition="student_test")
        # extra analysis (evaluation only)
        rel = lab["reliable"]
        hmm = np.asarray(out["predicted_arcs_per_point"])
        clf = np.asarray(out["classifier_only_arcs_per_point"])
        ng = near_gap_mask(trip.t) & rel
        route = np.loadtxt(os.path.join(d, f"{tid}.route"), dtype=int, ndmin=1)
        ov_clf = route_overlap(collapse(clf), route, trip.arc_len)
        rows.append({
            "trip": tid, "points": trip.n_points, "gps_gaps_over_5s": out["decoder"]["gps_gaps_over_5s"],
            "chain_breaks": out["decoder"]["chain_breaks"],
            "accuracy_classifier": ev["accuracy_classifier_only"], "accuracy_hmm": ev["accuracy_hmm"],
            "continuity_classifier": out["decoder"]["route_continuity_classifier_only"],
            "continuity_hmm": out["decoder"]["route_continuity_hmm"],
            "route_f1_classifier": ov_clf["route_f1"], "route_f1_hmm": ev["route_f1"],
            "route_f1_hmm_without_gap_fill": route_overlap(collapse(hmm), route, trip.arc_len)["route_f1"],
            "route_len_f1_hmm": ev["route_length_f1"],
            "near_gap_points": int(ng.sum()),
            "near_gap_acc_classifier": float((clf[ng] == lab["label_arc"][ng]).mean()) if ng.any() else np.nan,
            "near_gap_acc_hmm": float((hmm[ng] == lab["label_arc"][ng]).mean()) if ng.any() else np.nan,
            "viterbi_ms": out["timing_ms"]["viterbi"], "total_ms": out["timing_ms"]["total"],
        })
        r = rows[-1]
        print(f"{tid}: acc {r['accuracy_classifier']:.3f}->{r['accuracy_hmm']:.3f}  "
              f"cont {r['continuity_classifier']:.3f}->{r['continuity_hmm']:.3f}  "
              f"routeF1 {r['route_f1_classifier']:.3f}->{r['route_f1_hmm']:.3f}  breaks {r['chain_breaks']}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(C.RESULTS, "hmm_vs_classifier_per_trip.csv"), index=False)
    summ = df.drop(columns=["trip"]).mean(numeric_only=True)
    summ.to_csv(os.path.join(C.RESULTS, "hmm_vs_classifier_summary.csv"), header=["mean"])
    print(summ.round(4).to_string())

    # untuned (default HMMConfig) vs Bayesian-tuned decoder on the same student-test trips
    import json
    import time
    from dataclasses import asdict
    from pathsnap.hmm import HMMConfig
    untuned_path = os.path.join(C.MODELS, "decoder_config_untuned.json")
    json.dump({"model_name": MODEL, "hmm": asdict(HMMConfig())}, open(untuned_path, "w"), indent=2)
    comp = []
    for name, cfg_path in (("untuned (beta=8, back_tol=12)", untuned_path),
                           ("Bayesian-tuned", os.path.join(C.MODELS, "decoder_config.json"))):
        pr = RoutePredictor(os.path.join(C.MODELS, f"{MODEL}.joblib"), cfg_path)
        acc, cont, f1, ms = [], [], [], []
        for tid in list_trips(C.STUDENT_TEST_DIR):
            d = os.path.join(C.STUDENT_TEST_DIR, tid)
            out, trip = pr.predict(d, include_network=False)
            ev, _ = score_against_reference(d, out, partition="student_test")
            acc.append(ev["accuracy_hmm"]); f1.append(ev["route_f1"])
            cont.append(out["decoder"]["route_continuity_hmm"]); ms.append(out["timing_ms"]["viterbi"])
        comp.append({"decoder": name, **pr.cfg.__dict__, "student_test_accuracy": np.mean(acc),
                     "student_test_continuity": np.mean(cont), "student_test_route_f1": np.mean(f1),
                     "viterbi_ms_per_trip": np.mean(ms)})
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(C.RESULTS, "decoder_untuned_vs_tuned_student_test.csv"), index=False)
    print(comp.round(4).to_string(index=False))

    x = np.arange(len(df))
    for metric, label in (("accuracy", "observation-level accuracy"), ("continuity", "route continuity"),
                          ("route_f1", "route F1 (arc set)")):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.bar(x - 0.2, df[f"{metric}_classifier"], 0.4, label="classifier only", color="#9ecae1")
        ax.bar(x + 0.2, df[f"{metric}_hmm"], 0.4, label="classifier + HMM/Viterbi", color="#08519c")
        ax.set_xticks(x, df["trip"], rotation=45)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.set_title(f"Student-test trips: {label}")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(C.FIGURES, f"hmm_vs_classifier_{metric}.png"), dpi=130)
        plt.close(fig)
