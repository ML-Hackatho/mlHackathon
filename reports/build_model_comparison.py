"""Build the cross-workstream comparison table and figure from saved results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def main() -> None:
    p1 = pd.read_csv(ROOT / "outputs/baselines/baseline_student_test_results.csv").set_index("model")
    p2 = pd.read_csv(ROOT / "outputs/person2/model_comparison_results.csv").set_index("model")
    continuity = pd.read_csv(OUT / "model_route_continuity_student_test.csv").set_index("model")
    mlp = json.loads((ROOT / "person3_route_decoding_ui/results/mlp_metrics.json").read_text(encoding="utf-8"))
    mlp_hmm = pd.read_csv(ROOT / "person3_route_decoding_ui/results/hmm_vs_classifier_summary.csv", index_col=0)["mean"]
    xgb_hmm = json.loads((ROOT / "outputs/person2/hmm_vs_classifier_student_test_summary.json").read_text(encoding="utf-8"))

    rows: list[dict[str, object]] = []
    for key, label in (("logistic_regression", "Logistic Regression"), ("random_forest", "Random Forest")):
        row = p1.loc[key]
        rows.append({
            "model": label,
            "cv_candidate_macro_f1_mean": row["cv_candidate_macro_f1_mean"],
            "cv_candidate_macro_f1_std": row["cv_candidate_macro_f1_std"],
            "cv_observation_top1_mean": row["cv_observation_top1_accuracy_mean"],
            "cv_observation_top1_std": row["cv_observation_top1_accuracy_std"],
            "student_candidate_macro_f1": row["candidate_macro_f1"],
            "student_observation_top1": row["observation_top1_accuracy"],
            "student_candidate_log_loss": row["candidate_log_loss"],
            "student_test_route_continuity_classifier_only": continuity.loc[key, "student_test_route_continuity_mean_per_trip"],
            "student_test_route_continuity_hmm": np.nan,
            "inference_ms_per_1000_observations": row["inference_seconds_student_test"] / row["observations"] * 1_000_000,
        })

    for key, label in (("lightgbm", "LightGBM"), ("xgboost", "XGBoost baseline"), ("xgboost_tuned", "XGBoost tuned")):
        row = p2.loc[key]
        rows.append({
            "model": label,
            "cv_candidate_macro_f1_mean": row["cv_candidate_macro_f1_mean"],
            "cv_candidate_macro_f1_std": row["cv_candidate_macro_f1_std"],
            "cv_observation_top1_mean": row["cv_observation_top1_accuracy_mean"],
            "cv_observation_top1_std": row["cv_observation_top1_accuracy_std"],
            "student_candidate_macro_f1": row["student_candidate_macro_f1"],
            "student_observation_top1": row["student_observation_top1_accuracy"],
            "student_candidate_log_loss": row["student_candidate_log_loss"],
            "student_test_route_continuity_classifier_only": continuity.loc[key, "student_test_route_continuity_mean_per_trip"],
            "student_test_route_continuity_hmm": xgb_hmm["continuity_hmm"] if key == "xgboost_tuned" else np.nan,
            "inference_ms_per_1000_observations": row["student_test_predict_seconds"] / row["student_observations"] * 1_000_000,
        })

    rows.append({
        "model": "MLP neural network",
        "cv_candidate_macro_f1_mean": mlp["cv_macro_f1_mean"],
        "cv_candidate_macro_f1_std": mlp["cv_macro_f1_std"],
        "cv_observation_top1_mean": mlp["cv_top1_mean"],
        "cv_observation_top1_std": mlp["cv_top1_std"],
        "student_candidate_macro_f1": mlp["test_macro_f1"],
        "student_observation_top1": mlp["test_top1"],
        "student_candidate_log_loss": mlp["test_log_loss"],
        "student_test_route_continuity_classifier_only": continuity.loc["mlp", "student_test_route_continuity_mean_per_trip"],
        "student_test_route_continuity_hmm": mlp_hmm["continuity_hmm"],
        "inference_ms_per_1000_observations": mlp["inference_ms_per_1000_points"],
    })

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "model_comparison_combined.csv", index=False)
    make_figure(result, OUT / "model_comparison_combined.png")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Saved {OUT / 'model_comparison_combined.csv'}")
    print(f"Saved {OUT / 'model_comparison_combined.png'}")


def make_figure(results: pd.DataFrame, path: Path) -> None:
    labels = results["model"].tolist()
    positions = np.arange(len(results))
    colors = ["#64748b" if not name.startswith("XGBoost tuned") else "#ea580c" for name in labels]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    fig.suptitle("PathSnap model comparison on grouped CV and student-test trips", fontsize=15, weight="bold")

    ax = axes[0, 0]
    ax.bar(positions, results["cv_candidate_macro_f1_mean"], yerr=results["cv_candidate_macro_f1_std"],
           capsize=3, color=colors, edgecolor="white")
    ax.set_title("5-fold grouped CV: candidate Macro F1")
    ax.set_ylabel("Mean ± SD")
    ax.set_ylim(0, 1)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    width = 0.36
    ax.bar(positions - width / 2, results["student_candidate_macro_f1"], width,
           label="Candidate Macro F1", color="#2563eb")
    ax.bar(positions + width / 2, results["student_observation_top1"], width,
           label="Observation Top-1", color="#16a34a")
    ax.set_title("Student-test classification metrics")
    ax.set_ylim(0, 1)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 0]
    ax.bar(positions, results["student_candidate_log_loss"], color="#7c3aed")
    ax.set_title("Student-test candidate log loss (lower is better)")
    ax.set_ylabel("Log loss")
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    ax.bar(positions - width / 2, results["student_test_route_continuity_classifier_only"], width,
           label="Classifier only", color="#0ea5e9")
    has_hmm = results["student_test_route_continuity_hmm"].notna()
    ax.bar(positions[has_hmm] + width / 2, results.loc[has_hmm, "student_test_route_continuity_hmm"], width,
           label="After tuned HMM", color="#f97316")
    ax.set_title("Directed route continuity")
    ax.set_ylim(0, 1)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
