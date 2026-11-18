"""Grouped-CV training and development evaluation for two baseline rankers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

from .features import FEATURE_COLUMNS, candidate_feature_table
from .io import discover_trip_dirs, load_trip


def _new_models() -> dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=500,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
    }


def _serializable_parameters(model: object) -> dict[str, object]:
    if isinstance(model, Pipeline):
        return {
            "steps": {
                name: {
                    "estimator": estimator.__class__.__name__,
                    "parameters": estimator.get_params(deep=False),
                }
                for name, estimator in model.steps
            }
        }
    return model.get_params(deep=False)


def _tables_for_partition(data_root: Path, partition: str) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    trip_dirs = discover_trip_dirs(data_root, partition)
    for trip_dir in trip_dirs:
        trip = load_trip(trip_dir, partition, include_route=True)
        if trip.route is None or trip.route.size == 0:
            print(f"Skipping {partition}/{trip.trip_id}: no reference route file")
            continue
        if trip.track.size == 0 or trip.nodes.size == 0 or trip.arcs.size == 0:
            print(f"Skipping {partition}/{trip.trip_id}: empty GPS or road-network data")
            continue
        print(f"Building candidates for {partition}/{trip.trip_id}")
        table = candidate_feature_table(trip, align_route=True)
        if not table.empty:
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def _score_rows(table: pd.DataFrame, probabilities: np.ndarray) -> dict[str, float | int]:
    labeled = table["label_available"].astype(bool).to_numpy()
    y_true = table.loc[labeled, "is_target"].astype(int).to_numpy()
    probs = probabilities[labeled]
    if y_true.size == 0:
        return {"candidate_rows": 0, "observations": 0}
    y_pred = (probs >= 0.5).astype(int)
    result: dict[str, float | int] = {
        "candidate_rows": int(y_true.size),
        "candidate_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "candidate_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "candidate_macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "candidate_log_loss": float(log_loss(y_true, np.column_stack((1 - probs, probs)), labels=[0, 1])),
    }

    labeled_rows = table.loc[labeled].copy()
    labeled_rows["score"] = probs
    observation_records: list[dict[str, object]] = []
    for (trip_id, obs_index), group in labeled_rows.groupby(
        ["trip_id", "obs_index"], sort=False
    ):
        best = group.iloc[int(np.argmax(group["score"].to_numpy()))]
        observation_records.append(
            {
                "trip_id": trip_id,
                "obs_index": obs_index,
                "true_arc_id": int(group["target_arc_id"].iloc[0]),
                "pred_arc_id": int(best["arc_id"]),
                "candidate_miss": bool(group["candidate_miss"].iloc[0]),
            }
        )
    observations = pd.DataFrame(observation_records)
    observations["correct"] = observations["true_arc_id"] == observations["pred_arc_id"]
    per_trip_macro = [
        f1_score(
            group["true_arc_id"],
            group["pred_arc_id"],
            average="macro",
            zero_division=0,
        )
        for _, group in observations.groupby("trip_id", sort=False)
    ]
    result.update(
        {
            "observations": int(len(observations)),
            "observation_top1_accuracy": float(observations["correct"].mean()),
            "observation_macro_f1_mean_per_trip": float(np.mean(per_trip_macro)),
            "candidate_miss_rate": float(observations["candidate_miss"].mean()),
        }
    )
    return result


def train_and_evaluate(data_root: Path, output_dir: Path, folds: int = 5) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    train_table = _tables_for_partition(data_root, "train")
    student_table = _tables_for_partition(data_root, "student_test")
    if train_table.empty or student_table.empty:
        raise ValueError("Need labeled train and student_test trips to train and compare baselines")

    train_labeled = train_table[train_table["label_available"]].copy()
    trainable = train_labeled[~train_labeled["candidate_miss"]].copy()
    train_trip_ids = np.asarray(sorted(train_labeled["trip_id"].unique()))
    if train_trip_ids.size < 2 or trainable["trip_id"].nunique() < 2:
        raise ValueError("At least two labeled training trips need candidate hits for grouped CV")
    splitter = GroupKFold(n_splits=min(folds, train_trip_ids.size))
    x_all = trainable[FEATURE_COLUMNS]
    y_all = trainable["is_target"].astype(int)
    student_eval = student_table[student_table["label_available"]].copy()
    if student_eval.empty:
        raise ValueError("Student-testing trips do not have reference routes for evaluation")
    x_student = student_eval[FEATURE_COLUMNS]

    fold_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    for model_name, prototype in _new_models().items():
        print(f"Grouped cross-validation: {model_name}")
        for fold_id, (fit_trip_idx, valid_trip_idx) in enumerate(
            splitter.split(train_trip_ids, groups=train_trip_ids), start=1
        ):
            fit_trip_ids = train_trip_ids[fit_trip_idx]
            valid_trip_ids = train_trip_ids[valid_trip_idx]
            fold_train = trainable[trainable["trip_id"].isin(fit_trip_ids)]
            if fold_train["is_target"].nunique() < 2:
                raise ValueError(
                    f"Fold {fold_id} has only one candidate-label class; revise the trip folds"
                )
            model = clone(prototype)
            started = time.perf_counter()
            model.fit(fold_train[FEATURE_COLUMNS], fold_train["is_target"].astype(int))
            fit_seconds = time.perf_counter() - started
            validation_rows = train_labeled[
                train_labeled["trip_id"].isin(valid_trip_ids)
            ].copy()
            valid_scores = model.predict_proba(validation_rows[FEATURE_COLUMNS])[:, 1]
            metrics = _score_rows(validation_rows, valid_scores)
            fold_records.append(
                {
                    "model": model_name,
                    "fold": fold_id,
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
            )

        model = clone(prototype)
        started = time.perf_counter()
        model.fit(x_all, y_all)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        student_scores = model.predict_proba(x_student)[:, 1]
        inference_seconds = time.perf_counter() - started
        student_metrics = _score_rows(student_eval, student_scores)
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        metadata = {
            "model_name": model_name,
            "feature_columns": FEATURE_COLUMNS,
            "parameters": _serializable_parameters(model),
            "training_trip_count": int(trainable["trip_id"].nunique()),
            "training_candidate_rows": int(len(trainable)),
            "candidate_miss_training_observations_excluded": int(
                train_labeled.groupby(["trip_id", "obs_index"])["candidate_miss"].first().sum()
            ),
            "artifact": model_path.name,
            "random_seed": 42,
        }
        (models_dir / f"{model_name}.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        summary_records.append(
            {
                "model": model_name,
                "fit_seconds": fit_seconds,
                "inference_seconds_student_test": inference_seconds,
                **student_metrics,
            }
        )

    folds_frame = pd.DataFrame(fold_records)
    folds_frame.to_csv(output_dir / "baseline_cv_folds.csv", index=False)
    summary_frame = pd.DataFrame(summary_records)
    cv_metrics = [
        "candidate_macro_f1",
        "candidate_log_loss",
        "observation_top1_accuracy",
        "observation_macro_f1_mean_per_trip",
        "candidate_miss_rate",
    ]
    fold_summary = folds_frame.groupby("model")[cv_metrics].agg(["mean", "std"])
    fold_summary.columns = [f"cv_{metric}_{stat}" for metric, stat in fold_summary.columns]
    fold_summary = fold_summary.reset_index()
    summary_frame = summary_frame.merge(fold_summary, on="model", how="left")
    summary_frame.to_csv(output_dir / "baseline_student_test_results.csv", index=False)
    plot = summary_frame.set_index("model")[
        ["observation_top1_accuracy", "observation_macro_f1_mean_per_trip"]
    ]
    ax = plot.plot(kind="bar", figsize=(8, 5), ylim=(0, 1), rot=0)
    ax.set_title("Person 1 baseline models: student-testing performance")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.legend(["Top-1 accuracy", "Macro F1 (mean per trip)"])
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "baseline_comparison.png", dpi=160, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Wrote {output_dir / 'baseline_cv_folds.csv'}")
    print(f"Wrote {output_dir / 'baseline_student_test_results.csv'}")
    return summary_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two grouped-CV PathSnap baselines.")
    parser.add_argument("--data-root", required=True, type=Path, help="Extracted folder containing train/ and student_test/.")
    parser.add_argument("--out", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    results = train_and_evaluate(args.data_root, args.out, args.folds)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
